import argparse
import ast
import json
import numpy as np
import os
import pandas as pd
import torch
import webdataset as wds

from collections import defaultdict
from tqdm import tqdm
from utils.utils import build_benthicnet_hierarchy, build_fathomnet_hierarchy, \
    annotation_to_bin_vector, build_inat_hierarchy, get_inat_annotation_path

def build_shard_index(shard_dir):
    index = {}
    pt_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".pt"))
    for filename in tqdm(pt_files, desc="  Indexing shards"):
        shard = torch.load(
            os.path.join(shard_dir, filename), map_location="cpu"
        )
        for key in shard:
            index[key] = filename
        del shard
    return index

def write_webdataset_from_shards(
        output_dir,
        partition,
        annotations,
        img_shard_dir,
        img_index,
        txt_shard_dir,
        samples_per_tar=10000
        ):
    print("  Building text shard index...")
    txt_index = build_shard_index(txt_shard_dir)
    print(f"  Indexed {len(txt_index)} text embeddings "
          f"across {len(set(txt_index.values()))} shards.")

    pair_to_keys = defaultdict(list)
    skipped = 0
    for key in annotations:
        if key not in img_index or key not in txt_index:
            skipped += 1
            continue
        pair_to_keys[(img_index[key], txt_index[key])].append(key)

    sorted_pairs = sorted(pair_to_keys.keys())

    tar_idx = 0
    written = 0
    written_in_tar = 0
    sink = None

    current_img_name = None
    current_img_shard = None

    total_samples = sum(len(v) for v in pair_to_keys.values())
    pbar = tqdm(total=total_samples, desc="  Writing samples")

    try:
        for img_shard_name, txt_shard_name in sorted_pairs:
            if img_shard_name != current_img_name:
                if current_img_shard is not None:
                    del current_img_shard
                current_img_shard = torch.load(
                    os.path.join(img_shard_dir, img_shard_name),
                    map_location="cpu"
                )
                current_img_name = img_shard_name

            txt_shard = torch.load(
                os.path.join(txt_shard_dir, txt_shard_name),
                map_location="cpu"
            )
            for key in pair_to_keys[(img_shard_name, txt_shard_name)]:
                if sink is None or written_in_tar >= samples_per_tar:
                    if sink is not None:
                        sink.close()
                    tar_path = os.path.join(
                        output_dir, f"{partition}_{tar_idx:06d}.tar"
                    )
                    sink = wds.TarWriter(tar_path)
                    tar_idx += 1
                    written_in_tar = 0

                clean_key = "__DOT__".join(key.split(".")[:-1])
                sample = {
                    "__key__": str(clean_key),
                    "img_embedding.pth": current_img_shard[key].clone(),
                    "txt_embedding.pth": txt_shard[key].clone(),
                    "label.npy": annotations[key],
                }
                sink.write(sample)
                written += 1
                written_in_tar += 1
                pbar.update(1)

            del txt_shard

    finally:
        pbar.close()
        if sink is not None:
            sink.close()

    print(
        f"  Wrote {written} samples across {tar_idx} tar files. "
        f"Skipped {skipped} (missing embeddings)."
    )


def gen_args():
    parser = argparse.ArgumentParser(
        description="Convert embedding shards to WebDataset tar files"
    )
    parser.add_argument(
        "--dataset_domain", type=str,
        help="Dataset domain (e.g., 'sub' for substrates or 'bio' for biota)"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="benthicnet",
        help="Dataset name (e.g., 'benthicnet')"
    )
    parser.add_argument(
        "--partition", type=str, default=None,
        help="Process only this partition (e.g., 'train'). "
             "If omitted, all partitions are processed."
    )
    parser.add_argument(
        "--text_model", type=str, default="Qwen3-Embedding-8B",
        help="Text embedding model name (default: 'Qwen3-Embedding-8B')"
    )
    parser.add_argument(
        "--emb_type", type=str, default="full",
        help="Embedding type (e.g., 'ltp' or 'full')"
    )
    parser.add_argument(
        "--samples_per_tar", type=int, default=10000,
        help="Number of samples per output tar file (default: 10000)"
    )
    parser.add_argument(
        "--bin_rep", action="store_true",
        help="Convert annotations to binary representation"
    )

    return parser.parse_args()


def main():
    args = gen_args()

    dataset_name = args.dataset_name
    data_domain = args.dataset_domain
    text_model = args.text_model
    emb_type = args.emb_type
    bin_rep = args.bin_rep
    samples_per_tar = args.samples_per_tar

    data_domain_full = "substrates" if data_domain == "sub" \
        else "biota" if data_domain == "bio" else data_domain

    if dataset_name == "benthicnet":
        annotation_col = "catami_substrate" if data_domain == "sub" else "catami_biota"
        annotations_df = pd.read_csv(
            f"./datasets/{dataset_name}/{dataset_name}_{data_domain_full}.csv"
        )
        hierarchy_info = build_benthicnet_hierarchy(annotations_df[annotation_col])
    elif dataset_name == "fathomnet-2025":
        annotation_col = "annotation"
        annotations_df = pd.read_csv(
            f"./datasets/{dataset_name}/{data_domain}_annotations.csv"
        )
        full_data_df = pd.read_csv(
            f"./datasets/{dataset_name}/full_annotations.csv"
        )
        index_to_taxon = json.load(
            open(f"./datasets/{dataset_name}/train/index_to_taxon.json", "r")
        )
        taxon_to_index = json.load(
            open(f"./datasets/{dataset_name}/train/taxon_to_index.json", "r")
        )
        hierarchy_info = build_fathomnet_hierarchy(
            full_data_df, idx_to_label=index_to_taxon, label_to_idx=taxon_to_index
        )
    else:
        annotations_df = pd.read_csv(
            f"./datasets/{dataset_name}/inat_annotations.csv"
        )
        full_data_df = annotations_df
        hierarchy_info = build_inat_hierarchy(full_data_df)

        annotation_col = "inat_path"
        annotations_df["inat_path"] = annotations_df.apply(
            get_inat_annotation_path, axis=1
        )
        annotations_df["image_path"] = annotations_df["path"]

    node_id_dict = hierarchy_info["node_id_dict"]
    descendant_matrix = hierarchy_info["descendant_matrix"]

    all_partitions = sorted(annotations_df["partition"].unique())
    if args.partition is not None:
        if args.partition not in all_partitions:
            raise ValueError(
                f"Partition '{args.partition}' not found. "
                f"Available: {all_partitions}"
            )
        partitions = [args.partition]
    else:
        partitions = all_partitions

    print(f"Partitions to process: {partitions}")

    if dataset_name == "inat":
        output_dir = f"./datasets/{dataset_name}/feature_tars/{emb_type}"
    else:
        output_dir = f"./datasets/{dataset_name}/feature_tars/{emb_type}/{data_domain}"
    os.makedirs(output_dir, exist_ok=True)

    for partition in partitions:
        print(f"\n{'=' * 60}")
        print(f"Processing partition: {partition}")
        print(f"{'=' * 60}")

        partition_df = annotations_df[
            annotations_df["partition"] == partition
        ].reset_index(drop=True)

        if bin_rep:
            partition_df["bin_annotation"] = partition_df[annotation_col].apply(
                lambda x: annotation_to_bin_vector(
                    x, node_id_dict, descendant_matrix
                )
            )
            annotations_dict = dict(
                zip(partition_df["image_path"], partition_df["bin_annotation"])
            )
        else:
            if dataset_name == "fathomnet-2025":
                partition_df["idx_annotation"] = partition_df[annotation_col].apply(
                    lambda x: np.array(ast.literal_eval(x))
                )
                partition_df["image_path"] = partition_df["path"]
            else:
                partition_df["idx_annotation"] = partition_df[annotation_col].apply(
                    lambda x: np.array(
                        [node_id_dict[component] for component in x.split("; ")]
                    )
                )
            annotations_dict = dict(
                zip(partition_df["image_path"], partition_df["idx_annotation"])
            )

        if dataset_name == "benthicnet":
            img_shard_dir = (
                f"./datasets/{dataset_name}/feature_shards/img_embeddings/"
                f"{emb_type}/{partition}"
            )
        elif dataset_name == "fathomnet-2025":
            fn_domain = "main" if data_domain == "image" else data_domain
            img_shard_dir = (
                f"./datasets/{dataset_name}/feature_shards/img_embeddings/"
                f"{emb_type}/{fn_domain}/{partition}"
            )
        else:
            img_shard_dir = (
                f"./datasets/{dataset_name}/feature_shards/img_embeddings/"
                f"{emb_type}/{partition}"
            )
        if dataset_name == "inat":
            txt_shard_dir = (
                f"./datasets/{dataset_name}/feature_shards/text_embeddings/"
                f"{text_model}/{emb_type}/{partition}"
            )
        else:
            txt_shard_dir = (
                f"./datasets/{dataset_name}/feature_shards/text_embeddings/"
                f"{text_model}/{emb_type}/{data_domain}/{partition}"
            )

        print(f"Image shards: {img_shard_dir}")
        print(f"Text shards:  {txt_shard_dir}")

        print("Building image shard index...")
        img_index = build_shard_index(img_shard_dir)
        print(f"Indexed {len(img_index)} image embeddings "
            f"across {len(set(img_index.values()))} shards.")

        print(
            f"Writing {partition} tars "
            f"({len(annotations_dict)} annotations, "
            f"{samples_per_tar} per tar)..."
        )

        write_webdataset_from_shards(
            output_dir, partition, annotations_dict,
            img_shard_dir, img_index, txt_shard_dir, samples_per_tar
        )
    print("\nAll partitions complete.")


if __name__ == "__main__":
    main()
