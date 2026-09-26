import argparse
import os
import pandas as pd
import torch

from huggingface_hub import login
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
from utils.dataset import ImagesDataset


def gen_args():
    parser = argparse.ArgumentParser(description="Extract image embeddings using DINOv3 ViT")
    parser.add_argument(
        "--login_token", type=str, help="Hugging Face token for authentication"
    )

    parser.add_argument(
        "--dataset_name", type=str, default="bn", help="Dataset name (e.g., 'bn' for BenthicNet)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for processing images"
    )
    parser.add_argument(
        "--shard_size_factor", type=int, default=100,
        help="Number of batches per shard (shard_size = batch_size * shard_size_factor)"
    )
    parser.add_argument(
        "--dataset_domain", type=str, default="main",
        help="Dataset domain (e.g., 'main' for full images, 'roi' for region-of-interest crops)"
    )

    parser.add_argument(
        "--check_images", action="store_true", help="Whether to check if images exist before processing"
    )
    parser.add_argument(
        "--linux", action="store_true", help="Whether to use Linux paths (default is Windows)"
    )
    parser.add_argument(
        "--compute_cluster", action="store_true", help="Whether the script is running on a compute cluster"
    )
    parser.add_argument(
        "--use_full_emb", action="store_true",
        help="Whether to use the full embedding (last_hidden_state) instead of just the CLS token"
    )

    return parser.parse_args()


def collate_fn(batch):
    paths, images = zip(*batch)
    return list(paths), list(images)


def save_shard(shard_data, save_dir, shard_idx):
    torch.save(shard_data, os.path.join(save_dir, f"shard_{shard_idx:06d}.pt"))


def main():
    args = gen_args()

    dataset_name = args.dataset_name
    batch_size = args.batch_size
    shard_size_factor = args.shard_size_factor
    check_images = args.check_images
    linux = args.linux
    compute_cluster = args.compute_cluster
    use_full_emb = args.use_full_emb

    dataset_domain = args.dataset_domain

    fathomnet = "fn" in dataset_name.lower() or "fathomnet" in dataset_name.lower()

    if not compute_cluster:
        login(args.login_token)
        os.environ["HF_HOME"] = "./cache"

    if dataset_name == "bn":
        dataset_path = "./datasets/benthicnet"
    elif dataset_name == "fn":
        dataset_path = "./datasets/fathomnet-2025"
    elif dataset_name == "inat":
        dataset_path = "./datasets/inat"
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    if fathomnet:
        root = ""
        if dataset_domain == "main":
            path = f"{dataset_path}/image_annotations.csv"
        elif dataset_domain == "roi":
            path = f"{dataset_path}/roi_annotations.csv"
        else:
            raise ValueError(f"Unsupported dataset domain for FathomNet: {dataset_domain}")
    elif dataset_name == "inat":
        root = ""
        path = f"{dataset_path}/inat_annotations.csv"
    else:
        if linux or compute_cluster:
            if compute_cluster:
                root = f"{os.environ['SLURM_TMPDIR']}/"
            else:
                root = "./datasets/benthicnet/images/"
        else:
            root = "./datasets/benthicnet/images/"
        path = f"{dataset_path}/benthicnet_labelled.csv"

    if use_full_emb:
        emb_type_dir = "full"
    else:
        emb_type_dir = "cls"

    full_image_df = pd.read_csv(path)
    if fathomnet or dataset_name == "inat":
        full_image_df["image_path"] = full_image_df["path"]
    else:
        full_image_df["image_path"] = (
            full_image_df["dataset"].astype(str) + "/"
            + full_image_df["site"].astype(str) + "/"
            + full_image_df["image"].astype(str) + ".jpg"
        )

    partition_map = (
        full_image_df[["image_path", "partition"]]
        .drop_duplicates(subset="image_path", keep="first")
        .set_index("image_path")["partition"]
        .to_dict()
    )

    partitions = sorted(set(partition_map.values()))
    print(f"Found partitions: {partitions}")

    unique_image_paths = full_image_df["image_path"].unique()
    unique_image_paths_full = [os.path.join(root, p) for p in unique_image_paths]

    if check_images:
        for i, image_path in enumerate(unique_image_paths_full):
            print(f"Checking image {i+1}/{len(unique_image_paths_full)}: {image_path}" + " " * 50, end="\r")
            if Path(image_path).is_file():
                pass

    dataset = ImagesDataset(unique_image_paths_full)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if compute_cluster:
        model_path = "./models/DinoV3-ViTL16-pretrain-lvd1689m/snapshots/tensors"
        processor = AutoImageProcessor.from_pretrained(
            model_path,
            local_files_only=True
        )
        model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True
        ).to(device)
    else:
        processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov3-vitl16-pretrain-lvd1689m"
        )

        model = AutoModel.from_pretrained(
            "facebook/dinov3-vitl16-pretrain-lvd1689m"
        ).to(device)

    model.eval()

    base_save_dir = f"{dataset_path}/feature_shards/img_embeddings/{emb_type_dir}"
    if fathomnet:
        base_save_dir = os.path.join(base_save_dir, dataset_domain)
    shard_size = batch_size * shard_size_factor

    partition_state = {}
    for part in partitions:
        part_dir = os.path.join(base_save_dir, part)
        os.makedirs(part_dir, exist_ok=True)
        partition_state[part] = {
            "save_dir": part_dir,
            "current_shard": {},
            "shard_idx": 0,
            "count": 0,
        }

    for (paths, images) in tqdm(dataloader):

        inputs = processor(
            images=images,
            return_tensors="pt"
        ).to(device)

        outputs = model(**inputs)
        features = outputs.last_hidden_state.detach().cpu()
        if not use_full_emb:
            features = features[:, 0, :].clone()

        for full_path, feat in zip(paths, features):
            if fathomnet or dataset_name == "inat":
                rel_path = full_path
            else:
                rel_path = "/".join(full_path.split("/")[-3:])

            partition = partition_map.get(rel_path)
            if partition is None:
                print(f"Warning: no partition found for {rel_path}, skipping.")
                continue

            state = partition_state[partition]
            state["current_shard"][rel_path] = feat
            state["count"] += 1

            if state["count"] % shard_size == 0:
                save_shard(state["current_shard"], state["save_dir"], state["shard_idx"])
                state["current_shard"] = {}
                state["shard_idx"] += 1

    for part, state in partition_state.items():
        if state["current_shard"]:
            save_shard(state["current_shard"], state["save_dir"], state["shard_idx"])

    for part, state in partition_state.items():
        total = state["count"]
        n_shards = state["shard_idx"] + (1 if state["count"] % shard_size != 0 else 0)
        print(f"  {part}: {total} embeddings across {n_shards} shards")

    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
