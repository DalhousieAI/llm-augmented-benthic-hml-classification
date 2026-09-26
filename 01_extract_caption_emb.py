import argparse
import os
import pandas as pd
import torch

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from tqdm import tqdm
from utils.dataset import CaptionsDataset


def gen_args():
    parser = argparse.ArgumentParser(description="Extract text embeddings using Qwen3-Embedding")

    parser.add_argument(
        "--login_token", type=str, help="Hugging Face token for authentication"
    )
    parser.add_argument(
        "--dataset_domain", type=str, help="Dataset domain (e.g., 'bio' for biota)"
    )

    parser.add_argument(
        "--model_id", type=str, default="Qwen/Qwen3-Embedding-8B",
        help="Model ID for Hugging Face (default: 'Qwen/Qwen3-Embedding-8B')"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="bn", help="Dataset name (e.g., 'bn' for BenthicNet)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for processing images"
    )
    parser.add_argument(
        "--shard_size_factor", type=int, default=100,
        help="Number of batches per shard (shard_size = batch_size * shard_size_factor)"
    )
    parser.add_argument(
        "--mrl_emb_dim", type=int, default=1024, help="Dimensionality of the MRL embeddings (default: 1024)"
    )

    parser.add_argument(
        "--normalize", action="store_true", help="Whether to normalize the embeddings (default is no normalization)"
    )
    parser.add_argument(
        "--compute_cluster", action="store_true", help="Whether the script is running on a compute cluster"
    )
    parser.add_argument(
        "--use_full_emb", action="store_true",
        help="Whether to use the full embedding (last_hidden_state) instead of last-token pooling"
    )

    return parser.parse_args()


def collate_fn(batch):
    paths, captions = zip(*batch)
    return list(paths), list(captions)


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]


def save_shard(shard_data, save_dir, shard_idx):
    torch.save(shard_data, os.path.join(save_dir, f"shard_{shard_idx:06d}.pt"))


def main():
    args = gen_args()

    model_id = args.model_id
    dataset_name = args.dataset_name
    dataset_domain = args.dataset_domain
    batch_size = args.batch_size
    shard_size_factor = args.shard_size_factor
    normalize = args.normalize
    compute_cluster = args.compute_cluster
    use_full_emb = args.use_full_emb
    mrl_emb_dim = args.mrl_emb_dim

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
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    if use_full_emb:
        emb_type_dir = "full"
    else:
        emb_type_dir = "ltp"

    if dataset_name == "bn":
        captions_path = f"{dataset_path}/outputs/{dataset_name}_{dataset_domain}_cap_captions.csv"
    elif dataset_name == "fn":
        if dataset_domain == "image":
            fn_domain = "main"
        else:
            fn_domain = dataset_domain
        captions_path = f"{dataset_path}/outputs/{dataset_name}_cap_{fn_domain}_captions.csv"
    elif dataset_name == "inat":
        captions_path = f"{dataset_path}/outputs/{dataset_name}_captions.csv"
    captions_df = pd.read_csv(captions_path)

    data_domain_full = "substrates" if dataset_domain == "sub" \
        else "biota" if dataset_domain == "bio" else dataset_domain
    if dataset_name == "bn":
        annotations_path = f"{dataset_path}/benthicnet_{data_domain_full}.csv"
    elif dataset_name == "fn":
        annotations_path = f"{dataset_path}/{dataset_domain}_annotations.csv"
    elif dataset_name == "inat":
        annotations_path = f"{dataset_path}/{dataset_name}_annotations.csv"

    annotations_df = pd.read_csv(annotations_path)

    if dataset_name == "fn" or dataset_name == "inat":
        annotations_df["image_path"] = annotations_df["path"]

    partition_map = (
        annotations_df[["image_path", "partition"]]
        .drop_duplicates(subset="image_path", keep="first")
        .set_index("image_path")["partition"]
        .to_dict()
    )

    partitions = sorted(set(partition_map.values()))
    print(f"Found partitions: {partitions}")

    dataset = CaptionsDataset(captions_df)

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
        model_name = model_id.split("/")[-1]
        model_path = f"./models/{model_name}/snapshots/tensors/"
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", local_files_only=True)
        model = AutoModel.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
            local_files_only=True).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        model = AutoModel.from_pretrained(
            model_id, attn_implementation="flash_attention_2", dtype=torch.bfloat16
        ).to(device)
    model.eval()

    model_name = model_id.split("/")[-1]
    if dataset_name == "inat":
        base_save_dir = f"{dataset_path}/feature_shards/text_embeddings/{model_name}/{emb_type_dir}/"
    else:
        base_save_dir = f"{dataset_path}/feature_shards/text_embeddings/{model_name}/{emb_type_dir}/{dataset_domain}"

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

    skipped = 0

    with torch.no_grad():
        for (paths, captions) in tqdm(dataloader):

            inputs = tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            if use_full_emb:
                features = outputs.last_hidden_state
                features = features[:, :, :mrl_emb_dim]
            else:
                features = last_token_pool(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )
                features = features[:, :mrl_emb_dim]

            if normalize:
                features = torch.nn.functional.normalize(features, p=2, dim=-1)

            features = features.cpu()

            for path, feat in zip(paths, features):
                partition = partition_map.get(path)
                if partition is None:
                    skipped += 1
                    continue

                state = partition_state[partition]
                state["current_shard"][path] = feat
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

    if skipped > 0:
        print(f"  Skipped {skipped} samples (no partition mapping found)")

    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
