import argparse
import gc
import glob
import hashlib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import psutil
import torch

from sklearn.metrics import average_precision_score
from tqdm import tqdm
from utils.architecture import (
    ConstrainedFFNNModel,
    GatedFusionWithConstraint,
    CrossAttentionFusionWithConstraint,
    SelfAttentionPoolingWithConstraint,
    AdaptiveFusionWithConstraint,
    load_branch,
)
from utils.dataset import InMemoryDataset, make_dataset
from utils.loss import HMLLoss
from utils.utils import (
    build_fathomnet_hierarchy,
    node_f1, set_seed, build_save_id,
    find_optimal_thresholds,
    corrupt_embedding,
)

class ToMultiHot:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, y):
        y = torch.as_tensor(y, dtype=torch.long)
        vec = torch.zeros(self.num_classes, dtype=torch.float32)
        vec.index_fill_(0, y, 1.0)
        return vec

def find_branch_ckpt(results_root, domain_name, modality, emb_type, seed,
                     val_ratio, random_test_set, test_ratio,
                     invert_modes=False, use_adaptive_fusion=False):
    sentinel = 7777777
    ref_id = build_save_id(
        domain_name=domain_name, modality=modality, emb_type=emb_type,
        seed=seed, num_epochs=sentinel, val_ratio=val_ratio,
        random_test_set=random_test_set, test_ratio=test_ratio,
        invert_modes=invert_modes, adaptive_fusion=use_adaptive_fusion,
        freeze_pooling=False, reinit_all=False, plus_epochs=0,
    )
    pattern = ref_id.replace(str(sentinel), "*")
    matches = glob.glob(os.path.join(results_root, pattern, "checkpoint.pt"))

    if len(matches) == 0:
        for vr in range(0, 100):
            vr_str = f"_v-{vr:02d}"
            if vr_str in pattern:
                fallback_pattern = pattern.replace(vr_str, "_v-*")
                matches = glob.glob(os.path.join(results_root, fallback_pattern, "checkpoint.pt"))
                if matches:
                    print(f"  [find_branch_ckpt] No exact val_ratio match; "
                          f"falling back to: {fallback_pattern}")
                break

    if len(matches) == 0:
        return None
    if len(matches) > 1:
        matches.sort(key=os.path.getmtime, reverse=True)
        print(f"  [find_branch_ckpt] Multiple matches found, using most recent: {matches[0]}")
    return matches[0]

def is_test_sample(key, test_ratio=0.2):
    test_ratio = max(0.0, min(1.0, test_ratio))
    h = int(hashlib.md5(("test_split:" + key).encode("utf-8")).hexdigest(), 16)
    return (h % 1000) < int(test_ratio * 1000)

def is_val_sample(key, val_ratio=0.1):
    val_ratio = max(0.0, min(1.0, val_ratio))
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    return (h % 1000) < int(val_ratio * 1000)

def pad_sequences(tensors):
    lengths = [t.shape[0] for t in tensors]
    max_len = max(lengths)
    dim = tensors[0].shape[-1]
    batch_size = len(tensors)

    padded = torch.zeros(batch_size, max_len, dim)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, (t, length) in enumerate(zip(tensors, lengths)):
        padded[i, :length] = t
        mask[i, :length] = True

    return padded, mask

def pad_collate(batch):
    keys, img_embs, txt_embs, labels = zip(*batch)

    labels = torch.stack(labels)
    keys = list(keys)

    if img_embs[0].dim() == 1:
        img_embs = torch.stack(img_embs)
        txt_embs = torch.stack(txt_embs)
        return keys, img_embs, txt_embs, labels, None, None
    else:
        img_embs, img_mask = pad_sequences(img_embs)
        txt_embs, txt_mask = pad_sequences(txt_embs)
        return keys, img_embs, txt_embs, labels, img_mask, txt_mask

def pad_collate_single(batch):
    keys, embs, labels = zip(*batch)
    labels = torch.stack(labels)
    keys = list(keys)
    if embs[0].dim() == 1:
        embs = torch.stack(embs)
        return keys, embs, labels, None
    else:
        embs, mask = pad_sequences(embs)
        return keys, embs, labels, mask

def graph_plots(train_metrics, val_metrics, metric_name, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics, label=f"Train {metric_name}")
    plt.plot(val_metrics, label=f"Val {metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def gen_args():
    parser = argparse.ArgumentParser(
        description="Train a constrained FFNN with HML loss on FathomNet features."
    )
    parser.add_argument(
        "--feature_set",
        type=str,
        default="fn_cls_sub",
        help="Feature set to use (e.g. fn_cls_sub, fn_full_bio, etc.)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="txt",
        help="Modality to use: 'img', 'txt', or 'full'"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="gelu",
        help="Activation function to use in the model (e.g. relu, gelu, etc.)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.7,
        help="Dropout rate to use in the model"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of layers in the model"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of epochs to train for"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.0,
        help="Proportion of training samples to use for validation"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Number of attention heads for cross-attention fusion and self-attention pooling"
    )
    parser.add_argument(
        "--pool_method",
        type=str,
        default="attn",
        choices=["attn", "mean", "cls"],
        help="Pooling method for cross-attention fusion (attn, mean, or cls)"
    )
    parser.add_argument(
        "--init_bias",
        type=float,
        default=2.0,
        help="Initial gate bias for gated fusion (higher = more weight on image)"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.2,
        help="Proportion of samples for test set when --random_test_set is enabled"
    )
    parser.add_argument(
        "--plus_epochs",
        type=int,
        default=0,
        help="Additional training epochs beyond --num_epochs. Checkpoint lookups "
            "still use --num_epochs, so branch/source checkpoints are unaffected."
    )
    parser.add_argument(
        "--image_corruption", type=float, default=0.0,
        help="Fraction of Gaussian noise to mix into image embeddings "
            "(0 = clean, 1 = pure noise). Used for ablation studies."
    )
    parser.add_argument(
        "--text_corruption", type=float, default=0.0,
        help="Fraction of Gaussian noise to mix into text embeddings "
            "(0 = clean, 1 = pure noise). Used for ablation studies."
    )

    parser.add_argument(
        "--compute_cluster", action="store_true", help="Whether the script is running on a compute cluster"
    )
    parser.add_argument(
        "--disable_pbars", action="store_true", help="Disable progress bars"
    )
    parser.add_argument(
        "--check_mem", action="store_true", help="Whether to print memory usage during training"
    )
    parser.add_argument(
        "--invert_modes", action="store_true", help="Whether to invert the modalities (default is image queries text)"
    )
    parser.add_argument(
        "--adaptive_fusion", action="store_true",
        help="Whether to use adaptive fusion with a router (only applicable for full sequential embeddings)"
    )
    parser.add_argument(
        "--random_test_set", action="store_true",
        help="Mix all tar shards and create a random train/test split"
    )
    parser.add_argument(
        "--reinit_all", action="store_true",
        help="Reinitialize all branch pooling, expert heads, and gate from scratch "
            "(train everything end-to-end; only for --adaptive_fusion)"
    )
    parser.add_argument(
        "--freeze_pooling", action="store_true",
        help="Load pre-trained pooling weights, freeze them, and retrain only the classifier head"
    )
    args = parser.parse_args()
    return args

def main():
    proc = psutil.Process(os.getpid())

    args = gen_args()
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_set = args.feature_set
    seed = args.seed
    modality = args.modality
    compute_cluster = args.compute_cluster
    disable_pbars = args.disable_pbars
    check_mem = args.check_mem

    random_test_set = args.random_test_set
    test_ratio = args.test_ratio

    parts = feature_set.split("_")
    if len(parts) != 3 or parts[0] != "fn":
        raise ValueError(
            f"Invalid feature_set '{feature_set}'. "
            f"Expected format 'fn_<emb_type>_<domain>' (e.g. fn_cls_sub, fn_full_bio)."
        )
    _, emb_type, domain = parts
    dataset_name = "fathomnet-2025"
    domain_name = "image" if domain == "main" else domain

    invert_modes = args.invert_modes
    use_adaptive_fusion = args.adaptive_fusion

    if invert_modes and modality != "full":
        raise ValueError("invert_modes can only be used when modality is 'full'")
    if use_adaptive_fusion and invert_modes:
        raise ValueError("--adaptive_fusion and --invert_modes cannot be used together")

    tar_dir = f"./datasets/{dataset_name}/feature_tars/{emb_type}/{domain}"
    if compute_cluster:
        tar_dir = os.path.join(os.environ.get("SLURM_TMPDIR"), domain)
    train_tar_paths = sorted(glob.glob(os.path.join(tar_dir, "train_*.tar")))
    test_tar_paths = sorted(glob.glob(os.path.join(tar_dir, "test_*.tar")))

    if not train_tar_paths:
        raise FileNotFoundError(f"No training tar shards found in {tar_dir}/train_*.tar")
    if not test_tar_paths:
        raise FileNotFoundError(f"No test tar shards found in {tar_dir}/test_*.tar")

    if random_test_set:
        all_tar_paths = sorted(set(train_tar_paths + test_tar_paths))
        print(f"--random_test_set enabled: mixing all {len(all_tar_paths)} shard(s) "
            f"with test_ratio={test_ratio:.2f}")

    print(f"Found {len(train_tar_paths)} training shard(s) and {len(test_tar_paths)} test shard(s)")

    if emb_type == "full" and modality == "full":
        if use_adaptive_fusion:
            print(f"Using AdaptiveFusion with router for full sequential embeddings")
        elif invert_modes:
            print(f"Inverting modalities: text queries image in cross-attention fusion")
        else:
            print(f"Using CrossAttentionFusion with pool_method={args.pool_method}")
    elif emb_type == "full" and modality in ("img", "txt"):
        print(f"Using SelfAttentionPooling for single modality '{modality}' with sequential embeddings")
    elif emb_type == "cls" and modality == "full":
        print(f"Using GatedFusion with init_bias={args.init_bias}")
    elif emb_type == "cls" and modality in ("img", "txt"):
        print(f"Using ConstrainedFFNN for single modality '{modality}' with CLS embeddings")
    else:
        raise ValueError(f"Unknown emb_type/modality combination: {emb_type}/{modality}")

    if use_adaptive_fusion:
        if emb_type != "full":
            raise ValueError("Adaptive fusion requires emb_type='full'")

    hyperparams = {
        "activation": args.activation,
        "dropout": args.dropout,
        "num_layers": args.num_layers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
    }

    emb_dim = 1024
    batch_size = args.batch_size
    val_ratio = args.val_ratio

    save_identification = build_save_id(
        domain_name=domain_name, modality=modality, emb_type=emb_type, seed=seed,
        num_epochs=hyperparams["num_epochs"],
        val_ratio=val_ratio, random_test_set=random_test_set,
        test_ratio=test_ratio, invert_modes=invert_modes,
        adaptive_fusion=use_adaptive_fusion,
        freeze_pooling=args.freeze_pooling, reinit_all=args.reinit_all,
        plus_epochs=args.plus_epochs, image_corruption=args.image_corruption,
        text_corruption=args.text_corruption,
    )

    results_root = f"./datasets/{dataset_name}/results"
    save_dir = os.path.join(results_root, save_identification)
    plots_dir = os.path.join(save_dir, "plots")

    checkpoint_path = os.path.join(save_dir, "checkpoint.pt")
    if os.path.isfile(checkpoint_path):
        print(f"Checkpoint already exists at {checkpoint_path}. "
              f"Skipping training - trial already complete.")
        return

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    set_seed(seed)

    hierarchy_df = pd.read_csv(f"./datasets/{dataset_name}/full_annotations.csv")

    index_to_taxon = json.load(open(f"./datasets/{dataset_name}/train/index_to_taxon.json", "r"))
    taxon_to_index = json.load(open(f"./datasets/{dataset_name}/train/taxon_to_index.json", "r"))

    hierarchy_info = build_fathomnet_hierarchy(
        hierarchy_df, idx_to_label=index_to_taxon, label_to_idx=taxon_to_index
        )
    descendant_matrix = hierarchy_info["descendant_matrix"]
    num_classes = descendant_matrix.shape[0]
    label_transform = ToMultiHot(num_classes)

    R = torch.from_numpy(descendant_matrix).float()
    R = R.to(device)

    collate_fn = pad_collate if modality == "full" else pad_collate_single

    if random_test_set:
        src_train_paths = all_tar_paths
        src_test_paths = all_tar_paths

        train_filter_mem = lambda x: (
            not is_test_sample(x[0], test_ratio=test_ratio)
            and not is_val_sample(x[0], val_ratio=val_ratio)
        )
        val_filter_mem = lambda x: (
            not is_test_sample(x[0], test_ratio=test_ratio)
            and is_val_sample(x[0], val_ratio=val_ratio)
        )
        test_filter_mem = lambda x: is_test_sample(x[0], test_ratio=test_ratio)

        train_filter_key = lambda k: (
            not is_test_sample(k, test_ratio=test_ratio)
            and not is_val_sample(k, val_ratio=val_ratio)
        )
        val_filter_key = lambda k: (
            not is_test_sample(k, test_ratio=test_ratio)
            and is_val_sample(k, val_ratio=val_ratio)
        )
        test_filter_key = lambda k: is_test_sample(k, test_ratio=test_ratio)
    else:
        src_train_paths = train_tar_paths
        src_test_paths = test_tar_paths

        train_filter_mem = lambda x: not is_val_sample(x[0], val_ratio=val_ratio)
        val_filter_mem = lambda x: is_val_sample(x[0], val_ratio=val_ratio)
        test_filter_mem = None

        train_filter_key = lambda k: not is_val_sample(k, val_ratio=val_ratio)
        val_filter_key = lambda k: is_val_sample(k, val_ratio=val_ratio)
        test_filter_key = None

    if compute_cluster:
        train_dataset = InMemoryDataset(
            src_train_paths, label_transform, modality=modality,
            filter_fn=train_filter_mem,
            disable_pbars=disable_pbars,
            check_mem=check_mem,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            collate_fn=collate_fn,
        )

        if val_ratio > 0.0:
            val_dataset = InMemoryDataset(
                src_train_paths, label_transform, modality=modality,
                filter_fn=val_filter_mem,
                disable_pbars=disable_pbars,
                check_mem=check_mem,
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
            )

        test_dataset = InMemoryDataset(
            src_test_paths, label_transform, modality=modality,
            filter_fn=test_filter_mem,
            disable_pbars=disable_pbars,
            check_mem=check_mem,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
    else:
        train_dataset = make_dataset(
            src_train_paths,
            label_transform,
            modality=modality,
            seed=seed,
            key_filter_fn=train_filter_key,
            shuffle=True
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0,
            drop_last=True,
            collate_fn=collate_fn,
        )

        if val_ratio > 0.0:
            val_dataset = make_dataset(
                src_train_paths,
                label_transform,
                modality=modality,
                key_filter_fn=val_filter_key,
                shuffle=False,
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=0,
                collate_fn=collate_fn,
            )

        test_kwargs = dict(
            modality=modality,
            shuffle=False
        )
        if test_filter_key is not None:
            test_kwargs["key_filter_fn"] = test_filter_key

        test_dataset = make_dataset(
            src_test_paths,
            label_transform,
            **test_kwargs,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=collate_fn,
            shuffle=False,
        )

    if emb_type == "full":
        if modality == "full":
            if use_adaptive_fusion:
                branch_kwargs = dict(
                    domain_name=domain_name, emb_type=emb_type, seed=seed,
                    val_ratio=val_ratio, random_test_set=random_test_set,
                    test_ratio=test_ratio,
                )

                img_ckpt = find_branch_ckpt(results_root, modality="img", **branch_kwargs)
                txt_ckpt = find_branch_ckpt(results_root, modality="txt", **branch_kwargs)
                imgxtxt_ckpt = find_branch_ckpt(results_root, modality="full", **branch_kwargs)

                for name, path in [("img", img_ckpt), ("txt", txt_ckpt), ("imgxtxt", imgxtxt_ckpt)]:
                    if path is None:
                        raise FileNotFoundError(
                            f"Pre-trained {name} branch checkpoint not found for seed={seed}. "
                            f"Train it first before running adaptive fusion."
                        )

                print(f"Loading img branch from:     {img_ckpt}")
                print(f"Loading txt branch from:     {txt_ckpt}")
                print(f"Loading imgxtxt branch from: {imgxtxt_ckpt}")

                img_branch, img_hp = load_branch(img_ckpt, "unimodal", emb_dim, num_classes, R, device)
                txt_branch, txt_hp = load_branch(txt_ckpt, "unimodal", emb_dim, num_classes, R, device)
                imgxtxt_branch, imgxtxt_hp = load_branch(imgxtxt_ckpt, "cross", emb_dim, num_classes, R, device)

                model = AdaptiveFusionWithConstraint(
                    num_classes=num_classes,
                    feature_dim=emb_dim,
                    img_branch=img_branch,
                    txt_branch=txt_branch,
                    imgxtxt_branch=imgxtxt_branch,
                    R=R,
                    branch_hyperparams=[img_hp, txt_hp, imgxtxt_hp],
                    dropout=hyperparams["dropout"],
                )
            else:
                model = CrossAttentionFusionWithConstraint(
                    input_dim=emb_dim,
                    hidden_dim=emb_dim,
                    output_dim=num_classes,
                    hyperparams=hyperparams,
                    R=R,
                    num_heads=args.num_heads,
                    pool_method=args.pool_method,
                )
        else:
            model = SelfAttentionPoolingWithConstraint(
                input_dim=emb_dim,
                hidden_dim=emb_dim,
                output_dim=num_classes,
                hyperparams=hyperparams,
                R=R,
                num_heads=args.num_heads,
            )
    else:
        if modality == "full":
            model = GatedFusionWithConstraint(
                input_dim=emb_dim,
                hidden_dim=emb_dim,
                output_dim=num_classes,
                hyperparams=hyperparams,
                R=R,
                init_bias=args.init_bias,
            )
        else:
            model = ConstrainedFFNNModel(
                input_dim=emb_dim,
                hidden_dim=emb_dim,
                output_dim=num_classes,
                hyperparams=hyperparams,
                R=R,
            )

    model = model.to(device)

    assert not args.reinit_all or use_adaptive_fusion, \
        "Reinitialization can only be used with --adaptive_fusion"

    if args.freeze_pooling and args.reinit_all:
        raise ValueError("--freeze_pooling and --reinit_all cannot be used together")

    if args.freeze_pooling and emb_type != "full":
        raise ValueError("--freeze_pooling requires emb_type='full' (sequential embeddings)")

    if use_adaptive_fusion and args.reinit_all:
        model.reinit_all()
        print("All branches, expert heads, and gate reinitialized from scratch.")

    if args.freeze_pooling:
        if use_adaptive_fusion:
            model.reinit_expert_heads()
            print("Adaptive fusion: branch pooling frozen. Expert heads reinitialized from scratch.")
        else:
            source_ckpt_path = find_branch_ckpt(
                results_root, domain_name=domain_name, modality=modality,
                emb_type=emb_type, seed=seed, val_ratio=val_ratio,
                random_test_set=random_test_set, test_ratio=test_ratio,
                invert_modes=invert_modes, use_adaptive_fusion=False,
            )
            if source_ckpt_path is None:
                raise FileNotFoundError(
                    f"Pre-trained checkpoint not found for seed={seed}, modality='{modality}'. "
                    f"Train the model without --freeze_pooling first."
                )
            print(f"Loading pre-trained weights from: {source_ckpt_path}")
            ckpt = torch.load(source_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.freeze_pooling()
            model.reinit_classifier()
            print("Pooling frozen. Classifier reinitialized.")

    if use_adaptive_fusion:
        trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = torch.optim.Adam(
            trainable,
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"],
        )
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"],
        )

    criterion = HMLLoss(R=R)

    loss_trajectory = []
    aps = []
    bin_aps = []
    f1s = []
    precisions = []
    recalls = []

    if val_ratio > 0.0:
        val_loss_trajectory = []
        val_aps = []
        val_bin_aps = []
        val_f1s = []
        val_precisions = []
        val_recalls = []

    training_epochs = args.plus_epochs or hyperparams["num_epochs"]

    epoch_bar = tqdm(
        range(training_epochs),
        desc="Epochs",
        position=0,
        disable=disable_pbars
        )
    for epoch in epoch_bar:
        model.train()

        epoch_prediction_probs = []
        epoch_labels = []
        epoch_loss = 0.0

        if not compute_cluster:
            train_dataset.set_epoch(epoch)

        batch_bar = tqdm(
            enumerate(train_dataloader),
            desc=f"  Train Ep {epoch}",
            position=1,
            leave=False,
            disable=disable_pbars
        )
        for i, batch in batch_bar:
            if check_mem and i % 10 == 0:
                torch.cuda.empty_cache()
                print(f"[batch {i}] RSS={proc.memory_info().rss/1024**3:.1f}GB "
                    f"GPU={torch.cuda.memory_allocated()/1024**3:.1f}GB", flush=True)
            if modality == "full":
                _, img_embeddings, txt_embeddings, labels, img_mask, txt_mask = batch
                img_embeddings = img_embeddings.to(device)
                txt_embeddings = txt_embeddings.to(device)
                img_mask = img_mask.to(device) if img_mask is not None else None
                txt_mask = txt_mask.to(device) if txt_mask is not None else None
                img_embeddings = corrupt_embedding(img_embeddings, args.image_corruption, img_mask)
                txt_embeddings = corrupt_embedding(txt_embeddings, args.text_corruption, txt_mask)
            else:
                _, input_embedding, labels, input_mask = batch
                input_embedding = input_embedding.to(device)
                input_mask = input_mask.to(device) if input_mask is not None else None
                if modality == "img":
                    input_embedding = corrupt_embedding(input_embedding, args.image_corruption, input_mask)
                elif modality == "txt":
                    input_embedding = corrupt_embedding(input_embedding, args.text_corruption, input_mask)

            labels = labels.to(device)

            optimizer.zero_grad()
            if modality == "full":
                if emb_type == "full":
                    if invert_modes:
                        raw_output = model(
                            txt_embeddings, img_embeddings, q_mask=txt_mask, kv_mask=img_mask
                            )
                    else:
                        if use_adaptive_fusion:
                            raw_output = model(
                                img_embeddings, txt_embeddings, img_mask=img_mask, txt_mask=txt_mask
                                )
                        else:
                            raw_output = model(
                                img_embeddings, txt_embeddings, q_mask=img_mask, kv_mask=txt_mask
                                )
                else:
                    raw_output = model(img_embeddings, txt_embeddings)
            else:
                if emb_type == "full":
                    raw_output = model(input_embedding, mask=input_mask)
                else:
                    raw_output = model(input_embedding)

            loss, prediction_probs = criterion(raw_output, labels)

            loss.backward()
            optimizer.step()

            epoch_prediction_probs.append(prediction_probs.detach().cpu())
            epoch_labels.append(labels.detach().cpu())
            epoch_loss += loss.item()

            batch_bar.set_postfix(loss=epoch_loss / (i + 1))

            if i % 20 == 0:
                gc.collect()

        batch_bar.close()

        epoch_loss /= i + 1
        epoch_labels = torch.cat(epoch_labels)
        epoch_prediction_probs = torch.cat(epoch_prediction_probs)
        epoch_predictions = epoch_prediction_probs > 0.5

        epoch_ap = average_precision_score(
            epoch_labels, epoch_prediction_probs, average="micro"
        )
        epoch_bin_ap = average_precision_score(
            epoch_labels, epoch_predictions, average="micro"
        )

        epoch_f1, epoch_precision, epoch_recall, \
            _, _, _ = node_f1(epoch_labels, epoch_predictions)

        loss_trajectory.append(epoch_loss)
        aps.append(epoch_ap)
        bin_aps.append(epoch_bin_ap)
        f1s.append(epoch_f1)
        precisions.append(epoch_precision)
        recalls.append(epoch_recall)

        if val_ratio > 0.0:
            model.eval()

            val_epoch_prediction_probs = []
            val_epoch_labels = []
            val_epoch_loss = 0.0

            val_batch_bar = tqdm(
                enumerate(val_dataloader),
                desc=f"  Val   Ep {epoch}",
                position=1,
                leave=False,
                disable=disable_pbars
            )
            with torch.no_grad():
                for j, batch in val_batch_bar:
                    if modality == "full":
                        _, img_embeddings, txt_embeddings, labels, img_mask, txt_mask = batch
                        img_embeddings = img_embeddings.to(device)
                        txt_embeddings = txt_embeddings.to(device)
                        img_mask = img_mask.to(device) if img_mask is not None else None
                        txt_mask = txt_mask.to(device) if txt_mask is not None else None
                        img_embeddings = corrupt_embedding(img_embeddings, args.image_corruption, img_mask)
                        txt_embeddings = corrupt_embedding(txt_embeddings, args.text_corruption, txt_mask)
                    else:
                        _, input_embedding, labels, input_mask = batch
                        input_embedding = input_embedding.to(device)
                        input_mask = input_mask.to(device) if input_mask is not None else None
                        if modality == "img":
                            input_embedding = corrupt_embedding(input_embedding, args.image_corruption, input_mask)
                        elif modality == "txt":
                            input_embedding = corrupt_embedding(input_embedding, args.text_corruption, input_mask)

                    labels = labels.to(device)

                    if modality == "full":
                        if emb_type == "full":
                            if invert_modes:
                                raw_output, val_prediction_probs_batch = model(txt_embeddings, img_embeddings, q_mask=txt_mask, kv_mask=img_mask)
                            else:
                                if use_adaptive_fusion:
                                    raw_output, val_prediction_probs_batch = model(
                                        img_embeddings, txt_embeddings, img_mask=img_mask, txt_mask=txt_mask
                                        )
                                else:
                                    raw_output, val_prediction_probs_batch = model(
                                        img_embeddings, txt_embeddings, q_mask=img_mask, kv_mask=txt_mask
                                        )
                        else:
                            raw_output, val_prediction_probs_batch = model(img_embeddings, txt_embeddings)
                    else:
                        if emb_type == "full":
                            raw_output, val_prediction_probs_batch = model(input_embedding, mask=input_mask)
                        else:
                            raw_output, val_prediction_probs_batch = model(input_embedding)

                    loss, _ = criterion(raw_output, labels)

                    val_epoch_prediction_probs.append(val_prediction_probs_batch.detach().cpu())
                    val_epoch_labels.append(labels.detach().cpu())
                    val_epoch_loss += loss.item()

                    val_batch_bar.set_postfix(loss=val_epoch_loss / (j + 1))

            val_batch_bar.close()

            val_epoch_loss /= (j + 1)

            val_epoch_labels = torch.cat(val_epoch_labels)
            val_epoch_prediction_probs = torch.cat(val_epoch_prediction_probs)
            val_epoch_predictions = val_epoch_prediction_probs > 0.5

            val_epoch_ap = average_precision_score(
                val_epoch_labels, val_epoch_prediction_probs, average="micro"
            )
            val_epoch_bin_ap = average_precision_score(
                val_epoch_labels, val_epoch_predictions, average="micro"
            )

            val_epoch_f1, val_epoch_precision, val_epoch_recall, _, _, _ = node_f1(
                val_epoch_labels, val_epoch_predictions
            )

            val_loss_trajectory.append(val_epoch_loss)
            val_aps.append(val_epoch_ap)
            val_bin_aps.append(val_epoch_bin_ap)
            val_f1s.append(val_epoch_f1)
            val_precisions.append(val_epoch_precision)
            val_recalls.append(val_epoch_recall)

        postfix = {"loss": f"{epoch_loss:.4f}", "f1": f"{epoch_f1:.4f}"}
        if val_ratio > 0.0:
            postfix["v_loss"] = f"{val_epoch_loss:.4f}"
            postfix["v_f1"] = f"{val_epoch_f1:.4f}"
        epoch_bar.set_postfix(postfix)

    if val_ratio > 0.0:
        graph_plots(loss_trajectory, val_loss_trajectory, "Loss",
                    os.path.join(plots_dir, "loss.png"))
        graph_plots(aps, val_aps, "AP",
                    os.path.join(plots_dir, "ap.png"))
        graph_plots(bin_aps, val_bin_aps, "Binary AP",
                    os.path.join(plots_dir, "bin_ap.png"))
        graph_plots(f1s, val_f1s, "F1",
                    os.path.join(plots_dir, "f1.png"))
        graph_plots(precisions, val_precisions, "Precision",
                    os.path.join(plots_dir, "precision.png"))
        graph_plots(recalls, val_recalls, "Recall",
                    os.path.join(plots_dir, "recall.png"))

        print(f"Final Macro F1: {f1s[-1]:.4f}/{val_f1s[-1]:.4f}")
        print(f"Final Binary AP: {bin_aps[-1]:.4f}/{val_bin_aps[-1]:.4f}")
        print(f"Final Precision: {precisions[-1]:.4f}/{val_precisions[-1]:.4f}")
        print(f"Final Recall: {recalls[-1]:.4f}/{val_recalls[-1]:.4f}")
        print(f"Final Loss: {loss_trajectory[-1]:.4f}/{val_loss_trajectory[-1]:.4f}")
        print(f"Final AP: {aps[-1]:.4f}/{val_aps[-1]:.4f}")

    print("\nOptimizing per-node thresholds...")
    model.eval()

    if val_ratio > 0.0:
        thresh_probs = val_epoch_prediction_probs.numpy()
        thresh_labels = val_epoch_labels.numpy()
        thresh_source = "validation"
    else:
        thresh_probs_list = []
        thresh_labels_list = []

        if not compute_cluster:
            thresh_dataset = make_dataset(
                src_train_paths, label_transform, modality=modality,
                key_filter_fn=train_filter_key, shuffle=False
            )
            thresh_dataloader = torch.utils.data.DataLoader(
                thresh_dataset, batch_size=batch_size, num_workers=0,
                collate_fn=collate_fn
            )
        else:
            thresh_dataloader = train_dataloader

        thresh_bar = tqdm(
            enumerate(thresh_dataloader),
            desc="  Collecting preds for threshold opt",
            position=0,
            disable=disable_pbars
        )
        with torch.no_grad():
            for i, batch in thresh_bar:
                if modality == "full":
                    _, img_embeddings, txt_embeddings, labels, img_mask, txt_mask = batch
                    img_embeddings = img_embeddings.to(device)
                    txt_embeddings = txt_embeddings.to(device)
                    img_mask = img_mask.to(device) if img_mask is not None else None
                    txt_mask = txt_mask.to(device) if txt_mask is not None else None
                    img_embeddings = corrupt_embedding(img_embeddings, args.image_corruption, img_mask)
                    txt_embeddings = corrupt_embedding(txt_embeddings, args.text_corruption, txt_mask)
                else:
                    _, input_embedding, labels, input_mask = batch
                    input_embedding = input_embedding.to(device)
                    input_mask = input_mask.to(device) if input_mask is not None else None
                    if modality == "img":
                        input_embedding = corrupt_embedding(input_embedding, args.image_corruption, input_mask)
                    elif modality == "txt":
                        input_embedding = corrupt_embedding(input_embedding, args.text_corruption, input_mask)

                labels = labels.to(device)

                if modality == "full":
                    if emb_type == "full":
                        if invert_modes:
                            _, probs_batch = model(
                                txt_embeddings, img_embeddings,
                                q_mask=txt_mask, kv_mask=img_mask
                            )
                        else:
                            if use_adaptive_fusion:
                                _, probs_batch = model(
                                    img_embeddings, txt_embeddings,
                                    img_mask=img_mask, txt_mask=txt_mask
                                )
                            else:
                                _, probs_batch = model(
                                    img_embeddings, txt_embeddings,
                                    q_mask=img_mask, kv_mask=txt_mask
                                )
                    else:
                        _, probs_batch = model(img_embeddings, txt_embeddings)
                else:
                    if emb_type == "full":
                        _, probs_batch = model(input_embedding, mask=input_mask)
                    else:
                        _, probs_batch = model(input_embedding)

                thresh_probs_list.append(probs_batch.cpu())
                thresh_labels_list.append(labels.cpu())

        thresh_bar.close()
        thresh_probs = torch.cat(thresh_probs_list).numpy()
        thresh_labels = torch.cat(thresh_labels_list).numpy()
        thresh_source = "training"

    unconstrained_t, hierarchical_t = find_optimal_thresholds(
        thresh_probs, thresh_labels, descendant_matrix
    )

    node_id_dict = hierarchy_info.get("node_id_dict", {})
    if node_id_dict:
        id_to_name = {v: k for k, v in node_id_dict.items()}
    else:
        id_to_name = {i: str(i) for i in range(num_classes)}

    thresh_path = os.path.join(save_dir, "thresholds.txt")
    with open(thresh_path, "w") as f:
        f.write(f"# Per-node optimal thresholds (optimized on {thresh_source} data)\n")
        f.write(f"# node_index\tnode_name\tunconstrained\thierarchical\n")
        for i in range(num_classes):
            name = id_to_name.get(i, str(i))
            f.write(f"{i}\t{name}\t{unconstrained_t[i]:.4f}\t{hierarchical_t[i]:.4f}\n")

    np.save(os.path.join(save_dir, "thresholds.npy"), hierarchical_t)

    print(f"Thresholds saved to {thresh_path}")
    print(f"  Source: {thresh_source} data ({thresh_probs.shape[0]} samples)")
    print(f"  Range:  [{hierarchical_t.min():.4f}, {hierarchical_t.max():.4f}]")
    print(f"  Mean:   {hierarchical_t.mean():.4f}")
    print(f"  Median: {np.median(hierarchical_t):.4f}")

    model.eval()

    test_sample_ids = []
    test_prediction_probs = []

    test_batch_bar = tqdm(
        enumerate(test_dataloader),
        desc="Test",
        position=0,
        leave=True,
        disable=disable_pbars
    )
    with torch.no_grad():
        for i, batch in test_batch_bar:
            if modality == "full":
                keys, img_embeddings, txt_embeddings, labels, img_mask, txt_mask = batch
                img_embeddings = img_embeddings.to(device)
                txt_embeddings = txt_embeddings.to(device)
                img_mask = img_mask.to(device) if img_mask is not None else None
                txt_mask = txt_mask.to(device) if txt_mask is not None else None
                img_embeddings = corrupt_embedding(img_embeddings, args.image_corruption, img_mask)
                txt_embeddings = corrupt_embedding(txt_embeddings, args.text_corruption, txt_mask)
            else:
                keys, input_embedding, labels, input_mask = batch
                input_embedding = input_embedding.to(device)
                input_mask = input_mask.to(device) if input_mask is not None else None
                if modality == "img":
                    input_embedding = corrupt_embedding(input_embedding, args.image_corruption, input_mask)
                elif modality == "txt":
                    input_embedding = corrupt_embedding(input_embedding, args.text_corruption, input_mask)

            if modality == "full":
                if emb_type == "full":
                    if invert_modes:
                        _, test_prediction_probs_batch = model(txt_embeddings, img_embeddings, q_mask=txt_mask, kv_mask=img_mask)
                    else:
                        if use_adaptive_fusion:
                            _, test_prediction_probs_batch = model(
                                img_embeddings, txt_embeddings, img_mask=img_mask, txt_mask=txt_mask
                                )
                        else:
                            _, test_prediction_probs_batch = model(
                                img_embeddings, txt_embeddings, q_mask=img_mask, kv_mask=txt_mask
                                )
                else:
                    _, test_prediction_probs_batch = model(img_embeddings, txt_embeddings)
            else:
                if emb_type == "full":
                    _, test_prediction_probs_batch = model(input_embedding, mask=input_mask)
                else:
                    _, test_prediction_probs_batch = model(input_embedding)

            test_sample_ids.append(keys)
            test_prediction_probs.append(test_prediction_probs_batch.cpu())

    test_batch_bar.close()

    prediction_df = pd.DataFrame({
        "sample_id": [item for sublist in test_sample_ids for item in sublist],
        "prediction_probs": torch.cat(test_prediction_probs).tolist(),
    })
    prediction_df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

    checkpoint = {
        "epoch": training_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "hyperparams": hyperparams,
        "args": vars(args),
        "train_loss": loss_trajectory[-1],
    }
    if val_ratio > 0.0:
        checkpoint["val_loss"] = val_loss_trajectory[-1]
    torch.save(checkpoint, os.path.join(save_dir, "checkpoint.pt"))

    print(f"Results saved to {save_dir}")

if __name__ == "__main__":
    main()
