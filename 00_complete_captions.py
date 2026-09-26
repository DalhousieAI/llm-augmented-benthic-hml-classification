import argparse
import json
import os
import signal
import time
import pandas as pd
import torch

from huggingface_hub import login
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from utils.utils import (
    create_qwen_dataloader, run_inference, set_seed, generate_inat_message,
)
from utils.caption_merge import (
    load_caption_vars, build_label_pattern, find_missing_vars, merge_caption,
)


STOP_REQUESTED = {"flag": False}


def _request_stop(signum, frame):
    print(f"[signal] Received signal {signum}, requesting graceful stop...",
          flush=True)
    STOP_REQUESTED["flag"] = True


def subset_attrs_for(missing_props, full_attrs):
    missing_set = set(missing_props)
    out = {
        cat: [p for p in props if p in missing_set]
        for cat, props in full_attrs.items()
    }
    return {cat: props for cat, props in out.items() if props}


def gen_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--login_token", type=str, default="no_login_token")
    p.add_argument("--linux", action="store_true")
    p.add_argument("--compute_cluster", action="store_true")
    p.add_argument("--parameter_size", type=str,
                   choices=["8B", "32B"], default="8B")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_rounds", type=int, default=3,
                   help="Max retry passes over still-incomplete samples.")
    p.add_argument("--max_runtime_hours", type=float, default=6.75 * 24)
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--input_csv", type=str, required=True,
                   help="CSV from the captioning run, with image_path,captions.")
    p.add_argument("--output_csv", type=str, default=None,
                   help="Defaults to overwriting --input_csv.")
    p.add_argument("--task_type", type=str, default="inat_cap")
    return p.parse_args()


def main():
    script_start = time.time()
    args = gen_args()
    set_seed(args.seed)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _request_stop)

    if args.login_token != "no_login_token":
        login(args.login_token)

    output_csv = args.output_csv or args.input_csv
    output_path = os.path.dirname(output_csv) + "/"
    os.makedirs(output_path, exist_ok=True)

    vars_list = load_caption_vars()
    pattern = build_label_pattern(vars_list)
    lower_to_var = {v.lower(): v for v in vars_list}

    with open(
        "./datasets/inat/extracted_vars/inat_selected_keywords.json", "r"
    ) as f:
        full_attrs = json.load(f)

    df = pd.read_csv(args.input_csv)
    print(f"[load] {len(df)} captioned rows from {args.input_csv}")

    captions_map = dict(zip(df["image_path"], df["captions"]))

    if args.linux or args.compute_cluster:
        if args.compute_cluster:
            model_path = (f"./models/Qwen3-VL-{args.parameter_size}-Instruct/"
                          f"snapshots/tensors/")
        else:
            model_path = f"Qwen/Qwen3-VL-{args.parameter_size}-Instruct"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto", local_files_only=True,
        )
    else:
        model_path = f"Qwen/Qwen3-VL-{args.parameter_size}-Instruct"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype="auto", device_map="auto",
        )

    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=args.compute_cluster
    )
    processor.tokenizer.padding_side = "left"
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    safety_margin_sec = 15 * 60
    deadline = (script_start + args.max_runtime_hours * 3600
                - safety_margin_sec)

    for round_idx in range(args.max_rounds):
        if STOP_REQUESTED["flag"]:
            print("[complete] Stop requested; halting before round.")
            break

        missing = {
            path: find_missing_vars(cap, vars_list, pattern, lower_to_var)
            for path, cap in captions_map.items()
        }
        missing = {p: m for p, m in missing.items() if m}

        if not missing:
            print(f"[complete] All captions complete after round {round_idx}.")
            break

        total_slots = sum(len(m) for m in missing.values())
        print(f"[complete] Round {round_idx + 1}/{args.max_rounds}: "
              f"{len(missing)} images, {total_slots} missing variable-slots.")

        image_paths = list(missing.keys())
        messages = [
            generate_inat_message(
                p, args.task_type,
                attributes={"_": missing[p]},
            )
            for p in image_paths
        ]
        dataloader = create_qwen_dataloader(
            image_paths, messages, processor, model,
            batch_size=args.batch_size,
        )

        round_ckpt = (f"{output_path}{args.task_type}_retry_round"
                      f"{round_idx + 1}_intermediate.csv")

        retry_df = run_inference(
            dataloader, model, processor,
            task_type=args.task_type,
            output_path=output_path,
            deadline=deadline,
            stop_flag=STOP_REQUESTED,
            checkpoint_every=args.checkpoint_every,
            checkpoint_csv=round_ckpt,
            append_mode=False,
        )

        for _, r in retry_df.iterrows():
            p = r["image_path"]
            captions_map[p] = merge_caption(
                captions_map.get(p, ""), r["captions"],
                vars_list, pattern, lower_to_var,
            )

        df["captions"] = df["image_path"].map(captions_map)
        df.to_csv(output_csv, index=False)
        print(f"[complete] Round {round_idx + 1} merged and saved "
              f"-> {output_csv}")

    df["captions"] = df["image_path"].map(captions_map)
    df.to_csv(output_csv, index=False)

    still_missing = sum(
        1 for c in captions_map.values()
        if find_missing_vars(c, vars_list, pattern, lower_to_var)
    )
    elapsed_h = (time.time() - script_start) / 3600
    print(f"[complete] Done in {elapsed_h:.2f} h. "
          f"{still_missing} images still incomplete "
          f"(stop_requested={STOP_REQUESTED['flag']}).")
    print(f"[complete] Saved to {output_csv}")


if __name__ == "__main__":
    main()
