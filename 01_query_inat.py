import argparse
import os
import signal
import time
import pandas as pd
import torch

from huggingface_hub import login
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from utils.utils import (
    create_qwen_dataloader, run_inference,
    set_seed, inat_df_to_msgs, build_dataframe_from_directory,
)


STOP_REQUESTED = {"flag": False}


def _request_stop(signum, frame):
    print(f"[signal] Received signal {signum}, requesting graceful stop...",
          flush=True)
    STOP_REQUESTED["flag"] = True


def gen_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--login_token", type=str, default="no_login_token")
    parser.add_argument("--linux", action="store_true")
    parser.add_argument("--compute_cluster", action="store_true")
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--task_type", type=str,
                        choices=["inat", "inat_cap", "inat_cls"],
                        default="inat")
    parser.add_argument("--split", type=str,
                        choices=["train_mini", "val"], default="train_mini")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k_vars", type=int, default=25)
    parser.add_argument("--parameter_size", type=str,
                        choices=["8B", "32B"], default="8B")
    parser.add_argument("--kingdom", type=str, default=None,
                        choices=["Animalia", "Plantae", "Fungi", None])

    parser.add_argument(
        "--max_runtime_hours", type=float, default=6.75 * 24,
        help="Soft wall-time budget in hours. The script will stop "
             "inference and save progress before this is exceeded. "
             "Default leaves ~6h buffer under a 7-day SLURM limit.",
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=50,
        help="Flush partial results to disk every N processed batches.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="If set, skip image paths already present in the output CSV.",
    )
    return parser.parse_args()


def main():
    print("[boot] main() entered", flush=True)

    script_start = time.time()
    args = gen_args()
    set_seed(args.seed)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _request_stop)

    if args.login_token != "no_login_token":
        login(args.login_token)

    linux = args.linux
    computer_cluster = args.compute_cluster
    parameter_size = args.parameter_size
    test_mode = args.test_mode

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" if test_mode else "0"

    task_type = args.task_type
    split = args.split
    batch_size = args.batch_size

    train_df = pd.read_csv(f"./datasets/inat/train_mini.csv")
    try:
        val_df = pd.read_csv(f"./datasets/inat/val.csv")
    except FileNotFoundError:
        print("Validation CSV not found; proceeding with train_mini only.")
        val_df = pd.DataFrame(columns=train_df.columns)

    df = pd.concat([train_df, val_df], ignore_index=True)

    if args.kingdom is not None:
        df = df[df["kingdom"] == args.kingdom].reset_index(drop=True)
        print(f"Filtered to kingdom={args.kingdom}: {len(df)} images.")

    if test_mode:
        n_samples = min(2 * batch_size, len(df))
        df = df.sample(n=n_samples, random_state=args.seed).reset_index(drop=True)

    output_path = "./datasets/inat/outputs/"
    os.makedirs(output_path, exist_ok=True)

    file_suffix = f"_{split}"
    if args.kingdom is not None:
        file_suffix += f"_{args.kingdom.lower()}"

    if "cap" in task_type:
        file_identifier = "captions"
    elif "cls" in task_type:
        file_identifier = "predictions"
    else:
        file_identifier = None

    if file_identifier is not None:
        output_csv_path = (
            f"{output_path}{task_type}{file_suffix}_{file_identifier}.csv"
        )
    else:
        output_csv_path = None

    if args.resume and output_csv_path and os.path.exists(output_csv_path):
        try:
            done_df = pd.read_csv(output_csv_path)
            done_paths = set(done_df["image_path"].tolist()) \
                if "image_path" in done_df.columns else set()
            before = len(df)
            df = df[~df["path"].isin(done_paths)].reset_index(drop=True)
            print(f"[resume] Skipping {before - len(df)} already-processed "
                  f"images; {len(df)} remaining.")
        except Exception as e:
            print(f"[resume] Could not read existing output ({e}); "
                  f"starting fresh.")

    if len(df) == 0:
        print("Nothing left to do.")
        return

    print(f"Split: {split} | Samples: {len(df)} | Task: {task_type}")

    if linux or computer_cluster:
        if computer_cluster:
            model_path = (f"./models/Qwen3-VL-{parameter_size}-Instruct/"
                          f"snapshots/tensors/")
        else:
            model_path = f"Qwen/Qwen3-VL-{parameter_size}-Instruct"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            local_files_only=True,
        )
    else:
        model_path = f"Qwen/Qwen3-VL-{parameter_size}-Instruct"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype="auto", device_map="auto",
        )

    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=computer_cluster
    )
    processor.tokenizer.padding_side = "left"
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    image_paths = df["path"].to_list()
    messages = inat_df_to_msgs(df, task_type=task_type)
    dataloader = create_qwen_dataloader(
        image_paths, messages, processor, model, batch_size=batch_size,
    )

    safety_margin_sec = 15 * 60
    deadline = script_start + args.max_runtime_hours * 3600 - safety_margin_sec
    remaining_h = (deadline - time.time()) / 3600
    print(f"[time] Deadline in {remaining_h:.2f} h "
          f"(max_runtime_hours={args.max_runtime_hours}, "
          f"safety_margin={safety_margin_sec}s)")

    if "cap" in task_type or "cls" in task_type:
        output_df = run_inference(
            dataloader, model, processor,
            task_type=task_type,
            output_path=output_path,
            deadline=deadline,
            stop_flag=STOP_REQUESTED,
            checkpoint_every=args.checkpoint_every,
            checkpoint_csv=output_csv_path,
            append_mode=args.resume and os.path.exists(output_csv_path),
        )
        if args.resume and os.path.exists(output_csv_path):
            try:
                prev = pd.read_csv(output_csv_path)
                output_df = pd.concat([prev, output_df], ignore_index=True)
                output_df = output_df.drop_duplicates(
                    subset=["image_path"], keep="last"
                )
            except Exception as e:
                print(f"[resume] Merge failed ({e}); writing new rows only.")
        output_df.to_csv(output_csv_path, index=False)
        print(f"Saved output to: {output_csv_path}")

    else:
        proposed_variables, image_path_proposed_var_dict = run_inference(
            dataloader, model, processor,
            task_type=task_type,
            output_path=output_path,
            deadline=deadline,
            stop_flag=STOP_REQUESTED,
            checkpoint_every=args.checkpoint_every,
        )
        top_k = args.top_k_vars
        top_k_proposed_vars = dict(
            sorted(proposed_variables.items(),
                   key=lambda item: item[1], reverse=True)[:top_k]
        )
        print(f"Top {top_k} Proposed Variables:", top_k_proposed_vars)

        output_txt_path = f"{output_path}{task_type}{file_suffix}_out.txt"
        with open(output_txt_path, "w") as f:
            f.write("Top Proposed Variables:\n")
            for var, freq in top_k_proposed_vars.items():
                f.write(f"{var}: {freq}\n")

        output_csv_path = (
            f"{output_path}{task_type}{file_suffix}_image_path_proposed_var.csv"
        )
        image_path_var_df = pd.DataFrame(
            list(image_path_proposed_var_dict.items()),
            columns=["image_path", "proposed_variable"],
        )
        image_path_var_df.to_csv(output_csv_path, index=False)
        print(f"Saved proposed variables to: {output_txt_path}")
        print(f"Saved per-image variables to: {output_csv_path}")

    elapsed_h = (time.time() - script_start) / 3600
    print(f"[time] Done. Total wall time: {elapsed_h:.2f} h "
          f"(stop_requested={STOP_REQUESTED['flag']})")


if __name__ == "__main__":
    main()
