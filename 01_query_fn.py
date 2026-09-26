import argparse
import os
import pandas as pd
import torch

from huggingface_hub import login
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from utils.utils import \
    create_qwen_dataloader, run_inference, \
     set_seed, fathomnet_df_to_msgs

def gen_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--login_token", type=str, default="no_login_token",
        help="Hugging Face login token.",
    )
    parser.add_argument(
        "--linux", action="store_true",
        help="Flag indicating if the script is running on a Linux system.",
    )
    parser.add_argument(
        "--compute_cluster", action="store_true",
        help="Flag indicating if the script is running on a computing cluster.",
    )
    parser.add_argument(
        "--test_mode", action="store_true",
        help="Flag to run the script in test mode.",
    )
    parser.add_argument(
        "--task_type", type=str,
        choices=["fn", "fn_cap", "fn_cls"],
        default="fn",
        help="Type of FathomNet task: 'fn' for variable proposal, "
             "'fn_cap' for captioning, 'fn_cls' for classification.",
    )
    parser.add_argument(
        "--image_type", type=str,
        choices=["main", "roi"],
        default="main",
        help="Image type: 'main' for full images, 'roi' for region-of-interest crops.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for dataloader.",
    )
    parser.add_argument(
        "--top_k_vars", type=int, default=25,
        help="Number of top proposed variables to consider for refinement.",
    )
    parser.add_argument(
        "--parameter_size", type=str,
        choices=["8B", "32B"], default="8B",
        help="Model parameter size to use.",
    )
    parser.add_argument(
        "--full_data", action="store_true",
        help="Flag to use the full dataset instead of just the training partition.",
    )

    args = parser.parse_args()
    return args


def main():

    args = gen_args()
    set_seed(args.seed)

    if args.login_token != "no_login_token":
        login(args.login_token)

    linux = args.linux
    computer_cluster = args.compute_cluster
    parameter_size = args.parameter_size
    test_mode = args.test_mode

    if test_mode:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    else:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    task_type = args.task_type
    image_type = args.image_type
    batch_size = args.batch_size

    if image_type == "main":
        df = pd.read_csv("./datasets/fathomnet-2025/image_annotations.csv")
    else:
        df = pd.read_csv("./datasets/fathomnet-2025/roi_annotations.csv")

    if not args.full_data:
        df = df[df["partition"] == "train"].reset_index(drop=True)

    if test_mode:
        df = df.sample(n=2 * batch_size, random_state=args.seed).reset_index(drop=True)

    if linux or computer_cluster:
        if computer_cluster:
            model_path = f"./models/Qwen3-VL-{parameter_size}-Instruct/snapshots/tensors/"
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

    messages = fathomnet_df_to_msgs(df, task_type=task_type, image_type=image_type)

    dataloader = create_qwen_dataloader(
        image_paths, messages, processor, model, batch_size=batch_size,
    )

    output_path = "./datasets/fathomnet-2025/outputs/"

    if "cap" in task_type or "cls" in task_type:
        output_df = run_inference(
            dataloader, model, processor, task_type=task_type, output_path=output_path
        )

        os.makedirs(output_path, exist_ok=True)
        if "cap" in task_type:
            file_identifier = "captions"
        elif "cls" in task_type:
            file_identifier = "predictions"
        output_csv_path = f"{output_path}{task_type}_{image_type}_{file_identifier}.csv"
        output_df.to_csv(output_csv_path, index=False)
    else:
        proposed_variables, image_path_proposed_var_dict = run_inference(
            dataloader, model, processor, task_type=task_type
        )

        top_k = args.top_k_vars
        top_k_proposed_vars = dict(
            sorted(proposed_variables.items(), key=lambda item: item[1], reverse=True)[:top_k]
        )

        print(f"Top {top_k} Proposed Variables:", top_k_proposed_vars)

        os.makedirs(output_path, exist_ok=True)
        output_txt_path = f"{output_path}{task_type}_{image_type}_out.txt"

        with open(output_txt_path, "w") as f:
            f.write("Top Proposed Variables:\n")
            for var, freq in top_k_proposed_vars.items():
                f.write(f"{var}: {freq}\n")

        output_csv_path = f"{output_path}{task_type}_{image_type}_image_path_proposed_var.csv"
        image_path_var_df = pd.DataFrame(
            list(image_path_proposed_var_dict.items()),
            columns=["image_path", "proposed_variable"],
        )
        image_path_var_df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    main()
