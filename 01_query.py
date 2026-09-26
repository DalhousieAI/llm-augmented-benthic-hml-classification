import argparse
import os
import pandas as pd
import torch

from huggingface_hub import login
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from utils.utils import create_qwen_dataloader, run_inference, run_refinement_inference, set_seed
from utils.utils import benthicnet_df_to_msgs

def gen_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--login_token",
        type=str,
        default="no_login_token",
        help="Hugging Face login token.",
    )
    parser.add_argument(
        "--linux",
        action="store_true",
        help="Flag indicating if the script is running on a Linux system.",
    )
    parser.add_argument(
        "--compute_cluster",
        action="store_true",
        help="Flag indicating if the script is running on a computing cluster.",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Flag to run the script in test mode.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["bn_sub", "bn_bio", "bn_sub_cap", "bn_bio_cap", "bn_sub_cls", "bn_bio_cls"],
        default="bn_sub",
        help="Type of BenthicNet task: 'bn_sub' for substrate classification, 'bn_bio' for biota classification.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for dataloader.",
    )
    parser.add_argument(
        "--top_k_vars",
        type=int,
        default=25,
        help="Number of top proposed variables to consider for refinement.",
    )
    parser.add_argument(
        "--parameter_size",
        type=str,
        choices=["8B", "32B"],
        default="8B",
        help="Model parameter size to use.",
    )
    parser.add_argument(
        "--full_data",
        action="store_true",
        help="Flag to use the full dataset instead of just the training partition.",
    )

    args = parser.parse_args()
    return args

def gen_captions():

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
    batch_size = args.batch_size

    if "bn_sub" in task_type:
        df = pd.read_csv("./datasets/benthicnet/benthicnet_substrates.csv")
    elif "bn_bio" in task_type:
        df = pd.read_csv("./datasets/benthicnet/benthicnet_biota.csv")

    if not args.full_data:
        df = df[df["partition"] == "train"].reset_index(drop=True)

    if test_mode:
        df = df.sample(n=2*batch_size, random_state=args.seed).reset_index(drop=True)

    if linux or computer_cluster:
        if computer_cluster:
            model_path = f"./models/Qwen3-VL-{parameter_size}-Instruct/snapshots/tensors/"
            root = f"{os.environ['SLURM_TMPDIR']}/"
        else:
            model_path = f"Qwen/Qwen3-VL-{parameter_size}-Instruct"
            root = "./datasets/benthicnet/images/"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            local_files_only=True,
        )
    else:
        model_path = f"Qwen/Qwen3-VL-{parameter_size}-Instruct"
        root = "./datasets/benthicnet/images/"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype="auto", device_map="auto",
        )

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=computer_cluster)
    processor.tokenizer.padding_side = "left"
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    image_paths = df["image_path"].to_list()

    messages = benthicnet_df_to_msgs(df, task_type=task_type, root=root)

    dataloader = create_qwen_dataloader(
            image_paths, messages, processor, model, batch_size=batch_size,
    )

    output_path = "./datasets/benthicnet/outputs/"

    if "cap" in task_type or "cls" in task_type:

        output_df = run_inference(dataloader, model, processor, task_type=task_type, output_path=output_path)

        os.makedirs(output_path, exist_ok=True)
        if "cap" in task_type:
            file_identifier = "captions"
        elif "cls" in task_type:
            file_identifier = "predictions"
        output_csv_path = f"{output_path}{task_type}_{file_identifier}.csv"
        output_df.to_csv(output_csv_path, index=False)
    else:
        proposed_variables, image_path_proposed_var_dict = run_inference(dataloader, model, processor, task_type=task_type)

        top_k = args.top_k_vars
        top_k_proposed_vars = dict(
            sorted(proposed_variables.items(), key=lambda item: item[1], reverse=True)[:top_k]
        )

        print(f"Top {top_k} Proposed Variables:", top_k_proposed_vars)

        refined_variables = run_refinement_inference(
            top_k_proposed_vars.keys(), model, processor
        )

        print("Refined Variables:", refined_variables)


        os.makedirs(output_path, exist_ok=True)
        output_txt_path = f"{output_path}{task_type}_out.txt"

        with open(output_txt_path, "w") as f:
            f.write("Top Proposed Variables:\n")
            for var, freq in top_k_proposed_vars.items():
                f.write(f"{var}: {freq}\n")
            f.write("\nRefined Variables:\n")
            for var in refined_variables:
                f.write(f"{var}\n")

        output_csv_path = f"{output_path}{task_type}_image_path_proposed_var.csv"
        image_path_var_df = pd.DataFrame(
            list(image_path_proposed_var_dict.items()),
            columns=["image_path", "proposed_variable"]
            )
        image_path_var_df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    gen_captions()
