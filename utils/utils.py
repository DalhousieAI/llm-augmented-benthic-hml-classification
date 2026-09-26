import ast
import itertools
import json
import networkx as nx
import math
import numpy as np
import os
import pandas as pd
import pickle
import re
import time
import torch

from numpy.linalg import norm
from scipy.optimize import root_scalar, linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.collator import QwenCollator
from utils.constants import \
    SUB_NODES, BIO_NODES, \
    SUB_KEYWORD_RULES, BIO_KEYWORD_RULES, \
    TAXONOMY_COLS
from utils.dataset import QwenChatDataset

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_prompt_text(
        task_type,
        text_input=None,
        colloquial_annotations=None,
        permitted_annotations=None,
        attributes_dict=None
        ):
    if "bn_sub" in task_type:
        identification_text = "substrate characteristics"
        specificity_guidance = (
            "Describe each substrate using standard geological and marine "
            "terminology. Include grain size, consolidation, and any visible "
            "surface features."
        )
    elif "bn_bio" in task_type:
        identification_text = "biota"
        specificity_guidance = (
            "Identify each organism to the lowest taxonomic level you can "
            "determine. Where exact taxonomy is uncertain, describe the "
            "morphotype."
        )
    if "cap" in task_type:
        assert attributes_dict is not None, "Attributes dictionary must be provided for captioning task."
        all_properties = sorted(set(
            prop for props in attributes_dict.values() for prop in props
        ))
        properties_str = ", ".join(all_properties)
        prompt_text = (
            "You are a marine benthic ecologist analyzing a seafloor photograph.\n\n"
            f"Task: For each ecological property below, carefully reason about whether it is "
            f"supported or contradicted by visible evidence of the {identification_text} in this image. "
            "Connect what you SEE to what you KNOW about benthic ecology. "
            "Score each property:\n"
            "  1 = visually supported\n"
            "  0 = not applicable or visually contradicted \n\n"
            f"Properties:\n{properties_str}\n\n"
            "Format: \n"
            "  [property_term]: [brief reasoning and justification (one sentence, ~5 words)] [score value] "
            "Adhere strictly to the format. Output one property per line."
        )
    elif "cls" in task_type:
        prompt_text = (
            f"Identify ALL distinct {identification_text} visible in the benthic image. "
            f"{specificity_guidance}\n\n"
            "Output ONLY the identified items, separated by commas. "
            "Do not repeat items."
        )
    elif "ref" in task_type:
        assert text_input is not None, "Text input must be provided for refinement task."
        assert permitted_annotations is not None, "Permitted annotations must be provided for refinement task."
        prompt_text = (
            f"Convert the annotations in {text_input} to the MOST APPROPRIATE node in {permitted_annotations}."
            "IGNORE predictions that do not have a good match. "
            "The output MUST NOT exceed the input in length. Annotations not in the provided list of nodes are NOT allowed. "
            "Return the final set of annotations in lower case, separated by commas, and WITHOUT conjunctions."
        )
    else:
        prompt_text = (
             "Task: Infer functional ecological properties that help distinguish "
            f"between different {identification_text}.\n\n"
            "Go beyond what a simple visual description would capture — "
            "use ecological reasoning to infer properties from what you observe. "
            "Focus especially on diagnostic properties where ecological and world knowledge "
            "would help resolve highly specific or rare categories.\n\n"
            "Do NOT return simple visual descriptions (e.g., color, shape, texture). "
            "Do not restate the annotation categories themselves. "
            "Return up to 10 of the most diagnostic properties as 1-3 lowercase words each, "
            "separated by commas. Return nothing else."
        )
        if colloquial_annotations is not None:
            prompt_text = (
                "You are a marine benthic ecologist.\n"
                f"This seafloor image has the following annotations: {colloquial_annotations}.\n"
                "Use these annotations to guide which ecological properties "
                "would be most diagnostic for identifying the annotated categories.\n\n"
            ) + prompt_text
    return prompt_text

def gen_variables_refinement_prompt(variables):
    prompt_text = (
        "You are a marine benthic ecologist. Refine the following list of variables "
        "generated from seafloor imagery.\n\n"
        "REMOVE:\n"
        "- Color, tone, brightness, or lighting terms (e.g. dark, mottled, shadowed, dimly lit)\n"
        "- Camera or visibility artifacts (e.g. low-visibility, translucent)\n"
        "- Classification labels that name what the substrate is (e.g. sedimentary, submerged)\n"
        "- Redundant synonyms, keeping only the most ecologically precise term from each group "
        "(e.g. keep 'granular' but remove 'gritty' and 'grainy')\n"
        "- Terms too vague to distinguish between habitat types (e.g. variable, uniform)\n\n"
        "KEEP:\n"
        "- Physical properties that affect what organisms can do: burrowing ease, attachment "
        "potential, water exchange, structural refuge\n"
        "- Properties implying biological activity (e.g. bioturbated)\n"
        "- Properties implying environmental process (e.g. wave-exposed, sediment-accumulating)\n"
        "- Habitat complexity or structural descriptors\n\n"
        "Do not add new terms. Return only terms from the original list. "
        "Return as 1-2 word lowercase terms separated by commas, "
        "without underscores or conjunctions.\n\n"
        f"Variables to refine: {', '.join(variables)}."
    )
    return prompt_text

def generate_message(
        image_path,
        task_type,
        root="",
        attributes=None,
        text_input=None,
        colloquial_annotations=None,
        permitted_annotations=None,
        ):
    if "cap" in task_type:
        if task_type == "bn_sub_cap":
            with open(
                "./datasets/benthicnet/extracted_vars/bn_sub_selected_keywords.json", "r"
                ) as f:
                attributes = json.load(f)
        elif task_type == "bn_bio_cap":
            with open(
                "./datasets/benthicnet/extracted_vars/bn_bio_selected_keywords.json", "r"
                ) as f:
                attributes = json.load(f)

    content = []
    if "ref" not in task_type:
        content.append({
            "type": "image",
            "image": root + image_path
        })
    content.append({
        "type": "text",
        "text": generate_prompt_text(
            task_type,
            attributes_dict=attributes,
            text_input=text_input,
            colloquial_annotations=colloquial_annotations,
            permitted_annotations=permitted_annotations
        )
    })

    message = []
    message.append({
        "role": "user",
        "content": content
    })

    return message

def gen_variables_refinement_message(variables):
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": gen_variables_refinement_prompt(variables)
                },
            ],
        }
    ]
    return message

def generate_colloquial_annotations(annotations, annotation_kw_map):
    annotations = annotations.split(";")
    colloquial_annotations = [annotation_kw_map[a.strip()] for a in annotations]
    return ", ".join(colloquial_annotations)


def benthicnet_df_to_msgs(df, task_type, root=""):
    image_paths = df["image_path"]
    if "bn_sub" in task_type:
        annotation_col = "catami_substrate"
        permitted_annotations = SUB_NODES
        keyword_rules = SUB_KEYWORD_RULES
    elif "bn_bio" in task_type:
        annotation_col = "catami_biota"
        permitted_annotations = BIO_NODES
        keyword_rules = BIO_KEYWORD_RULES

    if "ref" in task_type:
        annotation_col = "predictions"

    image_annotations = df[annotation_col]

    annotation_kw_map = {}
    for entry in keyword_rules:
        kw = entry[0]
        annotation = entry[1]
        annotation_kw_map[annotation] = kw
    df["colloquial_annotations"] = df.apply(
        lambda row: generate_colloquial_annotations(row[annotation_col], annotation_kw_map),
        axis=1
    )

    messages_df = pd.DataFrame({
        "image_path": image_paths,
        "annotation": image_annotations,
        "colloquial_annotations": df["colloquial_annotations"]
    })

    messages_df["messages"] = messages_df.apply(
        lambda row: generate_message(
            row["image_path"],
            task_type,
            root=root,
            text_input=row["annotation"] if "ref" in task_type else None,
            colloquial_annotations=row["colloquial_annotations"],
            permitted_annotations=permitted_annotations if "ref" in task_type else None,
        ),
        axis=1
    )

    return messages_df["messages"].tolist()

def generate_fn_prompt_text(
        task_type,
        text_input=None,
        colloquial_annotations=None,
        attributes_dict=None,
        ):
    identification_text = "marine organisms (biota)"
    specificity_guidance = (
        "Identify each organism to the lowest taxonomic level you can "
        "determine. Where exact taxonomy is uncertain, describe the "
        "morphotype."
    )

    if "cap" in task_type:
        assert attributes_dict is not None, (
            "Attributes dictionary must be provided for captioning task."
        )
        all_properties = sorted(set(
            prop for props in attributes_dict.values() for prop in props
        ))
        properties_str = ", ".join(all_properties)
        prompt_text = (
            "You are a marine biologist analyzing an underwater photograph.\n\n"
            f"Task: For each ecological property below, carefully reason about whether it is "
            f"supported or contradicted by visible evidence of the {identification_text} in this image. "
            "Connect what you SEE to what you KNOW about marine ecology. "
            "Score each property:\n"
            "  1 = visually supported\n"
            "  0 = not applicable or visually contradicted \n\n"
            f"Properties:\n{properties_str}\n\n"
            "Format: \n"
            "  [property_term]: [brief reasoning and justification (one sentence, ~5 words)] [score value] "
            "Adhere strictly to the format. Output one property per line."
        )
    elif "cls" in task_type:
        prompt_text = (
            f"Identify ALL distinct {identification_text} visible in this underwater image. "
            f"{specificity_guidance}\n\n"
            "Output ONLY the identified items, separated by commas. "
            "Do not repeat items."
        )
    elif "ref" in task_type:
        assert text_input is not None, (
            "Text input must be provided for refinement task."
        )
        prompt_text = (
            f"Convert the annotations in {text_input} to the most appropriate "
            "standard taxonomic names. "
            "IGNORE predictions that do not have a good match. "
            "The output MUST NOT exceed the input in length. "
            "Return the final set of annotations in lower case, separated by commas, "
            "and WITHOUT conjunctions."
        )
    else:
        prompt_text = (
            "Task: Infer functional ecological properties that help distinguish "
            f"between different {identification_text}.\n\n"
            "Go beyond what a simple visual description would capture — "
            "use ecological reasoning to infer properties from what you observe. "
            "Focus especially on diagnostic properties where ecological and world knowledge "
            "would help resolve highly specific or rare categories.\n\n"
            "Do NOT return simple visual descriptions (e.g., color, shape, texture). "
            "Do not restate the annotation categories themselves. "
            "Return up to 10 of the most diagnostic properties as 1-3 lowercase words each, "
            "separated by commas. Return nothing else."
        )
        if colloquial_annotations is not None:
            prompt_text = (
                "You are a marine biologist.\n"
                f"This underwater image has the following annotations: {colloquial_annotations}.\n"
                "Use these annotations to guide which ecological properties "
                "would be most diagnostic for identifying the annotated categories.\n\n"
            ) + prompt_text

    return prompt_text


def gen_fn_variables_refinement_prompt(variables):
    prompt_text = (
        "You are a marine biologist. Refine the following list of variables "
        "generated from underwater imagery.\n\n"
        "REMOVE:\n"
        "- Color, tone, brightness, or lighting terms (e.g. dark, mottled, shadowed, dimly lit)\n"
        "- Camera or visibility artifacts (e.g. low-visibility, translucent)\n"
        "- Classification labels that name what the organism is (e.g. fish, coral)\n"
        "- Redundant synonyms, keeping only the most ecologically precise term from each group\n"
        "- Terms too vague to distinguish between taxa (e.g. variable, uniform)\n\n"
        "KEEP:\n"
        "- Physical properties that affect ecological function: feeding mode, locomotion, "
        "attachment strategy, structural complexity\n"
        "- Properties implying biological activity (e.g. bioturbated, colonial)\n"
        "- Properties implying environmental association (e.g. deep-sea, hydrothermal)\n"
        "- Morphological or life history traits useful for taxonomic distinction\n\n"
        "Do not add new terms. Return only terms from the original list. "
        "Return as 1-2 word lowercase terms separated by commas, "
        "without underscores or conjunctions.\n\n"
        f"Variables to refine: {', '.join(variables)}."
    )
    return prompt_text


def gen_fn_variables_refinement_message(variables):
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": gen_fn_variables_refinement_prompt(variables),
                },
            ],
        }
    ]
    return message


def generate_fn_message(
        image_path,
        task_type,
        attributes=None,
        text_input=None,
        colloquial_annotations=None,
        ):
    if "cap" in task_type:
        with open(
            "./datasets/fathomnet-2025/extracted_vars/fn_selected_keywords.json", "r"
        ) as f:
            attributes = json.load(f)

    content = []
    if "ref" not in task_type:
        content.append({
            "type": "image",
            "image": image_path,
        })
    content.append({
        "type": "text",
        "text": generate_fn_prompt_text(
            task_type,
            attributes_dict=attributes,
            text_input=text_input,
            colloquial_annotations=colloquial_annotations,
        ),
    })

    message = [{"role": "user", "content": content}]
    return message


def parse_fn_labels(labels_str):
    labels_list = ast.literal_eval(labels_str)
    return ", ".join(labels_list)


def fathomnet_df_to_msgs(df, task_type, image_type="main"):
    image_paths = df["path"]

    if image_type == "main":
        df = df.copy()
        df["colloquial_annotations"] = df["labels"].apply(parse_fn_labels)
    else:
        df = df.copy()
        df["colloquial_annotations"] = df["label"]

    messages_df = pd.DataFrame({
        "image_path": image_paths,
        "annotation": df["colloquial_annotations"],
        "colloquial_annotations": df["colloquial_annotations"],
    })

    messages_df["messages"] = messages_df.apply(
        lambda row: generate_fn_message(
            row["image_path"],
            task_type,
            text_input=row["annotation"] if "ref" in task_type else None,
            colloquial_annotations=row["colloquial_annotations"],
        ),
        axis=1,
    )

    return messages_df["messages"].tolist()

def build_colloquial_annotation(row, granularity="family_species"):
    if granularity == "species_only":
        return row["species"]
    elif granularity == "family_species":
        return f"{row['family']}: {row['species']}"
    elif granularity == "order_family_species":
        return f"{row['order']}, {row['family']}: {row['species']}"
    elif granularity == "full":
        return (
            f"{row['kingdom']} > {row['phylum']} > {row['class']} > "
            f"{row['order']} > {row['family']} > {row['species']}"
        )
    else:
        return row["species"]

def generate_inat_prompt_text(
        task_type,
        text_input=None,
        colloquial_annotations=None,
        permitted_annotations=None,
        attributes_dict=None,
):

    identification_text = "organism (plant, animal, or fungus)"
    specificity_guidance = (
        "Identify each organism to the lowest taxonomic level you can "
        "determine from visual evidence. Where exact species-level identification "
        "is uncertain, provide the most specific rank possible (genus, family, etc.) "
        "and note key morphological features that informed your decision."
    )

    if "cap" in task_type:
        assert attributes_dict is not None, (
            "Attributes dictionary must be provided for captioning task."
        )
        all_properties = sorted(set(
            prop for props in attributes_dict.values() for prop in props
        ))
        properties_str = ", ".join(all_properties)
        prompt_text = (
            "You are a field naturalist and taxonomic expert analyzing a "
            "nature photograph.\n\n"
            f"Task: For each ecological property below, carefully reason about whether it is "
            f"supported or contradicted by visible evidence of the {identification_text} in this image. "
            "Connect what you SEE to what you KNOW about ecology, natural history, and taxonomy. "
            "Score each property:\n"
            "  1 = visually supported\n"
            "  0 = not applicable or visually contradicted\n\n"
            f"Properties:\n{properties_str}\n\n"
            "Format:\n"
            "  [property_term]: [brief reasoning and justification (one sentence, ~5 words)] [score value]\n"
            "Adhere strictly to the format. Output one property per line."
        )

    elif "cls" in task_type:
        prompt_text = (
            f"Identify the primary {identification_text} visible in this nature photograph. "
            f"{specificity_guidance}\n\n"
            "Output ONLY the identified taxon. Use the scientific name where possible."
        )

    elif "ref" in task_type:
        assert text_input is not None, (
            "Text input must be provided for refinement task."
        )
        assert permitted_annotations is not None, (
            "Permitted annotations must be provided for refinement task."
        )
        prompt_text = (
            f"Map the predicted taxon '{text_input}' to the single MOST APPROPRIATE "
            f"node in the following label set: {permitted_annotations}.\n"
            "If no good match exists, return 'none'.\n"
            "Return exactly one label from the list, in lower case, with no additional text."
        )

    else:
        prompt_text = (
            "Task: Infer functional ecological properties that help distinguish "
            f"this {identification_text} from taxonomically similar species.\n\n"
            "Go beyond what a simple visual description would capture — "
            "use ecological reasoning to infer properties from what you observe. "
            "Focus especially on diagnostic properties where ecological knowledge, "
            "biogeography, and life history would help resolve highly specific or "
            "rare categories.\n\n"
            "Do NOT return simple visual descriptions (e.g., color, shape, texture). "
            "Do not restate the annotation categories themselves. "
            "Return up to 10 of the most diagnostic properties as 1-3 lowercase words each, "
            "separated by commas. Return nothing else."
        )
        if colloquial_annotations is not None:
            prompt_text = (
                "You are an expert field naturalist and taxonomist.\n"
                f"This nature photograph has the following annotation: {colloquial_annotations}.\n"
                "Use this annotation to guide which ecological properties "
                "would be most diagnostic for identifying the organism.\n\n"
            ) + prompt_text

    return prompt_text


def gen_inat_variables_refinement_prompt(variables):
    prompt_text = (
        "You are a biodiversity scientist. Refine the following list of variables "
        "generated from iNaturalist nature photographs.\n\n"
        "REMOVE:\n"
        "- Color, tone, brightness, or lighting terms (e.g. dark, mottled, shadowed, vivid)\n"
        "- Camera or photographic artifacts (e.g. blurred, overexposed, cropped)\n"
        "- Classification labels that name what the organism is (e.g. insect, flowering plant)\n"
        "- Redundant synonyms, keeping only the most ecologically precise term from each group "
        "(e.g. keep 'deciduous' but remove 'leaf-shedding')\n"
        "- Terms too vague to distinguish between taxa (e.g. variable, common, natural)\n\n"
        "KEEP:\n"
        "- Functional traits that affect ecological role: feeding strategy, growth form, "
        "dispersal mode, pollination syndrome, thermoregulation\n"
        "- Properties implying habitat association (e.g. epiphytic, fossorial, riparian)\n"
        "- Properties implying life history strategy (e.g. annual, r-selected, semelparous)\n"
        "- Properties implying biogeographic or phenological context (e.g. tropical, "
        "spring-flowering, migratory)\n"
        "- Morphological traits with clear taxonomic diagnostic value (e.g. zygomorphic, "
        "endothermic, radially symmetric)\n\n"
        "Do not add new terms. Return only terms from the original list. "
        "Return as 1-2 word lowercase terms separated by commas, "
        "without underscores or conjunctions.\n\n"
        f"Variables to refine: {', '.join(variables)}."
    )
    return prompt_text


def gen_inat_variables_refinement_message(variables):
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": gen_inat_variables_refinement_prompt(variables),
                },
            ],
        }
    ]
    return message


def generate_inat_message(
        image_path,
        task_type,
        attributes=None,
        text_input=None,
        colloquial_annotations=None,
        permitted_annotations=None,
):
    if "cap" in task_type and attributes is None:
        with open(
            "./datasets/inat/extracted_vars/inat_selected_keywords.json", "r"
        ) as f:
            attributes = json.load(f)

    content = []
    if "ref" not in task_type:
        content.append({
            "type": "image",
            "image": image_path,
        })
    content.append({
        "type": "text",
        "text": generate_inat_prompt_text(
            task_type,
            attributes_dict=attributes,
            text_input=text_input,
            colloquial_annotations=colloquial_annotations,
            permitted_annotations=permitted_annotations,
        ),
    })

    message = [{"role": "user", "content": content}]
    return message


def parse_inat_labels(labels_input):
    if isinstance(labels_input, list):
        return ", ".join(labels_input)
    labels_str = str(labels_input).strip()
    if labels_str.startswith("["):
        labels_list = ast.literal_eval(labels_str)
        return ", ".join(labels_list)
    return labels_str


def inat_df_to_msgs(
        df,
        task_type,
        label_col="species",
        image_col="path",
        permitted_annotations=None,
):
    df = df.copy()
    image_paths = df[image_col]

    if "ref" in task_type:
        annotation_col = "predictions" if "predictions" in df.columns else label_col
    else:
        annotation_col = label_col

    df["annotation"] = df[annotation_col].apply(parse_inat_labels)

    messages_df = pd.DataFrame({
        "image_path": image_paths,
        "annotation": df["annotation"],
        "colloquial_annotations": df["colloquial_annotations"],
    })

    messages_df["messages"] = messages_df.apply(
        lambda row: generate_inat_message(
            row["image_path"],
            task_type,
            text_input=row["annotation"] if "ref" in task_type else None,
            colloquial_annotations=row["colloquial_annotations"],
            permitted_annotations=permitted_annotations if "ref" in task_type else None,
        ),
        axis=1,
    )

    return messages_df["messages"].tolist()

def create_qwen_dataloader(
        image_paths,
        messages,
        processor,
        model,
        batch_size=64,
        shuffle=False,
        prefetch_factor=1,
        num_workers=0,
        drop_last=False,
    ):
    if num_workers == 0:
        prefetch_factor = None


    dataset = QwenChatDataset(image_paths, messages)
    collator = QwenCollator(processor, model)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    return dataloader

def run_inference(
        dataloader,
        model,
        processor,
        task_type,
        output_path=None,
        max_allocated_time=86400,
        deadline=None,
        stop_flag=None,
        checkpoint_every=50,
        checkpoint_csv=None,
        append_mode=False,
):
    model.eval()

    proposed_variable_set = dict()
    image_path_proposed_var_dict = dict()

    output_task = any(x in task_type for x in ("cap", "cls", "ref"))

    col_header = None
    if "cap" in task_type:
        col_header = "captions"
    elif "cls" in task_type:
        col_header = "predictions"
    elif "ref" in task_type:
        col_header = "refined_predictions"

    if output_task:
        outputs_df = pd.DataFrame(columns=["image_path", col_header])

    start_time = time.time()

    if deadline is None:
        effective_deadline = start_time + 0.98 * max_allocated_time
    else:
        effective_deadline = deadline

    pending_rows = []
    header_written = append_mode

    if output_task and checkpoint_csv is None and output_path is not None:
        checkpoint_csv = (
            f"{output_path}{task_type}_{col_header}_intermediate.csv"
        )
    propose_ckpt_path = None
    if (not output_task) and output_path is not None:
        propose_ckpt_path = f"{output_path}{task_type}_proposed_intermediate.pkl"

    def _flush_outputs(final=False):
        nonlocal header_written, outputs_df, pending_rows
        if not output_task or checkpoint_csv is None:
            return
        if not pending_rows:
            return
        df_part = pd.DataFrame(pending_rows, columns=["image_path", col_header])
        outputs_df = pd.concat([outputs_df, df_part], ignore_index=True)
        write_header = (not header_written) and (not os.path.exists(checkpoint_csv))
        df_part.to_csv(
            checkpoint_csv,
            mode="a",
            header=write_header,
            index=False,
        )
        header_written = True
        pending_rows = []
        if final:
            print(f"[run_inference] Final flush -> {checkpoint_csv} "
                  f"({len(outputs_df)} rows total in memory).")

    def _flush_proposed():
        if propose_ckpt_path is None:
            return
        try:
            with open(propose_ckpt_path, "wb") as f:
                pickle.dump(
                    {
                        "proposed_variable_set": proposed_variable_set,
                        "image_path_proposed_var_dict": image_path_proposed_var_dict,
                    },
                    f,
                )
        except Exception as e:
            print(f"[run_inference] Failed to pickle proposed checkpoint: {e}")

    average_sample_processing_time = []
    stopped_early = False

    with torch.no_grad():
        for batch_idx, (image_paths, msgs) in enumerate(tqdm(dataloader)):
            if stop_flag is not None and stop_flag.get("flag"):
                print(f"\n[run_inference] Stop flag set before batch "
                      f"{batch_idx}; breaking out.")
                stopped_early = True
                break

            now = time.time()
            if now >= effective_deadline:
                print(f"\n[run_inference] Deadline reached before batch "
                      f"{batch_idx} (elapsed {now - start_time:.1f}s); "
                      f"breaking out.")
                if stop_flag is not None:
                    stop_flag["flag"] = True
                stopped_early = True
                break

            if average_sample_processing_time:
                avg_t = (sum(average_sample_processing_time)
                         / len(average_sample_processing_time))
                projected_end = now + avg_t * len(image_paths)
                if projected_end >= effective_deadline:
                    print(f"\n[run_inference] Projected batch end "
                          f"({projected_end - start_time:.1f}s) exceeds "
                          f"deadline; stopping before batch {batch_idx}.")
                    if stop_flag is not None:
                        stop_flag["flag"] = True
                    stopped_early = True
                    break

            batch_start_time = time.time()

            generated_ids = model.generate(
                **msgs,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.8,
                top_k=20,
                temperature=0.7,
                repetition_penalty=1.0,
            )

            input_len = msgs.input_ids.shape[1]
            generated_ids_trimmed = generated_ids[:, input_len:]
            outputs = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            average_sample_processing_time.append(batch_time / len(image_paths))

            if output_task:
                for ip, out in zip(image_paths, outputs):
                    pending_rows.append({"image_path": ip, col_header: out})

                if checkpoint_every and ((batch_idx + 1) % checkpoint_every == 0):
                    _flush_outputs(final=False)
            else:
                proposed_variable_set = outputs_to_freq_count_dict(
                    proposed_variable_set, outputs
                )
                for img_path, output in zip(image_paths, outputs):
                    image_path_proposed_var_dict[img_path] = output

                if checkpoint_every and ((batch_idx + 1) % checkpoint_every == 0):
                    _flush_proposed()

        print()

    if output_task:
        _flush_outputs(final=True)
    else:
        _flush_proposed()

    if average_sample_processing_time:
        avg_t = (sum(average_sample_processing_time)
                 / len(average_sample_processing_time))
        print("Average sample processing time: {:.2f} seconds".format(avg_t))
        print("Average sample processing speed: {:.2f} samples/second".format(
            1.0 / avg_t))
    else:
        print("No batches were processed.")

    end_time = time.time()
    print(f"Inference {'stopped early' if stopped_early else 'completed'} "
          f"in {end_time - start_time:.2f} seconds.")

    if output_task:
        return outputs_df

    return proposed_variable_set, image_path_proposed_var_dict


def run_refinement_inference(variables, model, processor):
    model.eval()
    with torch.no_grad():
        print("Refining proposed variables...")
        refinement_instruction = gen_variables_refinement_message(
            variables
        )
        inputs = processor.apply_chat_template(
            refinement_instruction,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )

        inputs = inputs.to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            top_p=0.8,
            top_k=20,
            temperature=0.7,
            repetition_penalty=1.0,
        )

        generated_ids_trimmed = generated_ids[0][len(inputs.input_ids[0]):]

        refined_variables = processor.decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    refined_variables = outputs_to_unique_set([refined_variables])

    return refined_variables

def outputs_to_freq_count_dict(freq_dict, outputs):
    for output in outputs:
        variable_set = outputs_to_unique_set([output])
        for var in variable_set:
            if var in freq_dict:
                freq_dict[var] += 1
            else:
                freq_dict[var] = 1
    return freq_dict

def outputs_to_unique_set(outputs):
    return set(
        i.strip().lower()
        for i in itertools.chain.from_iterable(
            s.split(",") for s in outputs
        )
        if i.strip()
    )

def extract_unique_nodes(hierarchy_text):
    nodes = set()

    for line in hierarchy_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        if "->" in line:
            parent, children = line.split("->", 1)
            nodes.add(parent.strip())

            for child in children.split(","):
                if child not in nodes:
                    nodes.add(child.strip())
                else:
                    print(f"Warning: Duplicate node '{child.strip()}' found in hierarchy.")
        else:
            nodes.add(line)

    return sorted(nodes)

def build_benthicnet_hierarchy(annotations):
    children_dict = {}

    annotations.apply(
        lambda ann: extract_hierarchy_benthicnet(ann, children_dict)
    )

    node_id_dict = assign_node_ids(children_dict)

    graph = dict_to_graph(children_dict)

    children_id_dict = translate_text_dict_to_id_dict(children_dict, node_id_dict)

    distance_matrix = gen_distance_matrix(graph, node_id_dict)

    visual_text_graph = dict_to_visual_text_graph(children_dict)

    node_colloquial_names = get_node_colloquial_names(children_dict)

    descendant_matrix = compute_descendant_matrix(graph, node_id_dict)

    return {
        "children_dict": children_dict,
        "node_id_dict": node_id_dict,
        "graph": graph,
        "children_id_dict": children_id_dict,
        "distance_matrix": distance_matrix,
        "visual_text_graph": visual_text_graph,
        "max_distance_dict": maximum_distances_for_graph(distance_matrix, node_id_dict),
        "node_colloquial_names": node_colloquial_names,
        "descendant_matrix": descendant_matrix
    }

def extract_hierarchy_benthicnet(annotation, children_dict):
    components = annotation.split("; ")

    for component in components:
        nodes = component.split(" > ")
        parent_path = None

        for i in range(len(nodes)):
            current_path = " > ".join(nodes[:i+1])

            if current_path not in children_dict:
                children_dict[current_path] = []

            if parent_path is not None:
                if current_path not in children_dict[parent_path]:
                    children_dict[parent_path].append(current_path)

            parent_path = current_path

def assign_node_ids(children_dict):
    node_id_dict = dict()
    current_id = 0
    for parent, children in children_dict.items():
        if parent not in node_id_dict:
            node_id_dict[parent] = current_id
            current_id += 1
        for child in children:
            if child not in node_id_dict:
                node_id_dict[child] = current_id
                current_id += 1
    return node_id_dict

def dict_to_graph(children_dict):
    G = nx.DiGraph()
    for parent, children in children_dict.items():
        if not children and parent not in G:
            G.add_node(parent)
            continue
        for child in children:
            G.add_edge(parent, child)
    return G

def dict_to_visual_text_graph(children_dict):
    visual_text_graph = ""
    for parent, children in children_dict.items():
        parent = parent.split(" > ")[-1]
        if not children:
            continue
        visual_text_graph += f"{parent} -> "
        for child in children:
            child = child.split(" > ")[-1]
            visual_text_graph += f"{child}, "
        visual_text_graph = visual_text_graph.rstrip(", ") + "\n"
    return visual_text_graph

def translate_text_dict_to_id_dict(text_dict, node_id_dict):
    id_dict = dict()
    for parent, children in text_dict.items():
        parent_id = node_id_dict[parent]
        children_ids = [node_id_dict[child] for child in children]
        id_dict[parent_id] = children_ids
    return id_dict

def gen_distance_matrix(graph, node_id_dict):
    undirected_graph = graph.to_undirected(as_view=True)

    num_nodes = len(node_id_dict)
    distance_matrix = np.full((num_nodes, num_nodes), np.inf)

    for source in undirected_graph.nodes:
        lengths = nx.single_source_shortest_path_length(
            undirected_graph, source
        )
        source_id = node_id_dict[source]
        for target, length in lengths.items():
            target_id = node_id_dict[target]
            distance_matrix[source_id, target_id] = length

    return distance_matrix

def maximum_distances_for_graph(distance_matrix, node_id_dict):
    dist_mat = distance_matrix.copy()
    dist_mat[dist_mat == np.inf] = -1
    max_distance_dict = {}
    max_distance_list = np.max(dist_mat, axis=1)

    for _, node_id in node_id_dict.items():
        max_distance_dict[node_id] = int(max_distance_list[node_id])
    return max_distance_dict

def get_node_colloquial_names(children_dict):
    unique_nodes = set()
    for parent, children in children_dict.items():
        unique_nodes.add(parent)
        for child in children:
            unique_nodes.add(child)

    unique_nodes_list = sorted(list(unique_nodes))

    colloquial_names = {}
    seen_count = {}
    index_map = {}
    for i, node in enumerate(unique_nodes_list):
        node_parts = node.split(" > ")
        if len(node_parts) == 1:
            potential_name = node_parts[0]
        elif len(node_parts) > 1 and len(node_parts) <= 3:
            potential_name = " ".join(reversed(node_parts[1:]))
        else:
            potential_name = " ".join(reversed(node_parts[-3:]))

        if potential_name not in seen_count.keys():
            colloquial_names[node] = potential_name
            seen_count[potential_name] = 1
            index_map[potential_name] = [i]
        else:
            colloquial_names[node] = potential_name
            seen_count[potential_name] += 1
            index_map[potential_name].append(i)

    for name, count in seen_count.items():
        if count > 1:
            indices = index_map[name]
            original_nodes = [unique_nodes_list[i] for i in indices]
            join_index = 1
            new_potential_names = [
                " ".join(reversed(n.split(" > ")[join_index:]))
                for n in original_nodes
            ]

            for i, original_node in enumerate(original_nodes):
                colloquial_names[original_node] = new_potential_names[i]

    for node in colloquial_names.keys():
        colloquial_names[node] = colloquial_names[node].lower()

    return colloquial_names

def compute_descendant_matrix(graph, node_id_dict):
    num_nodes = len(node_id_dict)
    descendant_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for source in graph.nodes:
        descendants = nx.descendants(graph, source)
        source_id = node_id_dict[source]
        for descendant in descendants:
            descendant_id = node_id_dict[descendant]
            descendant_matrix[source_id, descendant_id] = 1

    np.fill_diagonal(descendant_matrix, 1)

    return descendant_matrix

def annotation_to_bin_vector(
        annotation,
        node_id_dict,
        descendant_matrix
        ):

    bin_vector = np.zeros(len(node_id_dict), dtype=int)
    annotation_components = annotation.split("; ")
    for a in annotation_components:
        idx = node_id_dict[a]
        bin_vector[idx] = 1

    bin_vector = np.max(descendant_matrix*bin_vector, axis=-1)

    return bin_vector

def extract_hierarchy_fathomnet(row, children_dict, idx_to_label):
    nodes = []
    for col in TAXONOMY_COLS:
        val = row[col]
        if pd.notna(val):
            nodes.append(idx_to_label[str(int(val))])
        else:
            break

    if not nodes:
        return

    if nodes[0] not in children_dict:
        children_dict[nodes[0]] = []

    parent = nodes[0]
    for node_name in nodes[1:]:
        if node_name not in children_dict:
            children_dict[node_name] = []
        if node_name not in children_dict[parent]:
            children_dict[parent].append(node_name)
        parent = node_name


def assign_node_ids_fathomnet(children_dict, label_to_idx):
    node_id_dict = {}

    all_nodes = set()
    for parent, children in children_dict.items():
        all_nodes.add(parent)
        for child in children:
            all_nodes.add(child)

    for node in all_nodes:
        if node not in label_to_idx:
            raise ValueError(f"Node '{node}' not found in label_to_idx mapping")
        node_id_dict[node] = label_to_idx[node]

    return node_id_dict


def dict_to_visual_text_graph_fathomnet(children_dict):
    visual_text_graph = ""
    for parent, children in children_dict.items():
        if not children:
            continue
        visual_text_graph += f"{parent} -> {', '.join(children)}\n"
    return visual_text_graph


def annotation_to_bin_vector_fathomnet(annotation, node_id_dict, descendant_matrix):
    bin_vector = np.zeros(len(node_id_dict), dtype=int)

    for node_id in annotation:
        bin_vector[node_id] = 1

    bin_vector = np.max(descendant_matrix * bin_vector, axis=-1)

    return bin_vector


def build_fathomnet_hierarchy(full_annotations, idx_to_label, label_to_idx):
    children_dict = {}

    taxonomy_unique = full_annotations[TAXONOMY_COLS].drop_duplicates()
    taxonomy_unique.apply(
        lambda row: extract_hierarchy_fathomnet(row, children_dict, idx_to_label),
        axis=1
    )

    idx_to_label = {int(k): v for k, v in idx_to_label.items()}
    label_to_idx = {v: int(k) for k, v in idx_to_label.items()}

    node_id_dict = assign_node_ids_fathomnet(children_dict, label_to_idx)
    graph = dict_to_graph(children_dict)
    children_id_dict = translate_text_dict_to_id_dict(children_dict, node_id_dict)
    distance_matrix = gen_distance_matrix(graph, node_id_dict)
    visual_text_graph = dict_to_visual_text_graph_fathomnet(children_dict)
    descendant_matrix = compute_descendant_matrix(graph, node_id_dict)

    return {
        "children_dict": children_dict,
        "node_id_dict": node_id_dict,
        "graph": graph,
        "children_id_dict": children_id_dict,
        "distance_matrix": distance_matrix,
        "visual_text_graph": visual_text_graph,
        "max_distance_dict": maximum_distances_for_graph(distance_matrix, node_id_dict),
        "descendant_matrix": descendant_matrix,
    }

def solve_b(k, n_ext_max, b_min=1e-6, b_max=100):

    if not (0 < k < 1):
        raise ValueError("k must be in (0, 1)")
    if n_ext_max <= 1:
        raise ValueError("n_ext_max must be > 1")

    def g(b):
        exp_bn = np.exp(-b * n_ext_max)
        return (np.exp(-b) - exp_bn) / (1 - exp_bn) - k

    sol = root_scalar(
        g,
        bracket=[b_min, b_max],
        method="brentq"
    )

    if not sol.converged:
        raise RuntimeError("Root finding did not converge")

    return sol.root

def multiplicity_factor(
        n_ext,
        n_ext_max,
        k,
):

    if n_ext >= n_ext_max:
        return 0

    if k == (n_ext_max - 1) / n_ext_max:
        return 1-n_ext / n_ext_max

    b = solve_b(k, n_ext_max)
    a = 1.0 / (1.0 - np.exp(-b * n_ext_max))
    c = 1.0 - a

    return a * np.exp(-b * n_ext) + c

def max_unique_sum(A):
    rows, cols = linear_sum_assignment(-A)
    return A[rows, cols].sum(), list(zip(rows, cols))

def inv_hml_sample_score(distance_matrix, node_id_dict, true_labels, pred_labels, max_distance_dict):
    if isinstance(true_labels[0], str):
        true_labels = [node_id_dict[label] for label in true_labels if label in node_id_dict]
    if isinstance(pred_labels[0], str):
        pred_labels = [node_id_dict[label] for label in pred_labels if label in node_id_dict]

    score_mat = []
    num_truths = len(true_labels)
    num_preds = len(pred_labels)
    num_nodes = len(distance_matrix)

    for true_label in true_labels:
        max_dist = max_distance_dict[true_label]
        distances = distance_matrix[true_label, pred_labels]
        distances = [d if d != np.inf else max_dist for d in distances]
        scores = [(max_dist-d)/max_dist if max_dist != 0 else 0 for d in distances]

        score_mat.append(scores)

    score_mat = np.array(score_mat)
    attained_score = max_unique_sum(score_mat)
    if num_preds < num_truths:
        multiplicity_component = 1
    else:
        multiplicity_component = multiplicity_factor(
            n_ext=num_preds - num_truths,
            n_ext_max=num_nodes - num_truths,
            k=0.5
        )
    return multiplicity_component * attained_score[0]/num_truths

def get_most_specific_nodes(active_nodes, descendant_matrix):
    if len(active_nodes) <= 1:
        return active_nodes

    active = np.array(active_nodes)
    sub_matrix = descendant_matrix[np.ix_(active, active)].copy()
    np.fill_diagonal(sub_matrix, 0)
    is_ancestor = sub_matrix.sum(axis=1) > 0
    return active[~is_ancestor].tolist()

def compute_avg_inv_hml(distance_matrix, node_id_dict, test_annotations, preds, max_distance_dict, descendant_matrix):
    scores = []
    for i in range(len(test_annotations)):
        true_labels_i = np.where(test_annotations[i] == 1)[0].tolist()
        pred_labels_i = np.where(preds[i] == 1)[0].tolist()

        true_labels_i = get_most_specific_nodes(true_labels_i, descendant_matrix)
        pred_labels_i = get_most_specific_nodes(pred_labels_i, descendant_matrix)

        if len(pred_labels_i) == 0:
            scores.append(0.0)
            continue

        if len(true_labels_i) == 0:
            raise ValueError(f"Sample {i} has no true labels - check annotation parsing.")

        score = inv_hml_sample_score(
            distance_matrix, node_id_dict,
            true_labels_i, pred_labels_i,
            max_distance_dict,
        )
        scores.append(score)
    return np.mean(scores), scores

def normalize_label(label):
    s = label.lower()
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def node_f1(labels, predictions):
    y = torch.as_tensor(labels).bool()
    p = torch.as_tensor(predictions).bool()

    tp = (y & p).sum(dim=0).to(torch.float64)
    fp = (~y & p).sum(dim=0).to(torch.float64)
    fn = (y & ~p).sum(dim=0).to(torch.float64)

    denom_p = tp + fp
    denom_r = tp + fn
    precision = torch.where(denom_p > 0, tp / denom_p, torch.zeros_like(tp))
    recall    = torch.where(denom_r > 0, tp / denom_r, torch.zeros_like(tp))
    denom_f   = precision + recall
    f1 = torch.where(denom_f > 0, 2 * precision * recall / denom_f, torch.zeros_like(tp))

    macro_f1 = f1.mean().item()
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()

    f1_list = f1.tolist()
    prec_list = precision.tolist()
    rec_list = recall.tolist()
    node_f1_dict = dict(enumerate(f1_list))
    node_precision_dict = dict(enumerate(prec_list))
    node_recall_dict = dict(enumerate(rec_list))

    return (macro_f1, macro_precision, macro_recall,
            node_f1_dict, node_precision_dict, node_recall_dict)

import ast
from tqdm import tqdm


def em_decompose(
        keyword_dict, node_list, delimiter="; ", tol=1e-3, max_iter=10000, fathomnet=False
        ):
    def parse_label(label):
        if fathomnet:
            if isinstance(label, (list, tuple)):
                return list(label)
            parsed = ast.literal_eval(label)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        else:
            return [l.strip() for l in label.split(delimiter)]

    counts = {}
    multi_label_entries = {}

    for keyword, label_counts in keyword_dict.items():
        counts[keyword] = {node: 0.0 for node in node_list}
        multi_label_entries[keyword] = []
        for label, count in label_counts.items():
            individual = parse_label(label)
            if len(individual) == 1:
                counts[keyword][individual[0]] += count
            else:
                multi_label_entries[keyword].append((individual, count))

    for i in tqdm(range(max_iter)):
        new_counts = {kw: {node: 0.0 for node in node_list} for kw in keyword_dict}

        for keyword in keyword_dict:
            for label, count in keyword_dict[keyword].items():
                individual = parse_label(label)
                if len(individual) == 1:
                    new_counts[keyword][individual[0]] += count

        for keyword, entries in multi_label_entries.items():
            for labels, count in entries:
                weights = {l: max(counts[keyword][l], 1e-10) for l in labels}
                total = sum(weights.values())
                for l in labels:
                    if total > 0:
                        new_counts[keyword][l] += count * (weights[l] / total)

        max_delta = 0.0
        for keyword in counts:
            all_labels = set(counts[keyword].keys()) | set(new_counts[keyword].keys())
            for label in all_labels:
                delta = abs(counts[keyword].get(label, 0.0) - new_counts[keyword].get(label, 0.0))
                max_delta = max(max_delta, delta)

        counts = new_counts

        if max_delta < tol:
            print(f"Converged after {i + 1} iterations (max delta: {max_delta:.2e})")
            break
    else:
        print(f"Did not converge after {max_iter} iterations (max delta: {max_delta:.2e})")

    return {kw: {label: (v if v > 1e-6 else 0.0) for label, v in c.items()} for kw, c in counts.items()}

def kl_scoring(em_counts_dict, node_prior_dict, prior):
    smoothed_counts = {}
    for keyword, label_counts in em_counts_dict.items():
        smoothed_counts[keyword] = {}
        for label, count in label_counts.items():
            smoothed_counts[keyword][label] = count + prior * node_prior_dict[label]

    normalized_smoothed_counts = {}
    for keyword, label_counts in smoothed_counts.items():
        total = sum(label_counts.values())
        normalized_smoothed_counts[keyword] = {label: count / total for label, count in label_counts.items()}

    kl_scores = {}
    for keyword, label_probs in normalized_smoothed_counts.items():
        kl_score = 0.0
        for label, prob in label_probs.items():
            if prob > 0:
                kl_score += prob * math.log(prob / node_prior_dict[label])
        kl_scores[keyword] = kl_score

    return kl_scores

def deduplicate_by_coverage(var_df, kl_scores_dict, var_col="proposed_variable",
                         image_col="image_path", similarity_threshold=0.8):
    keyword_to_images = {}
    for _, row in var_df.iterrows():
        for kw in row[var_col].split(", "):
            kw = kw.strip().lower()
            if kw in kl_scores_dict:
                keyword_to_images.setdefault(kw, set()).add(row[image_col])

    sorted_keywords = sorted(kl_scores_dict.keys(), key=lambda k: -kl_scores_dict[k])

    kept = []
    kept_image_sets = []

    for kw in tqdm(sorted_keywords):
        if kw not in keyword_to_images:
            continue
        img_set = keyword_to_images[kw]
        is_duplicate = False
        for prev_set in kept_image_sets:
            overlap = len(img_set & prev_set)
            union = len(img_set | prev_set)
            if union > 0 and overlap / union >= similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(kw)
            kept_image_sets.append(img_set)

    removed = set(kl_scores_dict.keys()) - set(kept)
    print(f"Kept {len(kept)} / {len(kl_scores_dict)} keywords, removed {len(removed)} near-duplicates")

    return {kw: kl_scores_dict[kw] for kw in kept}

def deduplicate_by_distribution(kl_scores, em_counts, similarity_threshold=0.99):
    nodes = sorted({n for counts in em_counts.values() for n in counts})

    def to_vec(keyword):
        total = sum(em_counts[keyword].values())
        if total == 0:
            return np.zeros(len(nodes))
        return np.array([em_counts[keyword].get(n, 0) / total for n in nodes])

    sorted_keywords = sorted(kl_scores.keys(), key=lambda k: -kl_scores[k])

    kept = []
    kept_vecs = []

    for kw in sorted_keywords:
        vec = to_vec(kw)
        is_duplicate = False
        for prev_vec in kept_vecs:
            cos_sim = np.dot(vec, prev_vec) / (norm(vec) * norm(prev_vec) + 1e-10)
            if cos_sim >= similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(kw)
            kept_vecs.append(vec)

    removed = set(kl_scores.keys()) - set(kept)
    print(f"Kept {len(kept)} / {len(kl_scores)}, removed {len(removed)}")
    return {kw: kl_scores[kw] for kw in kept}

def greedy_keywords_selection_by_node(
        kl_scores,
        em_counts,
        annotations_df,
        annotation_col,
        var_df,
        delimiter="; ",
        keyword_budget=None,
        ):

    image_paths = set(annotations_df["image_path"])

    print("Assigning keywords to nodes based on EM counts...")
    keyword_to_node = {}
    for keyword in tqdm(kl_scores):
        counts = em_counts[keyword]
        total = sum(counts.values())
        if total > 0:
            probs = {node: c / total for node, c in counts.items()}
            keyword_to_node[keyword] = max(probs, key=probs.get)

    print("Grouping keywords by assigned node...")
    node_to_keywords = {}
    for keyword, node in tqdm(keyword_to_node.items()):
        if node not in node_to_keywords:
            node_to_keywords[node] = []
        node_to_keywords[node].append(keyword)

    print("Building node to annotated image mappings...")
    node_to_annotated_images = {}
    for node in tqdm(node_to_keywords):
        mask = annotations_df[annotation_col].apply(
            lambda x, n=node: n in [l.strip() for l in x.split(delimiter)]
        )
        node_to_annotated_images[node] = set(annotations_df.loc[mask, "image_path"])

    print("Building keyword to image mappings...")
    keyword_to_images = {}
    for keyword in tqdm(kl_scores):
        mask = var_df["proposed_variable"].apply(
            lambda x, kw=keyword: kw in [k.strip() for k in x.split(", ")]
        )
        keyword_to_images[keyword] = set(var_df.loc[mask, "image_path"]) & image_paths

    selected_per_node = {}
    coverage_log_per_node = {}

    print("Performing greedy keyword selection for each node...")
    for node, candidates in tqdm(node_to_keywords.items()):
        target_images = node_to_annotated_images[node]

        if len(target_images) == 0:
            continue

        covered = set()
        selected = []
        log = []
        remaining = list(candidates)

        while remaining:
            if keyword_budget and len(selected) >= keyword_budget.get(node, 0):
                break

            best_keyword = None
            best_new_coverage = 0
            for keyword in remaining:
                new_coverage = len((keyword_to_images[keyword] & target_images) - covered)
                if new_coverage > best_new_coverage:
                    best_new_coverage = new_coverage
                    best_keyword = keyword

            if best_new_coverage == 0:
                break

            selected.append(best_keyword)
            covered |= (keyword_to_images[best_keyword] & target_images)
            remaining.remove(best_keyword)

            log.append({
                "keyword": best_keyword,
                "kl_score": kl_scores[best_keyword],
                "new_images": best_new_coverage,
                "total_covered": len(covered),
                "coverage_pct": len(covered) / len(target_images) * 100
            })

        selected_per_node[node] = selected
        coverage_log_per_node[node] = log

        print(f"{node}: {len(selected)} keywords, "
              f"{len(covered)}/{len(target_images)} images "
              f"({len(covered)/len(target_images)*100:.1f}%)")

    return selected_per_node, coverage_log_per_node

import ast
from tqdm import tqdm


def topk_keywords_selection_by_node(
        kl_scores,
        em_counts,
        annotations_df,
        annotation_col,
        var_df,
        node_frequency_dict=None,
        delimiter="; ",
        keyword_budget=None,
        fathomnet=False,
        ):

    if fathomnet:
        path_var = "path"
    else:
        path_var = "image_path"
    image_paths = set(annotations_df[path_var])

    def parse_annotation(x):
        if fathomnet:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        else:
            return [l.strip() for l in x.split(delimiter)]

    print("Assigning keywords to nodes based on EM counts...")
    keyword_to_node = {}
    for keyword in tqdm(kl_scores):
        counts = em_counts[keyword]
        total = sum(counts.values())
        if total > 0:
            probs = {node: c / total for node, c in counts.items()}
            keyword_to_node[keyword] = max(probs, key=probs.get)

    print("Grouping keywords by assigned node...")
    node_to_keywords = {}
    for keyword, node in tqdm(keyword_to_node.items()):
        if node not in node_to_keywords:
            node_to_keywords[node] = []
        node_to_keywords[node].append(keyword)

    print("Building node to annotated image mappings...")
    node_to_annotated_images = {}
    for node in tqdm(node_to_keywords):
        mask = annotations_df[annotation_col].apply(
            lambda x, n=node: n in parse_annotation(x)
        )
        node_to_annotated_images[node] = set(annotations_df.loc[mask, path_var])

    if fathomnet and "path" in var_df.columns and "image_path" not in var_df.columns:
        var_path_col = "path"
    else:
        var_path_col = "image_path"

    print("Building keyword to image mappings...")
    keyword_to_images = {}
    for keyword in tqdm(kl_scores):
        mask = var_df["proposed_variable"].apply(
            lambda x, kw=keyword: kw in [k.strip() for k in x.split(", ")]
        )
        keyword_to_images[keyword] = set(var_df.loc[mask, var_path_col]) & image_paths

    if node_frequency_dict:
        node_to_keywords = dict(sorted(
            node_to_keywords.items(),
            key=lambda x: -node_frequency_dict.get(x[0], 0)
        ))

    selected_per_node = {}
    coverage_log_per_node = {}

    print("Selecting top-K keywords per node by KL score...")
    for node, candidates in tqdm(node_to_keywords.items()):
        target_images = node_to_annotated_images[node]

        if len(target_images) == 0:
            continue

        budget = keyword_budget.get(node, 0) if keyword_budget else len(candidates)
        if budget == 0:
            continue

        selected = sorted(candidates, key=lambda kw: -kl_scores[kw])[:budget]

        covered = set()
        log = []
        for keyword in selected:
            img_set = keyword_to_images.get(keyword, set())
            new_covered = (img_set & target_images) - covered
            covered |= new_covered
            log.append({
                "keyword": keyword,
                "kl_score": kl_scores[keyword],
                "new_images": len(new_covered),
                "total_covered": len(covered),
                "coverage_pct": len(covered) / len(target_images) * 100 if target_images else 0
            })

        selected_per_node[node] = selected
        coverage_log_per_node[node] = log

        print(f"{node}: {len(selected)} keywords, "
              f"{len(covered)}/{len(target_images)} images "
              f"({len(covered)/len(target_images)*100:.1f}%)")

    return selected_per_node, coverage_log_per_node

def compute_keyword_budget(
        node_frequency_dict,
        total_budget,
        min_per_node=3):

    nonzero = {node: count for node, count in node_frequency_dict.items() if count > 0}

    sorted_nodes = sorted(nonzero.keys(), key=lambda n: nonzero[n])

    budget = {node: 0 for node in sorted_nodes}
    remaining = total_budget

    for node in sorted_nodes:
        alloc = min(min_per_node, remaining)
        budget[node] = alloc
        remaining -= alloc
        if remaining <= 0:
            break

    while remaining > 0:
        for node in sorted_nodes:
            if remaining <= 0:
                break
            budget[node] += 1
            remaining -= 1

    for node, b in sorted(budget.items(), key=lambda x: -x[1]):
        if b > 0:
            print(f"{node}: {b} keywords (count: {nonzero[node]})")

    return budget

def compute_keyword_budget_light(leaf_nodes, total_budget):
    budget = {node: 0 for node in leaf_nodes}
    remaining = total_budget

    while remaining > 0:
        for node in leaf_nodes:
            if remaining <= 0:
                break
            budget[node] += 1
            remaining -= 1

    for node, b in sorted(budget.items(), key=lambda x: -x[1]):
        if b > 0:
            print(f"{node}: {b} keywords")

    return budget

def build_save_id(domain_name, modality, emb_type, seed, num_epochs, val_ratio,
                  random_test_set=False, test_ratio=0.2, invert_modes=False,
                  adaptive_fusion=False, reinit_all=False, plus_epochs=0,
                  reinit_expert_heads=False, freeze_pooling=False,
                  image_corruption=0.0, text_corruption=0.0):
    val_ratio_str = f"{val_ratio:.2f}".replace("0.", "")
    save_id = (
        f"{domain_name}_m-{modality}_e-{emb_type}"
        f"_s-{seed}_ep-{num_epochs}"
        f"_v-{val_ratio_str}"
    )
    if invert_modes:
        save_id += "_inv"
    if adaptive_fusion:
        save_id += "_adaptive"
    if random_test_set:
        test_ratio_str = f"{test_ratio:.2f}".replace("0.", "")
        save_id += f"_rtest-{test_ratio_str}"
    if reinit_all:
        save_id += "_reinit"
    if freeze_pooling:
        save_id += "_freeze_pool"
    if plus_epochs > 0:
        save_id += f"_plus-{plus_epochs}"
    if reinit_expert_heads:
        save_id += "_reinit_expert"
    if image_corruption > 0.0:
        save_id += f"_ic-{int(round(image_corruption * 100)):02d}"
    if text_corruption > 0.0:
        save_id += f"_tc-{int(round(text_corruption * 100)):02d}"
    return save_id

def find_optimal_thresholds(probs_matrix, labels_matrix, descendant_matrix):
    import numpy as np
    from tqdm import tqdm

    probs = np.asarray(probs_matrix, dtype=np.float32)
    labels = np.asarray(labels_matrix, dtype=np.float32)
    n_samples, n_nodes = probs.shape

    node_thresholds = np.full(n_nodes, 0.5)

    for node in tqdm(range(n_nodes), desc="  Optimizing thresholds"):
        p = probs[:, node]
        y = labels[:, node] > 0

        n_pos = int(y.sum())
        if n_pos == 0 or n_pos == n_samples:
            continue

        order = np.argsort(-p, kind="stable")
        p_sorted = p[order]
        tp = np.cumsum(y[order])
        k = np.arange(1, n_samples + 1)
        f1 = 2.0 * tp / (n_pos + k)

        valid = np.empty(n_samples, dtype=bool)
        valid[:-1] = p_sorted[:-1] > p_sorted[1:]
        valid[-1] = True
        f1 = np.where(valid, f1, -1.0)

        best = int(np.argmax(f1))
        if f1[best] <= 0:
            continue

        if best < n_samples - 1:
            t = 0.5 * (p_sorted[best] + p_sorted[best + 1])
        else:
            t = max(p_sorted[-1] - 1e-6, 0.0)

        node_thresholds[node] = float(np.clip(t, 0.01, 0.99))

    hier_thresholds = node_thresholds.copy()
    for i in range(n_nodes):
        descs = np.where(descendant_matrix[i] > 0)[0]
        descs = descs[descs != i]
        if len(descs) > 0:
            hier_thresholds[i] = min(hier_thresholds[i], hier_thresholds[descs].min())

    return node_thresholds, hier_thresholds

    hier_thresholds = node_thresholds.copy()
    for i in range(n_nodes):
        descs = np.where(descendant_matrix[i] > 0)[0]
        descs = descs[descs != i]
        if len(descs) > 0:
            hier_thresholds[i] = min(hier_thresholds[i], hier_thresholds[descs].min())

    return node_thresholds, hier_thresholds

def build_dataframe_from_directory(data_dir):
    records = []

    for folder_name in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        parts = folder_name.split("_")
        if len(parts) < 8:
            print(f"Warning: skipping folder with unexpected format: {folder_name}")
            continue

        category_id = parts[0]
        kingdom = parts[1]
        phylum = parts[2]
        class_name = parts[3]
        order = parts[4]
        family = parts[5]
        genus = parts[6]
        epithet = "_".join(parts[7:])
        species = f"{genus} {epithet.replace('_', ' ')}"

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            if not os.path.isfile(img_path):
                continue
            records.append({
                "path": img_path,
                "category_id": category_id,
                "kingdom": kingdom,
                "phylum": phylum,
                "class": class_name,
                "order": order,
                "family": family,
                "genus": genus,
                "epithet": epithet,
                "species": species,
                "folder_name": folder_name,
            })

    df = pd.DataFrame(records)
    print(f"Built DataFrame from '{data_dir}': {len(df)} images across "
          f"{df['category_id'].nunique()} categories.")
    df["colloquial_annotations"] = df.apply(
        build_colloquial_annotation, axis=1, granularity="family_species"
    )
    return df

INAT_TAXONOMY_COLS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


def build_inat_hierarchy(annotations_df):
    children_dict = {}

    for _, row in annotations_df.iterrows():
        extract_hierarchy_inat(row, INAT_TAXONOMY_COLS, children_dict)

    node_id_dict = assign_node_ids(children_dict)
    graph = dict_to_graph(children_dict)
    children_id_dict = translate_text_dict_to_id_dict(children_dict, node_id_dict)
    distance_matrix = gen_distance_matrix(graph, node_id_dict)
    visual_text_graph = dict_to_visual_text_graph(children_dict)
    node_colloquial_names = get_node_colloquial_names_inat(children_dict)
    descendant_matrix = compute_descendant_matrix(graph, node_id_dict)

    return {
        "children_dict": children_dict,
        "node_id_dict": node_id_dict,
        "graph": graph,
        "children_id_dict": children_id_dict,
        "distance_matrix": distance_matrix,
        "visual_text_graph": visual_text_graph,
        "max_distance_dict": maximum_distances_for_graph(distance_matrix, node_id_dict),
        "node_colloquial_names": node_colloquial_names,
        "descendant_matrix": descendant_matrix,
    }


def extract_hierarchy_inat(row, taxonomy_cols, children_dict):
    nodes = []
    for col in taxonomy_cols:
        val = row[col]
        if pd.notna(val) and str(val).strip() != "":
            nodes.append(str(val).strip())
        else:
            break

    parent_path = None
    for i in range(len(nodes)):
        current_path = " > ".join(nodes[: i + 1])

        if current_path not in children_dict:
            children_dict[current_path] = []

        if parent_path is not None:
            if current_path not in children_dict[parent_path]:
                children_dict[parent_path].append(current_path)

        parent_path = current_path


def get_node_colloquial_names_inat(children_dict):
    unique_nodes = set()
    for parent, children in children_dict.items():
        unique_nodes.add(parent)
        for child in children:
            unique_nodes.add(child)

    unique_nodes_list = sorted(unique_nodes)

    colloquial_names = {}
    seen_count = {}
    index_map = {}

    for i, node in enumerate(unique_nodes_list):
        taxon_name = node.split(" > ")[-1]

        if taxon_name not in seen_count:
            colloquial_names[node] = taxon_name
            seen_count[taxon_name] = 1
            index_map[taxon_name] = [i]
        else:
            colloquial_names[node] = taxon_name
            seen_count[taxon_name] += 1
            index_map[taxon_name].append(i)

    for name, count in seen_count.items():
        if count > 1:
            indices = index_map[name]
            original_nodes = [unique_nodes_list[i] for i in indices]
            for orig_node in original_nodes:
                parts = orig_node.split(" > ")
                if len(parts) >= 2:
                    colloquial_names[orig_node] = f"{parts[-2]}: {parts[-1]}"

    for node in colloquial_names:
        colloquial_names[node] = colloquial_names[node].lower()

    return colloquial_names

def get_inat_annotation_path(row):
    parts = []
    for col in INAT_TAXONOMY_COLS:
        val = row[col]
        if pd.notna(val) and str(val).strip() != "":
            parts.append(str(val).strip())
        else:
            break
    return " > ".join(parts)

def corrupt_embedding(emb, corruption, mask=None):
    if corruption <= 0.0:
        return emb
    corruption = float(min(max(corruption, 0.0), 1.0))
    std = emb.detach().std().clamp_min(1e-6)
    noise = torch.randn_like(emb) * std
    out = (1.0 - corruption) * emb + corruption * noise
    if mask is not None and out.dim() == 3:
        out = out * mask.unsqueeze(-1).to(out.dtype)
    return out
