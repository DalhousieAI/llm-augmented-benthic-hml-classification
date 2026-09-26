import argparse
import ast
import json
import math
import os

import pandas as pd
from tqdm import tqdm

from utils.utils import (
    build_benthicnet_hierarchy,
    build_fathomnet_hierarchy,
    build_inat_hierarchy,
    deduplicate_by_coverage,
    deduplicate_by_distribution,
    em_decompose,
    get_inat_annotation_path,
    kl_scoring,
    topk_keywords_selection_by_node,
)


def gen_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["benthicnet", "fathomnet-2025", "inat"], default="benthicnet")
    parser.add_argument("--domain", default="bio", help="bio or sub for BenthicNet, main or roi for FathomNet")
    parser.add_argument("--partition", default="train_mini", help="iNat annotation file")
    parser.add_argument("--kl_threshold", type=float, default=1.0)
    parser.add_argument("--kw_per_node", type=int, default=1)
    parser.add_argument("--inat_max_vars", type=int, default=15)
    return parser.parse_args()


def kl_scoring_inat(em_counts_dict, num_nodes, prior):
    uniform_prior = 1.0 / num_nodes
    log_n = math.log(num_nodes)
    smooth_floor = prior * uniform_prior
    kl_scores = {}
    for keyword, label_counts in em_counts_dict.items():
        num_unseen = num_nodes - len(label_counts)
        total = sum(label_counts.values()) + prior
        entropy = 0.0
        for count in label_counts.values():
            p = (count + smooth_floor) / total
            entropy -= p * math.log(p)
        if num_unseen > 0:
            p_unseen = smooth_floor / total
            entropy -= num_unseen * p_unseen * math.log(p_unseen)
        kl_scores[keyword] = log_n - entropy
    return kl_scores


def main():
    args = gen_args()
    dataset = args.dataset
    short_form_domain = args.domain
    short_form_dataset = "bn" if dataset == "benthicnet" else "fn" if dataset == "fathomnet-2025" else dataset
    domain = "biota" if short_form_domain == "bio" else "substrates" if short_form_domain == "sub" else short_form_domain
    fathomnet = "fathomnet" in dataset

    if domain == "biota":
        annotation_col = "catami_biota"
    elif domain == "substrates":
        annotation_col = "catami_substrate"
    elif dataset == "fathomnet-2025":
        annotation_col = "label"
    elif dataset == "inat":
        annotation_col = "colloquial_annotations"
    else:
        annotation_col = "annotation"
    if domain == "main":
        annotation_col += "s"

    if dataset == "fathomnet-2025":
        var_df = pd.read_csv(f"datasets/{dataset}/outputs/{short_form_dataset}_{short_form_domain}_image_path_proposed_var.csv")
        if domain == "main":
            annotations_df = pd.read_csv(f"datasets/{dataset}/image_annotations.csv")
        else:
            annotations_df = pd.read_csv(f"datasets/{dataset}/roi_annotations.csv")
            annotations_df[annotation_col] = annotations_df[annotation_col].apply(lambda x: f"['{x}']")
    elif dataset == "inat":
        var_df = pd.read_csv(f"datasets/{dataset}/outputs/{dataset}_{args.partition}_image_path_proposed_var.csv")
        annotations_df = pd.read_csv(f"datasets/{dataset}/{args.partition}.csv")
    else:
        var_df = pd.read_csv(f"datasets/{dataset}/outputs/{short_form_dataset}_{short_form_domain}_image_path_proposed_var.csv")
        annotations_df = pd.read_csv(f"datasets/{dataset}/{dataset}_{domain}.csv")

    unique_vars = set()
    for vars_list in var_df["proposed_variable"].str.split(",").apply(lambda x: [v.strip() for v in x]):
        unique_vars.update(vars_list)
    print(f"{len(unique_vars)} unique proposed variables")

    if dataset == "benthicnet":
        hierarchy_info = build_benthicnet_hierarchy(annotations_df[annotation_col])
    elif dataset == "fathomnet-2025":
        full_df = pd.read_csv(f"datasets/{dataset}/full_annotations.csv")
        idx_to_label = json.load(open(f"datasets/{dataset}/train/index_to_taxon.json"))
        label_to_idx = {v: k for k, v in idx_to_label.items()}
        hierarchy_info = build_fathomnet_hierarchy(full_df, idx_to_label=idx_to_label, label_to_idx=label_to_idx)
    else:
        hierarchy_info = build_inat_hierarchy(annotations_df)
        annotations_df["annotation_path"] = annotations_df.apply(get_inat_annotation_path, axis=1)

    node_id_dict = hierarchy_info["node_id_dict"]
    graph = hierarchy_info["graph"]
    nodes_list = list(node_id_dict.keys())
    leaf_nodes = [node for node in nodes_list if graph.out_degree(node) == 0]

    key_col = "path" if (fathomnet or dataset == "inat") else "image_path"
    var_df["annotations"] = var_df["image_path"].map(annotations_df.set_index(key_col)[annotation_col])
    var_annotation_counts = {}
    for var in tqdm(unique_vars):
        var_rows = var_df[var_df["proposed_variable"].str.contains(var, regex=False)]
        var_annotation_counts[var] = {}
        for annotation in var_rows["annotations"].dropna().unique():
            var_annotation_counts[var][annotation] = var_rows[var_rows["annotations"] == annotation].shape[0]

    if dataset == "inat":
        colloquial_to_node = {}
        for _, row in annotations_df.drop_duplicates(subset="colloquial_annotations").iterrows():
            colloquial_to_node[row["colloquial_annotations"]] = " > ".join([
                row["kingdom"], row["phylum"], row["class"], row["order"],
                row["family"], row["genus"], row["species"],
            ])
        var_annotation_em_counts = {}
        for keyword, label_counts in var_annotation_counts.items():
            var_annotation_em_counts[keyword] = {}
            for label, count in label_counts.items():
                labels = [l.strip() for l in label.split("; ")]
                var_annotation_em_counts[keyword]["; ".join(colloquial_to_node.get(l, l) for l in labels)] = count
    else:
        os.makedirs(f"datasets/{dataset}/extracted_vars", exist_ok=True)
        var_annotation_em_counts = em_decompose(var_annotation_counts, nodes_list, fathomnet=fathomnet)
        with open(f"datasets/{dataset}/extracted_vars/{short_form_dataset}_{short_form_domain}_var_annotation_em_counts.json", "w") as f:
            json.dump(var_annotation_em_counts, f, indent=4)

    if dataset == "inat":
        annotation_col = "annotation_path"
        annotations_df_train = annotations_df
        node_frequency_dict = annotations_df_train[annotation_col].value_counts().reindex(nodes_list, fill_value=0).to_dict()
    else:
        annotations_df_train = annotations_df[annotations_df["partition"] == "train"]
        if fathomnet:
            node_frequency_dict = {
                node: annotations_df_train[annotation_col].apply(lambda x: node in ast.literal_eval(x)).sum()
                for node in nodes_list
            }
        else:
            node_frequency_dict = {
                node: annotations_df_train[annotation_col].apply(lambda x: node in [n.strip() for n in x.split("; ")]).sum()
                for node in nodes_list
            }
    total_annotations = sum(node_frequency_dict.values())
    node_prior_dict = {node: count / total_annotations for node, count in node_frequency_dict.items()}

    num_nodes = len(node_prior_dict)
    if dataset == "inat":
        kl_scores = kl_scoring_inat(var_annotation_em_counts, num_nodes, prior=num_nodes)
    else:
        kl_scores = kl_scoring(var_annotation_em_counts, node_prior_dict, prior=num_nodes)
    kl_scores_thresholded = {kw: s for kw, s in kl_scores.items() if s > args.kl_threshold}
    deduped_dist_kl_scores = deduplicate_by_distribution(kl_scores_thresholded, var_annotation_em_counts)
    deduped_kl_scores = deduplicate_by_coverage(var_df, deduped_dist_kl_scores)

    leaf_nodes_sorted = sorted(leaf_nodes, key=lambda x: node_frequency_dict[x], reverse=True)
    keyword_budget = {node: args.kw_per_node for node in leaf_nodes_sorted}

    if dataset == "inat":
        annotations_df["image_path"] = annotations_df["path"]
    selected, coverage_log = topk_keywords_selection_by_node(
        deduped_kl_scores,
        var_annotation_em_counts,
        annotations_df_train,
        annotation_col,
        var_df,
        node_frequency_dict=node_frequency_dict,
        keyword_budget=keyword_budget,
        fathomnet=fathomnet,
    )

    if dataset == "inat":
        selected = {k: v for k, v in coverage_log.items() if v[0]["coverage_pct"] > 50.0}
        selected = dict(sorted(selected.items(), key=lambda x: x[1][0]["kl_score"], reverse=True)[:args.inat_max_vars])
        selected = {k: [v[0]["keyword"]] for k, v in selected.items()}

    print(f"{len(selected)} nodes selected")
    for node, keywords in selected.items():
        print(f"{node}: {keywords}")
    os.makedirs(f"datasets/{dataset}/extracted_vars", exist_ok=True)
    with open(f"datasets/{dataset}/extracted_vars/{short_form_dataset}_{short_form_domain}_selected_keywords.json", "w") as f:
        json.dump(selected, f, indent=4)


if __name__ == "__main__":
    main()
