import json
import re


def load_caption_vars(
    path="./datasets/inat/extracted_vars/inat_selected_keywords.json",
):
    with open(path, "r") as f:
        attributes = json.load(f)
    return sorted(set(
        prop for props in attributes.values() for prop in props
    ))


def build_label_pattern(vars):
    ordered = sorted(set(vars), key=len, reverse=True)
    alt = "|".join(re.escape(v) for v in ordered)
    return re.compile(r"(?<![\w])(" + alt + r")\s*:", re.IGNORECASE)


def extract_property_segments(caption, vars, pattern=None, lower_to_var=None):
    if not isinstance(caption, str):
        return {}
    pattern = pattern or build_label_pattern(vars)
    lower_to_var = lower_to_var or {v.lower(): v for v in vars}

    matches = list(pattern.finditer(caption))
    segments = {}
    for i, m in enumerate(matches):
        var = lower_to_var.get(m.group(1).lower())
        if var is None or var in segments:
            continue
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(caption)
        seg = " ".join(caption[start:end].replace("\n", " ").split())
        segments[var] = seg
    return segments


def find_missing_vars(caption, vars, pattern=None, lower_to_var=None):
    present = extract_property_segments(caption, vars, pattern, lower_to_var)
    return [v for v in sorted(vars) if v not in present]


def merge_caption(original_caption, retry_caption, vars,
                  pattern=None, lower_to_var=None):
    pattern = pattern or build_label_pattern(vars)
    lower_to_var = lower_to_var or {v.lower(): v for v in vars}
    merged = extract_property_segments(original_caption, vars, pattern, lower_to_var)
    retry = extract_property_segments(retry_caption, vars, pattern, lower_to_var)
    for v, seg in retry.items():
        if v not in merged:
            merged[v] = seg
    return "\n".join(merged[v] for v in sorted(vars) if v in merged)
