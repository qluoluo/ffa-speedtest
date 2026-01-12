#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_GROUP_MAP = {
    "threshold": [r"threshold"],
    "stage1": [r"stage1"],
    "stage2": [r"stage2"],
    "scan": [r"scan"],
    "refine": [r"refine"],
}

DEFAULT_GROUP_ORDER = ["threshold", "stage1", "stage2", "scan", "refine", "other"]

UNIT_TO_MS = {
    "s": 1000.0,
    "sec": 1000.0,
    "second": 1000.0,
    "seconds": 1000.0,
    "ms": 1.0,
    "msec": 1.0,
    "millisecond": 1.0,
    "milliseconds": 1.0,
    "us": 1e-3,
    "usec": 1e-3,
    "usecond": 1e-3,
    "useconds": 1e-3,
    "microsecond": 1e-3,
    "microseconds": 1e-3,
    "ns": 1e-6,
    "nsec": 1e-6,
    "nsecond": 1e-6,
    "nseconds": 1e-6,
    "nanosecond": 1e-6,
    "nanoseconds": 1e-6,
}

HEADER_UNIT_RE = re.compile(r"^(?P<name>.+?)\s*\((?P<unit>[^)]+)\)\s*$")


def normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def parse_header_with_unit(fieldname: str):
    match = HEADER_UNIT_RE.match(fieldname)
    if not match:
        return fieldname, None
    return match.group("name").strip(), match.group("unit").strip()


def detect_columns(fieldnames):
    kernel_col = None
    time_col = None
    time_col_unit = None
    metric_name_col = None
    metric_value_col = None
    metric_unit_col = None
    section_col = None

    for raw_name in fieldnames:
        name, _ = parse_header_with_unit(raw_name)
        key = normalize_key(name)
        if kernel_col is None and (key == "kernelname" or ("kernel" in key and "name" in key)):
            kernel_col = raw_name
        if metric_name_col is None and key == "metricname":
            metric_name_col = raw_name
        if metric_value_col is None and key == "metricvalue":
            metric_value_col = raw_name
        if metric_unit_col is None and key == "metricunit":
            metric_unit_col = raw_name
        if section_col is None and key == "sectionname":
            section_col = raw_name

    time_candidates = []
    for raw_name in fieldnames:
        name, unit = parse_header_with_unit(raw_name)
        key = normalize_key(name)
        if key in {"duration", "kerneltime", "totaltime", "time"}:
            time_candidates.append((raw_name, unit, key))

    if time_candidates:
        time_candidates.sort(key=lambda x: ["duration", "kerneltime", "totaltime", "time"].index(x[2]))
        time_col, time_col_unit, _ = time_candidates[0]

    return {
        "kernel_col": kernel_col,
        "time_col": time_col,
        "time_col_unit": time_col_unit,
        "metric_name_col": metric_name_col,
        "metric_value_col": metric_value_col,
        "metric_unit_col": metric_unit_col,
        "section_col": section_col,
    }


def unit_to_ms(unit: str | None, default_unit: str):
    unit = (unit or default_unit or "").strip().lower()
    unit = unit.replace(" ", "")
    if unit.endswith("seconds"):
        unit = unit.replace("seconds", "second")
    if unit in UNIT_TO_MS:
        return UNIT_TO_MS[unit]
    return None


def parse_time_to_ms(value: str | None, unit: str | None, default_unit: str):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.match(r"^([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*([a-zA-Z]+)?$", text)
    if not match:
        return None
    number = float(match.group(1))
    unit_in_value = match.group(2)
    factor = unit_to_ms(unit_in_value or unit, default_unit)
    if factor is None:
        return None
    return number * factor


def is_duration_metric(metric_name: str, section_name: str | None) -> bool:
    name = (metric_name or "").strip().lower()
    section = (section_name or "").strip().lower()
    if "launch" in section:
        if "duration" in name or name in {"time", "kernel time", "total time"}:
            return True
    if "duration" in name:
        return True
    return False


def assign_group(kernel_name: str, group_map: dict[str, list[str]]):
    for group, patterns in group_map.items():
        for pattern in patterns:
            if re.search(pattern, kernel_name, flags=re.IGNORECASE):
                return group
    return "other"


def parse_ncu_csv(path: Path, default_unit: str, include_re, exclude_re):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header")
        cols = detect_columns(reader.fieldnames)
        kernel_col = cols["kernel_col"]
        if kernel_col is None:
            raise ValueError(f"{path} missing Kernel Name column")

        durations = defaultdict(float)
        for row in reader:
            kernel_name = (row.get(kernel_col) or "").strip()
            if not kernel_name:
                continue
            if include_re and not include_re.search(kernel_name):
                continue
            if exclude_re and exclude_re.search(kernel_name):
                continue

            if cols["time_col"]:
                time_value = row.get(cols["time_col"])
                unit = cols["time_col_unit"]
                ms = parse_time_to_ms(time_value, unit, default_unit)
                if ms is None:
                    continue
                durations[kernel_name] += ms
            elif cols["metric_name_col"] and cols["metric_value_col"]:
                metric_name = row.get(cols["metric_name_col"])
                if not is_duration_metric(metric_name, row.get(cols["section_col"])):
                    continue
                metric_unit = row.get(cols["metric_unit_col"]) if cols["metric_unit_col"] else None
                ms = parse_time_to_ms(row.get(cols["metric_value_col"]), metric_unit, default_unit)
                if ms is None:
                    continue
                durations[kernel_name] += ms

    return durations


def load_group_map(path: str | None):
    if not path:
        return DEFAULT_GROUP_MAP
    with open(path, "r") as handle:
        data = json.load(handle)
    return {key: value for key, value in data.items()}


def load_labels(inputs, labels, label_regex):
    if labels:
        if len(labels) != len(inputs):
            raise ValueError("--labels must match number of inputs")
        return labels
    if label_regex:
        pattern = re.compile(label_regex)
        out = []
        for path in inputs:
            match = pattern.search(path)
            out.append(match.group(0) if match else Path(path).stem)
        return out
    return [Path(path).stem for path in inputs]


def sort_items(items, mode):
    if mode == "none":
        return items
    reverse = mode == "desc"
    return sorted(items, key=lambda item: item[1]["total_ms"], reverse=reverse)


def build_plot(data, group_order, out_path, title):
    labels = [item["label"] for item in data]
    totals = [item["total_ms"] for item in data]
    groups = group_order + [
        g for g in sorted({g for item in data for g in item["groups"]}) if g not in group_order
    ]

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(labels)), 6))
    bottom = [0.0 for _ in labels]
    colors = {
        "threshold": "tab:blue",
        "stage1": "tab:orange",
        "stage2": "tab:green",
        "scan": "tab:red",
        "refine": "tab:purple",
        "other": "tab:gray",
    }

    for group in groups:
        values = [item["groups"].get(group, 0.0) for item in data]
        if all(v == 0 for v in values):
            continue
        ax.bar(labels, values, bottom=bottom, label=group, color=colors.get(group))
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_ylabel("Kernel time (ms)")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, max(totals) * 1.1 if totals else 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Parse Nsight Compute CSV and plot kernel breakdown.")
    parser.add_argument("--inputs", nargs="+", required=True, help="NCU CSV files (exported Launch Statistics).")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    parser.add_argument("--out-json", default=None, help="Optional JSON summary output.")
    parser.add_argument("--labels", nargs="+", help="Optional labels for each input.")
    parser.add_argument("--label-regex", default=None, help="Regex to extract label from input path.")
    parser.add_argument("--group-map", default=None, help="JSON mapping of group -> list of regex patterns.")
    parser.add_argument("--unit", default="us", help="Default unit if CSV lacks units (ns/us/ms/s).")
    parser.add_argument("--include", default=None, help="Regex to include kernel names.")
    parser.add_argument("--exclude", default=None, help="Regex to exclude kernel names.")
    parser.add_argument("--sort", choices=["asc", "desc", "none"], default="asc", help="Sort by total time.")
    parser.add_argument("--title", default="NCU CUDAGraph kernel breakdown", help="Plot title.")
    args = parser.parse_args()

    group_map = load_group_map(args.group_map)
    labels = load_labels(args.inputs, args.labels, args.label_regex)
    include_re = re.compile(args.include) if args.include else None
    exclude_re = re.compile(args.exclude) if args.exclude else None

    summaries = []
    for input_path, label in zip(args.inputs, labels):
        durations = parse_ncu_csv(Path(input_path), args.unit, include_re, exclude_re)
        groups = defaultdict(float)
        for kernel_name, ms in durations.items():
            groups[assign_group(kernel_name, group_map)] += ms
        total_ms = sum(groups.values())
        pct = {group: (ms / total_ms * 100.0 if total_ms > 0 else 0.0) for group, ms in groups.items()}
        summaries.append({
            "label": label,
            "total_ms": total_ms,
            "groups": dict(groups),
            "pct": pct,
        })

    sorted_items = sort_items([(item["label"], item) for item in summaries], args.sort)
    sorted_summaries = [item for _, item in sorted_items]

    out_path = Path(args.out)
    build_plot(sorted_summaries, DEFAULT_GROUP_ORDER, out_path, args.title)

    if args.out_json:
        out_json = {
            "units": "ms",
            "inputs": sorted_summaries,
        }
        Path(args.out_json).write_text(json.dumps(out_json, indent=2))


if __name__ == "__main__":
    main()
