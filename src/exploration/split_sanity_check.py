"""

Sanity-check train/validation/test dataset splits for palynomorph annotations.

This script:
1. Loads a split JSON file containing train/val/test sample names.
2. Matches split samples to:
   - annotation count columns in a class-summary CSV
   - NDPA annotation files for circle size analysis
   - stain-status labels in a stain CSV
3. Summarizes class balance, total annotation counts, annotation size distributions,
   and stained vs. non-stained slide counts.
4. Saves summary CSV files and plots.

Expected inputs in the working directory:
- train_val_test.json
- annotation_by_category.csv
- Samples_with_annotations_stained_non_stained.csv
- *.ndpa

"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

"""Directories and configurations"""
BASE_DIR = Path(__file__).resolve().parent
SPLIT_JSON = BASE_DIR / "train_val_test.json"
CLASS_CSV = BASE_DIR / "annotation_by_category.csv"
STAIN_CSV = BASE_DIR / "Samples_with_annotations_stained_non_stained.csv"
CLASSES = ["pol", "spo", "paly", "din", "alg", "fun"]
NM_PER_PX = ((0.22885913720105275 + 0.22886437497139195) / 2) * 1000
MIN_DIAMETER_UM, MAX_DIAMETER_UM = 1, 250
SPLITS = ["train", "val", "test"]


def normalize_name(text: str) -> str:
    """Normalize sample/file names for fuzzy matching."""
    text = str(text).lower().replace(".ndpi.ndpa", "").replace(".ndpa", "").replace(".ndpi", "")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def score_match(a: str, b: str) -> tuple[int, int, int]:
    """Return a sortable fuzzy-match score; higher is better."""
    a, b = normalize_name(a), normalize_name(b)
    if a == b:
        return (4, 9999, 0)
    if a.startswith(b) or b.startswith(a):
        return (3, min(len(a), len(b)), -abs(len(a) - len(b)))
    if a in b or b in a:
        return (2, min(len(a), len(b)), -abs(len(a) - len(b)))
    ta, tb = set(a.split("_")), set(b.split("_"))
    return (1, len(ta & tb), -abs(len(ta) - len(tb)))


def best_match(name: str, candidates: list[str], used: set[str] | None = None, min_overlap: int = 2) -> str | None:
    """Return the best available candidate for a sample name."""
    pool = [c for c in candidates if used is None or c not in used] or candidates
    if not pool:
        return None
    score, match = max(((score_match(name, c), c) for c in pool), key=lambda x: x[0])
    return match if score[0] >= 2 or (score[0] == 1 and score[1] >= min_overlap) else None


def parse_circle_annotations(ndpa_path: Path) -> list[dict]:
    """Extract usable circle diameters from an NDPA file."""
    rows = []
    root = ET.parse(ndpa_path).getroot()
    for ann in root.iter():
        if ann.tag.split("}")[-1] != "annotation":
            continue
        if ann.attrib.get("displayname") != "AnnotateCircle" or ann.attrib.get("type") != "circle":
            continue
        radius = next((c.text for c in ann if c.tag.split("}")[-1] == "radius"), None)
        try:
            diameter_nm = 2 * float(radius)
        except (TypeError, ValueError):
            continue
        diameter_um = diameter_nm / 1000
        if not (MIN_DIAMETER_UM <= diameter_um <= MAX_DIAMETER_UM):
            continue
        rows.append({
            "diameter_um": diameter_um,
            "diameter_px": diameter_nm / NM_PER_PX,
            "source_file": ndpa_path.name,
        })
    return rows


def style_ax(ax) -> None:
    """Apply consistent axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.ticklabel_format(style="plain", useOffset=False, axis="y")
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def label_bars(ax, values, total: float | None = None, show_count: bool = False) -> None:
    """Add simple text labels above bars."""
    for i, v in enumerate(values):
        text = f"{int(v)}" if show_count or total in (None, 0) else f"{v / total * 100:.1f}%"
        if total not in (None, 0) and show_count:
            text = f"{int(v)}\n({v / total * 100:.1f}%)"
        ax.text(i, v, text, ha="center", va="bottom", fontsize=9)


def save_bar(series: pd.Series, title: str, ylabel: str, out: Path, total: float | None = None, show_count: bool = False) -> None:
    """Save a single bar plot from a Series."""
    plt.figure(figsize=(8, 5))
    ax = series.plot(kind="bar")
    ax.set_title(title)
    ax.set_xlabel(series.index.name or "")
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=0)
    style_ax(ax)
    label_bars(ax, series.values, total=total, show_count=show_count)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def main() -> None:
    """Run the split sanity check and save tables/plots."""
    for path in [SPLIT_JSON, CLASS_CSV, STAIN_CSV]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    split = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    df_classes_raw = pd.read_csv(CLASS_CSV).set_index(pd.read_csv(CLASS_CSV).columns[0])
    df_classes = df_classes_raw.loc[CLASSES].apply(pd.to_numeric, errors="coerce")
    class_cols = df_classes.columns.tolist()

    df_stain = pd.read_csv(STAIN_CSV)
    df_stain = df_stain.rename(columns={df_stain.columns[0]: "file_name", df_stain.columns[1]: "stain_status"})
    df_stain["stain_status"] = df_stain["stain_status"].astype(str).str.strip().str.lower()

    ndpa_files = sorted(BASE_DIR.glob("*.ndpa"))
    ndpa_stems = [p.name.replace(".ndpi.ndpa", "").replace(".ndpa", "") for p in ndpa_files]
    ndpa_map = dict(zip(ndpa_stems, ndpa_files))

    csv_rows, ndpa_rows, stain_rows = [], [], []
    used_csv, used_ndpa, used_stain = set(), set(), set()
    stain_candidates = df_stain["file_name"].tolist()

    for split_name in SPLITS:
        for sample in split.get(split_name, []):
            csv_match = best_match(sample, class_cols, used_csv)
            if csv_match:
                used_csv.add(csv_match)
            csv_rows.append({"split": split_name, "sample_name": sample, "matched_csv_column": csv_match})

            ndpa_match = best_match(sample, ndpa_stems, used_ndpa)
            if ndpa_match:
                used_ndpa.add(ndpa_match)
            ndpa_rows.append({
                "split": split_name,
                "sample_name": sample,
                "matched_ndpa_stem": ndpa_match,
                "matched_ndpa_file": ndpa_map.get(ndpa_match),
            })

            stain_match = best_match(sample, stain_candidates)
            stain_status = None
            if stain_match is not None:
                idxs = df_stain.index[df_stain["file_name"] == stain_match].tolist()
                idx = next((i for i in idxs if i not in used_stain), idxs[0] if idxs else None)
                if idx is not None:
                    used_stain.add(idx)
                    stain_status = df_stain.loc[idx, "stain_status"]
            stain_rows.append({
                "split": split_name,
                "sample_name": sample,
                "matched_stain_name": stain_match,
                "stain_status": stain_status,
            })

    df_csv_matches = pd.DataFrame(csv_rows)
    df_ndpa_matches = pd.DataFrame(ndpa_rows)
    df_stain_matches = pd.DataFrame(stain_rows)

    split_class_counts, split_total_counts = {}, {}
    for split_name in SPLITS:
        matched = df_csv_matches.loc[
            (df_csv_matches["split"] == split_name) & df_csv_matches["matched_csv_column"].notna(),
            "matched_csv_column",
        ].tolist()
        counts = df_classes[matched].sum(axis=1).reindex(CLASSES).fillna(0) if matched else pd.Series(0, index=CLASSES)
        split_class_counts[split_name] = counts
        split_total_counts[split_name] = float(counts.sum())

    size_rows = []
    for _, row in df_ndpa_matches.iterrows():
        if pd.notna(row["matched_ndpa_file"]):
            for x in parse_circle_annotations(Path(row["matched_ndpa_file"])):
                x.update({"split": row["split"], "sample_name": row["sample_name"]})
                size_rows.append(x)
    df_sizes = pd.DataFrame(size_rows)

    split_stain_counts = {}
    for split_name in SPLITS:
        counts = df_stain_matches.loc[
            (df_stain_matches["split"] == split_name) & df_stain_matches["stain_status"].notna(),
            "stain_status",
        ].value_counts()
        stained = int(counts.get("stained", 0))
        not_stained = int(counts.get("no stained", 0)) + int(counts.get("not stained", 0))
        split_stain_counts[split_name] = {"stained": stained, "not_stained": not_stained}

    summary = []
    for split_name in SPLITS:
        counts = split_class_counts[split_name]
        total = split_total_counts[split_name]
        size_sub = df_sizes[df_sizes["split"] == split_name] if not df_sizes.empty else pd.DataFrame()
        row = {
            "split": split_name,
            "total_annotations_from_csv": total,
            "num_size_annotations_from_ndpa": len(size_sub),
            "median_diameter_um": size_sub["diameter_um"].median() if not size_sub.empty else None,
            "mean_diameter_um": size_sub["diameter_um"].mean() if not size_sub.empty else None,
            "median_diameter_px": size_sub["diameter_px"].median() if not size_sub.empty else None,
            "mean_diameter_px": size_sub["diameter_px"].mean() if not size_sub.empty else None,
            "stained_slides": split_stain_counts[split_name]["stained"],
            "not_stained_slides": split_stain_counts[split_name]["not_stained"],
        }
        for cls in CLASSES:
            row[cls] = counts[cls]
            row[f"{cls}_pct"] = counts[cls] / total * 100 if total else 0
        summary.append(row)
    df_summary = pd.DataFrame(summary)

    df_summary.to_csv(BASE_DIR / "split_sanity_summary.csv", index=False)
    df_csv_matches.to_csv(BASE_DIR / "split_csv_matches.csv", index=False)
    df_ndpa_matches.to_csv(BASE_DIR / "split_ndpa_matches.csv", index=False)
    df_stain_matches.to_csv(BASE_DIR / "split_stain_matches.csv", index=False)
    if not df_sizes.empty:
        df_sizes.to_csv(BASE_DIR / "split_size_annotations.csv", index=False)

    save_bar(pd.Series(split_total_counts), "Total Annotation Number by Split", "Total Annotations", BASE_DIR / "annotation_number_by_split.png", show_count=True)
    for split_name in SPLITS:
        save_bar(
            split_class_counts[split_name],
            f"Annotation Class Distribution: {split_name.capitalize()}",
            "Count",
            BASE_DIR / f"class_distribution_{split_name}.png",
            total=split_total_counts[split_name],
            show_count=True,
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, split_name in zip(axes, SPLITS):
        counts = split_class_counts[split_name]
        counts.plot(kind="bar", ax=ax)
        ax.set_title(split_name.capitalize())
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_xticklabels(CLASSES, rotation=0)
        style_ax(ax)
        label_bars(ax, counts.values, total=split_total_counts[split_name])
    fig.suptitle("Annotation Class Distribution Across Train / Validation / Test", fontsize=14)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "class_distribution_train_val_test_panel.png", dpi=300)
    plt.close(fig)

    if not df_sizes.empty:
        for metric, xlabel, out in [
            ("diameter_px", "Diameter (pixels)", "size_distribution_pixels_by_split.png"),
            ("diameter_um", "Diameter (µm)", "size_distribution_microns_by_split.png"),
        ]:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
            for ax, split_name in zip(axes, SPLITS):
                sub = df_sizes[df_sizes["split"] == split_name]
                ax.hist(sub[metric], bins=30, edgecolor="black")
                ax.set_title(split_name.capitalize())
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Count")
                style_ax(ax)
                if len(sub):
                    med = sub[metric].median()
                    ax.axvline(med, linestyle="--", linewidth=1.5)
                    ax.text(med, ax.get_ylim()[1] * 0.9, f"median={med:.1f}", rotation=90, va="top", ha="right", fontsize=9)
            fig.suptitle(f"Size Distribution Across Train / Validation / Test: {xlabel}", fontsize=14)
            plt.tight_layout()
            plt.savefig(BASE_DIR / out, dpi=300)
            plt.close(fig)

    stain_df = pd.DataFrame(split_stain_counts).T[["stained", "not_stained"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    stain_df.plot(kind="bar", stacked=True, ax=axes[0])
    axes[0].set_title("Stained vs Not Stained Slides by Split")
    axes[0].set_xlabel("Split")
    axes[0].set_ylabel("Number of Slides")
    axes[0].set_xticklabels(stain_df.index.tolist(), rotation=0)
    style_ax(axes[0])
    stain_df.div(stain_df.sum(axis=1), axis=0).fillna(0).mul(100).plot(kind="bar", stacked=True, ax=axes[1])
    axes[1].set_title("Stained vs Not Stained Proportion by Split")
    axes[1].set_xlabel("Split")
    axes[1].set_ylabel("Percentage")
    axes[1].set_xticklabels(stain_df.index.tolist(), rotation=0)
    style_ax(axes[1])
    plt.tight_layout()
    plt.savefig(BASE_DIR / "stain_status_by_split.png", dpi=300)
    plt.close(fig)

    print("Saved summary tables and plots to:", BASE_DIR)


if __name__ == "__main__":
    main()
