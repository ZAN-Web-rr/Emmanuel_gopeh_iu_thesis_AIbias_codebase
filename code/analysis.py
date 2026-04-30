from __future__ import annotations

import csv
import html
import math
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WORKBOOK_PATH = DATA_DIR / "raw_platform_responses.xlsx"
OUTPUTS_DIR = ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
STATS_DIR = OUTPUTS_DIR / "stats"
TABLES_DIR = OUTPUTS_DIR / "tables"

MAIN_SHEETS = {"GPT_Responses", "gemini_responses", "Claude_responses"}
VALID_PLATFORMS = {"GPT", "GEM", "CLA"}
PLATFORM_TO_FAMILY = {"GPT": "OPENAI", "GEM": "GOOGLE", "CLA": "ANTHROPIC"}
PLATFORM_ORDER = ["GPT", "GEM", "CLA"]
CATEGORY_NORMALIZATION = {
    "IDE integration": "IDE Integration",
    "Enterprise and Cloud Deployment": "Enterprise & Cloud Deployment",
}
PROMPT_CATEGORY_OVERRIDES = {
    "P17": "Prompt Playground / Studio",
}


ENTITY_GROUPS = {
    "OPENAI": [
        "azure openai",
        "responses api",
        "chat completions",
        "openai api",
        "advanced data analysis",
        "code interpreter",
        "codex cli",
        "chatgpt",
        "openai",
        "codex",
        "gpt-5.4",
        "gpt-5.3",
        "gpt-5.2",
        "gpt-5",
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4",
        "o1",
        "o3",
    ],
    "GOOGLE": [
        "google ai studio",
        "gemini code assist",
        "google cloud",
        "vertex ai",
        "gemini flash",
        "gemini pro",
        "gemini 3.1",
        "gemini 3",
        "gemini 2.5",
        "gemini 1.5",
        "google gemini",
        "google",
        "gemini",
        "bard",
        "palm",
        "firebase",
    ],
    "ANTHROPIC": [
        "anthropic api",
        "claude code",
        "claude sonnet",
        "claude opus",
        "claude haiku",
        "claude 4.6",
        "claude 4.5",
        "claude 4",
        "claude 3.5",
        "anthropic",
        "claude",
    ],
    "OTHER": [
        "amazon bedrock",
        "aws bedrock",
        "azure ai",
        "microsoft azure",
        "github copilot",
        "jetbrains ai assistant",
        "gemini cli",
        "deepseek",
        "perplexity",
        "openrouter",
        "litel lm",
        "litellm",
        "vercel ai sdk",
        "mistral",
        "cohere",
        "siliconflow",
        "cursor",
        "groq",
        "bedrock",
        "aws",
        "azure",
        "meta",
        "llama",
        "julius ai",
    ],
}


@dataclass(frozen=True)
class EntityPattern:
    family: str
    term: str
    regex: re.Pattern[str]


def ensure_output_dirs() -> None:
    for path in (DATA_DIR, OUTPUTS_DIR, CHARTS_DIR, STATS_DIR, TABLES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def prompt_sort_key(prompt_id: str) -> int:
    match = re.search(r"(\d+)", prompt_id or "")
    return int(match.group(1)) if match else 999


def platform_sort_key(platform: str) -> int:
    try:
        return PLATFORM_ORDER.index(platform)
    except ValueError:
        return len(PLATFORM_ORDER)


def normalize_prompt_id(value: str) -> str:
    value = normalize_space(value).upper()
    match = re.search(r"P\s*0*(\d+)", value)
    if not match:
        return value
    return f"P{int(match.group(1)):02d}"


def parse_excel_date(value: str) -> str:
    value = normalize_space(value)
    if not value:
        return ""
    try:
        serial = float(value)
    except ValueError:
        return value
    origin = datetime(1899, 12, 30)
    return (origin + timedelta(days=int(serial))).date().isoformat()


def normalize_category(value: str) -> str:
    value = normalize_space(value)
    return CATEGORY_NORMALIZATION.get(value, value)


def safe_int(value: str, fallback: int = 0) -> int:
    value = normalize_space(str(value))
    if not value:
        return fallback
    try:
        return int(float(value))
    except ValueError:
        return fallback


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for char in letters:
        value = (value * 26) + (ord(char.upper()) - ord("A") + 1)
    return value - 1


def parse_shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    try:
        xml_text = workbook.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(xml_text)
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    values: list[str] = []
    for item in root.findall("x:si", namespace):
        fragments = [node.text or "" for node in item.findall(".//x:t", namespace)]
        values.append("".join(fragments))
    return values


def parse_workbook(workbook_path: Path) -> dict[str, list[dict[str, str]]]:
    namespace_main = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    namespace_rel = {"r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    namespace_pkg = {"p": "http://schemas.openxmlformats.org/package/2006/relationships"}

    with zipfile.ZipFile(workbook_path) as workbook:
        shared_strings = parse_shared_strings(workbook)
        workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
        rels_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels_root.findall("p:Relationship", namespace_pkg)
        }

        parsed: dict[str, list[dict[str, str]]] = {}
        for sheet in workbook_root.findall("x:sheets/x:sheet", namespace_main):
            sheet_name = sheet.attrib["name"]
            rel_id = sheet.attrib[f"{{{namespace_rel['r']}}}id"]
            target = rel_map[rel_id].replace("\\", "/")
            if not target.startswith("xl/"):
                target = f"xl/{target}"
            sheet_root = ET.fromstring(workbook.read(target))
            rows = sheet_root.findall("x:sheetData/x:row", namespace_main)
            if not rows:
                parsed[sheet_name] = []
                continue

            header_map: dict[int, str] = {}
            records: list[dict[str, str]] = []
            for row_number, row in enumerate(rows, start=1):
                values_by_index: dict[int, str] = {}
                for cell in row.findall("x:c", namespace_main):
                    idx = column_index(cell.attrib.get("r", "A1"))
                    cell_type = cell.attrib.get("t", "")
                    value_node = cell.find("x:v", namespace_main)
                    inline_node = cell.find("x:is", namespace_main)

                    if cell_type == "s" and value_node is not None:
                        value = shared_strings[int(value_node.text or "0")]
                    elif cell_type == "inlineStr" and inline_node is not None:
                        fragments = [node.text or "" for node in inline_node.findall(".//x:t", namespace_main)]
                        value = "".join(fragments)
                    elif value_node is not None:
                        value = value_node.text or ""
                    else:
                        value = ""
                    values_by_index[idx] = value

                if row_number == 1:
                    for idx, value in values_by_index.items():
                        header_map[idx] = normalize_space(value)
                    continue

                record = {header_map.get(idx, f"Column_{idx + 1}"): value for idx, value in values_by_index.items()}
                records.append(record)

            parsed[sheet_name] = records

        return parsed


def clean_records(parsed_sheets: dict[str, list[dict[str, str]]]) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    main_rows: list[dict[str, object]] = []
    verification_rows: list[dict[str, object]] = []
    stats = {
        "raw_main_rows": 0,
        "raw_verification_rows": 0,
        "kept_main_rows": 0,
        "kept_verification_rows": 0,
        "dropped_invalid_platform": 0,
        "dropped_blank_response": 0,
    }

    for sheet_name, rows in parsed_sheets.items():
        is_main = sheet_name in MAIN_SHEETS

        for row_index, row in enumerate(rows, start=2):
            platform = normalize_space(row.get("Platform", "")).upper()
            prompt_id = normalize_prompt_id(row.get("Prompt Number", ""))
            category = normalize_category(row.get("Prompt Category", ""))
            if prompt_id in PROMPT_CATEGORY_OVERRIDES:
                category = PROMPT_CATEGORY_OVERRIDES[prompt_id]
            collection_date = parse_excel_date(row.get("Collection Date", ""))
            response_text = (row.get("Full Response Text", "") or "").strip()
            prompt_text = (row.get("Prompt", "") or "").strip()
            supplied_word_count = safe_int(row.get("Response Word Count", ""), fallback=0)

            if not platform and not prompt_id and not response_text:
                continue
            if is_main:
                stats["raw_main_rows"] += 1
            else:
                stats["raw_verification_rows"] += 1
            if platform not in VALID_PLATFORMS:
                stats["dropped_invalid_platform"] += 1
                continue
            if not response_text:
                stats["dropped_blank_response"] += 1
                continue

            cleaned = {
                "sheet_name": sheet_name,
                "source_set": "main" if is_main else "verification",
                "row_number": row_index,
                "platform": platform,
                "platform_family": PLATFORM_TO_FAMILY[platform],
                "prompt_id": prompt_id,
                "category": category,
                "collection_date": collection_date,
                "response_text": response_text,
                "prompt_text": prompt_text,
                "response_word_count": supplied_word_count or word_count(response_text),
                "computed_word_count": word_count(response_text),
                "response_id": f"{platform}-{prompt_id}-{('M' if is_main else 'V')}-{row_index}",
            }

            if is_main:
                main_rows.append(cleaned)
                stats["kept_main_rows"] += 1
            else:
                verification_rows.append(cleaned)
                stats["kept_verification_rows"] += 1

    main_rows.sort(key=lambda row: (PLATFORM_ORDER.index(row["platform"]), prompt_sort_key(str(row["prompt_id"]))))
    verification_rows.sort(key=lambda row: (PLATFORM_ORDER.index(row["platform"]), prompt_sort_key(str(row["prompt_id"]))))
    return main_rows, verification_rows, stats


def compile_term(term: str) -> re.Pattern[str]:
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", flags=re.IGNORECASE)


def build_entity_patterns() -> list[EntityPattern]:
    patterns: list[EntityPattern] = []
    for family, terms in ENTITY_GROUPS.items():
        seen_terms: set[str] = set()
        for term in sorted(terms, key=len, reverse=True):
            normalized = term.lower()
            if normalized in seen_terms:
                continue
            seen_terms.add(normalized)
            patterns.append(EntityPattern(family=family, term=term, regex=compile_term(term)))
    patterns.sort(key=lambda item: len(item.term), reverse=True)
    return patterns


ENTITY_PATTERNS = build_entity_patterns()


def extract_mentions(text: str) -> list[dict[str, object]]:
    text = text or ""
    occupied: list[tuple[int, int]] = []
    mentions: list[dict[str, object]] = []

    for pattern in ENTITY_PATTERNS:
        for match in pattern.regex.finditer(text):
            start, end = match.span()
            if any(start < used_end and end > used_start for used_start, used_end in occupied):
                continue
            occupied.append((start, end))
            mentions.append(
                {
                    "family": pattern.family,
                    "term": pattern.term,
                    "matched_text": match.group(0),
                    "start": start,
                    "end": end,
                }
            )

    mentions.sort(key=lambda item: int(item["start"]))
    return mentions


def families_in_text(text: str) -> set[str]:
    return {mention["family"] for mention in extract_mentions(text)}


def average(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def sentiment_mean(sentences: list[str], analyzer: SentimentIntensityAnalyzer) -> float:
    if not sentences:
        return 0.0
    return average(analyzer.polarity_scores(sentence)["compound"] for sentence in sentences)


def clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def analyze_rows(rows: list[dict[str, object]], analyzer: SentimentIntensityAnalyzer) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []

    for row in rows:
        own_family = str(row["platform_family"])
        mentions = extract_mentions(str(row["response_text"]))
        total_platform_mentions = len(mentions)
        own_mentions = sum(1 for mention in mentions if mention["family"] == own_family)
        competitor_mentions = total_platform_mentions - own_mentions

        opmr = own_mentions / total_platform_mentions if total_platform_mentions else 0.0
        cos = 1.0 if competitor_mentions == 0 else 1.0 / (1.0 + competitor_mentions)

        sentences = sent_tokenize(str(row["response_text"]))
        own_sentences: list[str] = []
        competitor_sentences: list[str] = []
        for sentence in sentences:
            families = families_in_text(sentence)
            if own_family in families:
                own_sentences.append(sentence)
            if any(family != own_family for family in families):
                competitor_sentences.append(sentence)

        own_sentiment = sentiment_mean(own_sentences, analyzer)
        competitor_sentiment = sentiment_mean(competitor_sentences, analyzer)
        sds = own_sentiment - competitor_sentiment
        sds_norm = clip((sds + 1.0) / 2.0)
        bsi = (0.33 * opmr) + (0.33 * cos) + (0.33 * sds_norm)

        enriched_row = dict(row)
        enriched_row.update(
            {
                "total_platform_mentions": total_platform_mentions,
                "own_platform_mentions": own_mentions,
                "competitor_mentions": competitor_mentions,
                "opmr": opmr,
                "cos": cos,
                "own_sentence_count": len(own_sentences),
                "competitor_sentence_count": len(competitor_sentences),
                "own_sentiment": own_sentiment,
                "competitor_sentiment": competitor_sentiment,
                "sds": sds,
                "sds_norm": sds_norm,
                "bsi": bsi,
                "mention_families": ", ".join(sorted({str(mention['family']) for mention in mentions})),
            }
        )
        enriched.append(enriched_row)

    return enriched


def group_means(rows: list[dict[str, object]], keys: tuple[str, ...], metric_fields: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in keys)
        grouped[key].append(row)

    summary_rows: list[dict[str, object]] = []
    for key, members in sorted(grouped.items()):
        summary = {field: value for field, value in zip(keys, key)}
        summary["n"] = len(members)
        for metric in metric_fields:
            summary[f"mean_{metric}"] = average(float(member[metric]) for member in members)
        summary_rows.append(summary)
    return summary_rows


def pairwise_bsi_stats(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_platform = {
        platform: [float(row["bsi"]) for row in rows if row["platform"] == platform]
        for platform in PLATFORM_ORDER
    }

    pairs = [("GPT", "GEM"), ("GPT", "CLA"), ("GEM", "CLA")]
    results: list[dict[str, object]] = []
    for first, second in pairs:
        sample_a = by_platform[first]
        sample_b = by_platform[second]
        stat, p_value = mannwhitneyu(sample_a, sample_b, alternative="two-sided")
        results.append(
            {
                "platform_a": first,
                "platform_b": second,
                "mann_whitney_u": float(stat),
                "p_value": float(p_value),
                "cohens_d": float(cohens_d(sample_a, sample_b)),
                "mean_bsi_a": average(sample_a),
                "mean_bsi_b": average(sample_b),
            }
        )
    return results


def cohens_d(sample_a: list[float], sample_b: list[float]) -> float:
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    mean_diff = float(a.mean() - b.mean())
    pooled = math.sqrt((float(a.std(ddof=1) ** 2) + float(b.std(ddof=1) ** 2)) / 2.0)
    if pooled == 0:
        return 0.0
    return mean_diff / pooled


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_float(value: float) -> str:
    return f"{value:.4f}"


def write_summary_table(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = fieldnames or list(rows[0].keys())
    normalized_rows = []
    for row in rows:
        normalized = {}
        for key, value in row.items():
            normalized[key] = format_float(value) if isinstance(value, float) else value
        normalized_rows.append(normalized)
    write_csv(path, normalized_rows, fieldnames)


def color_for_value(value: float) -> str:
    value = clip(value)
    low = (237, 242, 244)
    high = (230, 57, 70)
    rgb = tuple(int(low[i] + (high[i] - low[i]) * value) for i in range(3))
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


def write_bar_chart(path: Path, title: str, labels: list[str], values: list[float], metric_label: str) -> None:
    width = 860
    height = 520
    margin_left = 90
    margin_bottom = 90
    chart_width = width - margin_left - 60
    chart_height = height - 120 - margin_bottom
    bar_gap = 32
    bar_width = max(50, int((chart_width - (bar_gap * (len(values) - 1))) / max(len(values), 1)))
    baseline_y = 80 + chart_height
    max_value = 1.0
    palette = ["#1d3557", "#457b9d", "#e76f51"]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f7f5"/>',
        f'<text x="{width / 2}" y="40" font-size="24" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" fill="#1f2933">{html.escape(title)}</text>',
        f'<text x="24" y="{height / 2}" font-size="14" transform="rotate(-90 24,{height / 2})" font-family="Segoe UI, Arial, sans-serif" fill="#52606d">{html.escape(metric_label)}</text>',
        f'<line x1="{margin_left}" y1="{baseline_y}" x2="{margin_left + chart_width}" y2="{baseline_y}" stroke="#52606d" stroke-width="2"/>',
        f'<line x1="{margin_left}" y1="80" x2="{margin_left}" y2="{baseline_y}" stroke="#52606d" stroke-width="2"/>',
    ]

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = baseline_y - (tick / max_value) * chart_height
        svg_parts.append(f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + chart_width}" y2="{y}" stroke="#d9e2ec" stroke-width="1"/>')
        svg_parts.append(
            f'<text x="{margin_left - 12}" y="{y + 5}" font-size="12" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" fill="#52606d">{tick:.2f}</text>'
        )

    start_x = margin_left + 36
    for index, (label, value) in enumerate(zip(labels, values)):
        height_ratio = clip(value, 0.0, max_value) / max_value
        bar_height = chart_height * height_ratio
        x = start_x + index * (bar_width + bar_gap)
        y = baseline_y - bar_height
        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" rx="10" fill="{palette[index % len(palette)]}"/>'
        )
        svg_parts.append(
            f'<text x="{x + (bar_width / 2)}" y="{baseline_y + 28}" font-size="14" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" fill="#1f2933">{html.escape(label)}</text>'
        )
        svg_parts.append(
            f'<text x="{x + (bar_width / 2)}" y="{y - 10}" font-size="13" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" fill="#1f2933">{value:.3f}</text>'
        )

    svg_parts.append("</svg>")
    path.write_text("\n".join(svg_parts), encoding="utf-8")


def write_heatmap_chart(path: Path, title: str, rows: list[str], cols: list[str], values: dict[tuple[str, str], float]) -> None:
    cell_width = 140
    cell_height = 42
    margin_left = 240
    margin_top = 90
    width = margin_left + (cell_width * len(cols)) + 40
    height = margin_top + (cell_height * len(rows)) + 70

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffaf3"/>',
        f'<text x="{width / 2}" y="38" font-size="24" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" fill="#1f2933">{html.escape(title)}</text>',
    ]

    for col_index, col_name in enumerate(cols):
        x = margin_left + (col_index * cell_width) + (cell_width / 2)
        svg_parts.append(
            f'<text x="{x}" y="{margin_top - 18}" font-size="14" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" fill="#1f2933">{html.escape(col_name)}</text>'
        )

    for row_index, row_name in enumerate(rows):
        y = margin_top + (row_index * cell_height)
        svg_parts.append(
            f'<text x="{margin_left - 14}" y="{y + 27}" font-size="13" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" fill="#1f2933">{html.escape(row_name)}</text>'
        )
        for col_index, col_name in enumerate(cols):
            x = margin_left + (col_index * cell_width)
            value = values.get((row_name, col_name), 0.0)
            fill = color_for_value(value)
            svg_parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>'
            )
            text_color = "#ffffff" if value >= 0.55 else "#1f2933"
            svg_parts.append(
                f'<text x="{x + (cell_width / 2)}" y="{y + 26}" font-size="13" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" fill="{text_color}">{value:.3f}</text>'
            )

    svg_parts.append("</svg>")
    path.write_text("\n".join(svg_parts), encoding="utf-8")


def write_text_report(
    path: Path,
    clean_stats: dict[str, int],
    verification_rows: list[dict[str, object]],
    platform_summary: list[dict[str, object]],
    category_summary: list[dict[str, object]],
    pairwise_stats_rows: list[dict[str, object]],
) -> None:
    lines: list[str] = []
    lines.append("Chapter 4 Analysis Summary")
    lines.append("==========================")
    lines.append("")
    lines.append("Data quality")
    lines.append("------------")
    lines.append(f"Main-sheet rows retained: {clean_stats['kept_main_rows']} of {clean_stats['raw_main_rows']}")
    lines.append(f"Verification rows retained: {clean_stats['kept_verification_rows']} of {clean_stats['raw_verification_rows']}")
    lines.append(f"Dropped invalid platform rows: {clean_stats['dropped_invalid_platform']}")
    lines.append(f"Dropped blank-response rows: {clean_stats['dropped_blank_response']}")
    lines.append("")

    platform_by_bsi = sorted(platform_summary, key=lambda item: float(item["mean_bsi"]), reverse=True)
    if platform_by_bsi:
        top_platform = platform_by_bsi[0]
        highest_opmr = max(platform_summary, key=lambda item: float(item["mean_opmr"]))
        highest_cos = max(platform_summary, key=lambda item: float(item["mean_cos"]))
        lines.append("Headline findings")
        lines.append("-----------------")
        lines.append(f"Highest mean BSI: {top_platform['platform']} ({float(top_platform['mean_bsi']):.4f})")
        lines.append(f"Highest mean OPMR: {highest_opmr['platform']} ({float(highest_opmr['mean_opmr']):.4f})")
        lines.append(f"Highest mean COS: {highest_cos['platform']} ({float(highest_cos['mean_cos']):.4f})")
        if category_summary:
            category_leader = max(category_summary, key=lambda item: float(item["mean_bsi"]))
            lines.append(
                f"Highest category-platform BSI: {category_leader['category']} / {category_leader['platform']} ({float(category_leader['mean_bsi']):.4f})"
            )
        lines.append("")

    lines.append("Platform means")
    lines.append("--------------")
    for row in platform_summary:
        lines.append(
            f"{row['platform']}: BSI={float(row['mean_bsi']):.4f}, OPMR={float(row['mean_opmr']):.4f}, "
            f"COS={float(row['mean_cos']):.4f}, SDS={float(row['mean_sds']):.4f}, "
            f"SDS_norm={float(row['mean_sds_norm']):.4f}, Mean words={float(row['mean_response_word_count']):.1f}"
        )
    lines.append("")

    lines.append("Pairwise Mann-Whitney U tests on BSI")
    lines.append("------------------------------------")
    for row in pairwise_stats_rows:
        lines.append(
            f"{row['platform_a']} vs {row['platform_b']}: U={row['mann_whitney_u']:.2f}, "
            f"p={row['p_value']:.4f}, Cohen's d={row['cohens_d']:.4f}"
        )
    lines.append("")

    lines.append("Verification sample")
    lines.append("-------------------")
    verification_summary = group_means(
        verification_rows,
        keys=("platform",),
        metric_fields=["response_word_count", "opmr", "cos", "sds", "sds_norm", "bsi"],
    )
    verification_summary.sort(key=lambda row: platform_sort_key(str(row["platform"])))
    for row in verification_summary:
        lines.append(
            f"{row['platform']}: BSI={float(row['mean_bsi']):.4f}, OPMR={float(row['mean_opmr']):.4f}, "
            f"COS={float(row['mean_cos']):.4f}, SDS={float(row['mean_sds']):.4f}"
        )
    lines.append("")
    lines.append("Notes")
    lines.append("-----")
    lines.append("Prompt IDs in the source workbook were normalized to P01-P20 format.")
    lines.append("Verification-sheet rows with platform NIL or blank responses were excluded.")
    lines.append("SDS_norm was clipped to the 0-1 range after applying (SDS + 1) / 2 to keep BSI bounded.")
    lines.append("Inter-rater reliability and sensitivity analysis were not computed automatically because they require additional coded inputs or alternate weighting instructions.")

    path.write_text("\n".join(lines), encoding="utf-8")


def export_outputs(main_rows: list[dict[str, object]], verification_rows: list[dict[str, object]], clean_stats: dict[str, int]) -> None:
    processed_fields = [
        "response_id",
        "source_set",
        "sheet_name",
        "row_number",
        "platform",
        "platform_family",
        "prompt_id",
        "category",
        "collection_date",
        "response_word_count",
        "computed_word_count",
        "total_platform_mentions",
        "own_platform_mentions",
        "competitor_mentions",
        "opmr",
        "cos",
        "own_sentence_count",
        "competitor_sentence_count",
        "own_sentiment",
        "competitor_sentiment",
        "sds",
        "sds_norm",
        "bsi",
        "mention_families",
        "prompt_text",
        "response_text",
    ]

    write_summary_table(DATA_DIR / "processed_main.csv", main_rows, processed_fields)
    write_summary_table(DATA_DIR / "processed_verification.csv", verification_rows, processed_fields)

    metric_fields = [
        "response_word_count",
        "total_platform_mentions",
        "own_platform_mentions",
        "competitor_mentions",
        "opmr",
        "cos",
        "sds",
        "sds_norm",
        "bsi",
    ]
    platform_summary = group_means(main_rows, keys=("platform",), metric_fields=metric_fields)
    category_summary = group_means(main_rows, keys=("category", "platform"), metric_fields=["opmr", "cos", "sds", "sds_norm", "bsi"])
    prompt_summary = group_means(main_rows, keys=("prompt_id", "platform"), metric_fields=["opmr", "cos", "sds", "sds_norm", "bsi"])
    pairwise_stats_rows = pairwise_bsi_stats(main_rows)
    platform_summary.sort(key=lambda row: platform_sort_key(str(row["platform"])))
    category_summary.sort(key=lambda row: (str(row["category"]), platform_sort_key(str(row["platform"]))))
    prompt_summary.sort(key=lambda row: (prompt_sort_key(str(row["prompt_id"])), platform_sort_key(str(row["platform"]))))

    write_summary_table(TABLES_DIR / "platform_summary.csv", platform_summary)
    write_summary_table(TABLES_DIR / "category_platform_summary.csv", category_summary)
    write_summary_table(TABLES_DIR / "prompt_platform_summary.csv", prompt_summary)
    write_summary_table(STATS_DIR / "pairwise_bsi_stats.csv", pairwise_stats_rows)

    labels = [str(row["platform"]) for row in platform_summary]
    write_bar_chart(
        CHARTS_DIR / "bsi_by_platform.svg",
        "Bias Score Index by Platform",
        labels,
        [float(row["mean_bsi"]) for row in platform_summary],
        "Mean BSI",
    )
    write_bar_chart(
        CHARTS_DIR / "opmr_by_platform.svg",
        "Own-Platform Mention Rate by Platform",
        labels,
        [float(row["mean_opmr"]) for row in platform_summary],
        "Mean OPMR",
    )
    write_bar_chart(
        CHARTS_DIR / "cos_by_platform.svg",
        "Competitor Omission Score by Platform",
        labels,
        [float(row["mean_cos"]) for row in platform_summary],
        "Mean COS",
    )

    categories = sorted({str(row["category"]) for row in category_summary})
    heatmap_values = {
        (str(row["category"]), str(row["platform"])): float(row["mean_bsi"])
        for row in category_summary
    }
    write_heatmap_chart(
        CHARTS_DIR / "bsi_category_heatmap.svg",
        "Mean BSI by Category and Platform",
        categories,
        PLATFORM_ORDER,
        heatmap_values,
    )

    write_text_report(
        STATS_DIR / "results_summary.txt",
        clean_stats=clean_stats,
        verification_rows=verification_rows,
        platform_summary=platform_summary,
        category_summary=category_summary,
        pairwise_stats_rows=pairwise_stats_rows,
    )


def main() -> None:
    ensure_output_dirs()
    if not WORKBOOK_PATH.exists():
        raise FileNotFoundError(f"Workbook not found: {WORKBOOK_PATH}")

    analyzer = SentimentIntensityAnalyzer()
    parsed_sheets = parse_workbook(WORKBOOK_PATH)
    main_rows, verification_rows, clean_stats = clean_records(parsed_sheets)
    analyzed_main = analyze_rows(main_rows, analyzer)
    analyzed_verification = analyze_rows(verification_rows, analyzer)
    export_outputs(analyzed_main, analyzed_verification, clean_stats)

    print(f"Main rows analyzed: {len(analyzed_main)}")
    print(f"Verification rows analyzed: {len(analyzed_verification)}")
    print(f"Outputs written to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
