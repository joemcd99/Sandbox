#!/usr/bin/env python3
"""Multi-faceted B2B name matching with false-positive controls.

This script:
1) Builds sample business DataFrames.
2) Cleans and normalizes legal entity names.
3) Computes multiple matching signals.
4) Applies strict gates/penalties and weighted scoring.
5) Produces one score and decision band per pair.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Dict, List, Set, Tuple

import pandas as pd
from rapidfuzz import fuzz

try:
    import jellyfish
except Exception:  # pragma: no cover
    jellyfish = None

try:
    from unidecode import unidecode
except Exception:  # pragma: no cover
    unidecode = None


LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "limited",
    "llc",
    "l l c",
    "llp",
    "lp",
    "plc",
    "gmbh",
    "sa",
    "ag",
    "bv",
    "pte",
    "pty",
    "sarl",
}

BUSINESS_NOISE = {
    "the",
    "group",
    "holdings",
    "holding",
    "partners",
    "ventures",
    "international",
    "global",
}

TOKEN_ALIASES = {
    "&": "and",
    "intl": "international",
    "int'l": "international",
    "technologies": "technology",
    "tech": "technology",
    "svcs": "services",
    "svc": "services",
    "mfg": "manufacturing",
    "manufacture": "manufacturing",
    "mgmt": "management",
}

ANCHOR_ALIASES = {
    "oneil": "oneill",
    "o'neil": "oneill",
    "hlth": "health",
}

GENERIC_CORE_TOKENS = {
    "solutions",
    "systems",
    "services",
    "consulting",
    "logistics",
    "trading",
    "management",
    "technology",
}


@dataclass
class ParsedBusiness:
    raw: str
    normalized: str
    tokens: List[str]
    token_set: Set[str]
    anchor: str
    first_token: str
    acronym: str
    metaphone: str
    soundex: str
    block_key: str


@dataclass
class MatchConfig:
    auto_threshold: float = 92.0
    review_threshold: float = 84.0
    reject_threshold: float = 74.0

    min_token_set: float = 88.0
    min_jaro_winkler: float = 86.0

    require_anchor_match: bool = True
    require_token_overlap: bool = True
    min_shared_tokens: int = 1

    top_margin_required: float = 4.0

    weights: Dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = {
                "exact": 0.30,
                "token_sort": 0.20,
                "token_set": 0.25,
                "jaro_winkler": 0.10,
                "phonetic": 0.10,
                "acronym": 0.05,
            }


def _ascii_fold(value: str) -> str:
    if unidecode is not None:
        return unidecode(value)
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")


def _safe_metaphone(value: str) -> str:
    if jellyfish is None or not value:
        return ""
    return jellyfish.metaphone(value)


def _safe_soundex(value: str) -> str:
    if jellyfish is None or not value:
        return ""
    return jellyfish.soundex(value)


def _safe_jaro_winkler(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if jellyfish is None:
        return fuzz.WRatio(a, b) / 100.0
    return float(jellyfish.jaro_winkler_similarity(a, b))


def _normalize_token(token: str) -> str:
    token = token.strip().lower()
    if token in TOKEN_ALIASES:
        token = TOKEN_ALIASES[token]
    return ANCHOR_ALIASES.get(token, token)


def _tokenize_business(value: str) -> List[str]:
    cleaned = value.lower().replace("&", " and ")
    cleaned = re.sub(r"[^a-z0-9\s']", " ", cleaned)
    tokens = [_normalize_token(t.strip("'")) for t in cleaned.split() if t.strip("'")]

    filtered: List[str] = []
    for t in tokens:
        if not t:
            continue
        if t in LEGAL_SUFFIXES:
            continue
        if t in BUSINESS_NOISE:
            continue
        filtered.append(t)
    return filtered


def _normalize_anchor(token: str) -> str:
    if not token:
        return ""
    return ANCHOR_ALIASES.get(token, token)


def parse_and_normalize_name(raw_name: str) -> ParsedBusiness:
    """Backwards-compatible function name; now parses business names."""
    raw_name = "" if raw_name is None else str(raw_name)
    folded = _ascii_fold(raw_name)

    tokens = _tokenize_business(folded)
    normalized = " ".join(tokens)
    acronym = "".join(t[0] for t in tokens if t)
    anchor = _normalize_anchor(tokens[-1]) if tokens else ""
    first_token = tokens[0] if tokens else ""
    anchor_soundex = _safe_soundex(anchor)
    first_initial = first_token[0] if first_token else ""
    block_key = f"{first_initial}:{anchor_soundex or anchor}"

    return ParsedBusiness(
        raw=raw_name,
        normalized=normalized,
        tokens=tokens,
        token_set=set(tokens),
        anchor=anchor,
        first_token=first_token,
        acronym=acronym,
        metaphone=_safe_metaphone(normalized.replace(" ", "")),
        soundex=_safe_soundex(normalized.replace(" ", "")),
        block_key=block_key,
    )


def _anchors_match(left: ParsedBusiness, right: ParsedBusiness) -> bool:
    return bool(left.anchor and right.anchor and left.anchor == right.anchor)


def _shared_token_count(left: ParsedBusiness, right: ParsedBusiness) -> int:
    return len(left.token_set.intersection(right.token_set))


def _component_scores(left: ParsedBusiness, right: ParsedBusiness) -> Dict[str, float]:
    exact = 100.0 if left.normalized and left.normalized == right.normalized else 0.0
    token_sort = float(fuzz.token_sort_ratio(left.normalized, right.normalized))
    token_set = float(fuzz.token_set_ratio(left.normalized, right.normalized))
    jaro_winkler = _safe_jaro_winkler(left.normalized, right.normalized) * 100.0

    phonetic = 0.0
    if left.metaphone and right.metaphone and left.metaphone == right.metaphone:
        phonetic += 50.0
    if left.soundex and right.soundex and left.soundex == right.soundex:
        phonetic += 50.0

    acronym = 0.0
    if left.acronym and right.acronym:
        acronym = float(fuzz.ratio(left.acronym, right.acronym))

    return {
        "exact": exact,
        "token_sort": token_sort,
        "token_set": token_set,
        "jaro_winkler": jaro_winkler,
        "phonetic": phonetic,
        "acronym": acronym,
    }


def _penalty_points(left: ParsedBusiness, right: ParsedBusiness) -> float:
    penalty = 0.0

    if len(left.tokens) <= 1 or len(right.tokens) <= 1:
        penalty += 15.0

    token_gap = abs(len(left.tokens) - len(right.tokens))
    if token_gap >= 2:
        penalty += 8.0
    elif token_gap == 1:
        penalty += 3.0

    if left.anchor and right.anchor and left.anchor == right.anchor:
        if left.anchor in GENERIC_CORE_TOKENS:
            penalty += 7.0

    if _shared_token_count(left, right) == 1 and left.anchor == right.anchor:
        penalty += 5.0

    return penalty


def _passes_hard_gates(
    left: ParsedBusiness,
    right: ParsedBusiness,
    components: Dict[str, float],
    config: MatchConfig,
) -> Tuple[bool, str]:
    if config.require_anchor_match and not _anchors_match(left, right):
        return False, "failed_anchor_gate"

    if config.require_token_overlap and _shared_token_count(left, right) < config.min_shared_tokens:
        return False, "failed_token_overlap_gate"

    if components["token_set"] < config.min_token_set:
        return False, "failed_token_set_minimum"

    if components["jaro_winkler"] < config.min_jaro_winkler:
        return False, "failed_jaro_minimum"

    return True, "passed"


def _decision_band(score: float, config: MatchConfig) -> str:
    if score >= config.auto_threshold:
        return "auto_match"
    if score >= config.review_threshold:
        return "manual_review"
    if score >= config.reject_threshold:
        return "weak_match"
    return "reject"


def _blocked_candidate_indices(left_profile: ParsedBusiness, right_profiles: Dict[int, ParsedBusiness]) -> List[int]:
    in_block = [idx for idx, p in right_profiles.items() if p.block_key == left_profile.block_key]
    if in_block:
        return in_block

    fallback = [idx for idx, p in right_profiles.items() if _anchors_match(left_profile, p)]
    if fallback:
        return fallback

    return list(right_profiles.keys())


def score_name_pair(
    left: ParsedBusiness,
    right: ParsedBusiness,
    config: MatchConfig,
) -> Tuple[float, Dict[str, float], str]:
    components = _component_scores(left, right)
    passed, gate_reason = _passes_hard_gates(left, right, components, config)
    if not passed:
        return 0.0, components, gate_reason

    weighted_sum = sum(components[k] * config.weights[k] for k in config.weights)
    weighted_score = weighted_sum / sum(config.weights.values())
    final = max(0.0, weighted_score - _penalty_points(left, right))
    return round(final, 2), components, gate_reason


def build_sample_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create sample business datasets with legal suffix variants."""
    left = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "name": [
                "North Star Logistics Ltd.",
                "Acme Industrial Co.",
                "Blue River Technologies Inc",
                "Summit Health Services LLC",
                "Redwood Manufacturing Limited",
                "Pioneer Energy Holdings",
                "Global Trade Partners Inc.",
                "Urban Data Systems Co",
            ],
        }
    )

    right = pd.DataFrame(
        {
            "customer_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "full_name": [
                "North Star Logistics Limited",
                "ACME Industrial Company",
                "Blue River Technology Incorporated",
                "Summit Health Service",
                "Redwood MFG Ltd",
                "Pioneer Energy",
                "Global Trading Partners",
                "Urban Data System Inc",
                "Northern Star Logistics Co",
                "Acme Logistics Inc",
            ],
        }
    )

    return left, right


def build_sample_labels() -> pd.DataFrame:
    """Small labeled set for B2B precision/false-positive evaluation."""
    return pd.DataFrame(
        {
            "left_name": [
                "North Star Logistics Ltd.",
                "Acme Industrial Co.",
                "Blue River Technologies Inc",
                "Pioneer Energy Holdings",
                "Acme Industrial Co.",
                "Urban Data Systems Co",
            ],
            "right_name": [
                "North Star Logistics Limited",
                "ACME Industrial Company",
                "Blue River Technology Incorporated",
                "Pioneer Energy",
                "Acme Logistics Inc",
                "Global Trading Partners",
            ],
            "is_true_match": [1, 1, 1, 1, 0, 0],
        }
    )


def rank_matches(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_col: str = "name",
    right_col: str = "full_name",
    top_n: int = 3,
    config: MatchConfig | None = None,
) -> pd.DataFrame:
    config = config or MatchConfig()

    left_profiles = {idx: parse_and_normalize_name(value) for idx, value in df_left[left_col].fillna("").items()}
    right_profiles = {idx: parse_and_normalize_name(value) for idx, value in df_right[right_col].fillna("").items()}

    rows: List[Dict[str, object]] = []

    for left_idx, left_name in left_profiles.items():
        per_left: List[Dict[str, object]] = []
        candidate_indices = _blocked_candidate_indices(left_name, right_profiles)

        for right_idx in candidate_indices:
            right_name = right_profiles[right_idx]
            score, details, gate_reason = score_name_pair(left_name, right_name, config)
            band = _decision_band(score, config)
            if band == "reject":
                continue

            per_left.append(
                {
                    "left_index": left_idx,
                    "left_name": left_name.raw,
                    "left_normalized": left_name.normalized,
                    "left_anchor": left_name.anchor,
                    "right_index": right_idx,
                    "right_name": right_name.raw,
                    "right_normalized": right_name.normalized,
                    "right_anchor": right_name.anchor,
                    "match_score": score,
                    "decision_band": band,
                    "gate_reason": gate_reason,
                    "shared_tokens": _shared_token_count(left_name, right_name),
                    "penalty_points": round(_penalty_points(left_name, right_name), 2),
                    **{f"component_{k}": round(v, 2) for k, v in details.items()},
                }
            )

        per_left.sort(key=lambda r: r["match_score"], reverse=True)

        if len(per_left) >= 2:
            top_delta = per_left[0]["match_score"] - per_left[1]["match_score"]
            per_left[0]["top_margin"] = round(top_delta, 2)
            if top_delta < config.top_margin_required and per_left[0]["decision_band"] == "auto_match":
                per_left[0]["decision_band"] = "manual_review"
                per_left[0]["gate_reason"] = "tight_runner_up_margin"
        elif len(per_left) == 1:
            per_left[0]["top_margin"] = 999.0

        rows.extend(per_left[:top_n])

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result = result.sort_values(["right_index", "match_score"], ascending=[True, False])
    dedup_rows = []
    for _, group in result.groupby("right_index", as_index=False):
        top = group.iloc[0].copy()
        if len(group) >= 2:
            margin = float(group.iloc[0]["match_score"] - group.iloc[1]["match_score"])
            if margin < config.top_margin_required and top["decision_band"] == "auto_match":
                top["decision_band"] = "manual_review"
                top["gate_reason"] = "right_side_competition"
        dedup_rows.append(top)

    final = pd.DataFrame(dedup_rows)
    final = final.sort_values(["left_index", "match_score"], ascending=[True, False])
    return final.reset_index(drop=True)


def evaluate_with_labels(
    labels_df: pd.DataFrame,
    left_col: str = "left_name",
    right_col: str = "right_name",
    label_col: str = "is_true_match",
    config: MatchConfig | None = None,
) -> pd.DataFrame:
    config = config or MatchConfig()

    y_true: List[int] = []
    y_pred: List[int] = []

    for _, row in labels_df.iterrows():
        left = parse_and_normalize_name(row[left_col])
        right = parse_and_normalize_name(row[right_col])
        score, _, _ = score_name_pair(left, right, config)
        pred = 1 if score >= config.auto_threshold else 0
        y_true.append(int(row[label_col]))
        y_pred.append(pred)

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return pd.DataFrame(
        {
            "metric": ["precision", "recall", "false_positive_rate", "tp", "fp", "fn", "tn"],
            "value": [round(precision, 4), round(recall, 4), round(fpr, 4), tp, fp, fn, tn],
        }
    )


def _print_quick_demo(df_left: pd.DataFrame, df_right: pd.DataFrame, matches: pd.DataFrame) -> None:
    print("\n=== Sample DataFrame: Left (B2B) ===")
    print(df_left)
    print("\n=== Sample DataFrame: Right (B2B) ===")
    print(df_right)
    print("\n=== Controlled B2B Match Results ===")
    if matches.empty:
        print("No matches above thresholds.")
    else:
        print(matches.to_string(index=False))


def main() -> None:
    left, right = build_sample_dataframes()
    config = MatchConfig()
    matches = rank_matches(left, right, left_col="name", right_col="full_name", top_n=3, config=config)
    metrics = evaluate_with_labels(build_sample_labels(), config=config)
    _print_quick_demo(left, right, matches)
    print("\n=== Sample Labeled Evaluation (B2B) ===")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
