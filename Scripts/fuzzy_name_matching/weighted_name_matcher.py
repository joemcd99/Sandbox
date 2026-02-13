#!/usr/bin/env python3
"""
Weighted, layered name matching.

This script:
1) Cleans and normalizes name fields.
2) Applies multiple matching approaches (exact, fuzzy, phonetic, initials).
3) Combines approach scores into one weighted score.
4) Returns best matches above a threshold.
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import pandas as pd
from rapidfuzz import fuzz, process


TITLE_TOKENS = {
    "mr",
    "mrs",
    "ms",
    "miss",
    "dr",
    "prof",
    "sir",
    "madam",
    "mx",
}

SUFFIX_TOKENS = {
    "jr",
    "sr",
    "ii",
    "iii",
    "iv",
    "v",
}

NICKNAME_MAP = {
    "bob": "robert",
    "rob": "robert",
    "bobby": "robert",
    "bill": "william",
    "will": "william",
    "liz": "elizabeth",
    "beth": "elizabeth",
    "kate": "katherine",
    "kathy": "katherine",
    "mike": "michael",
    "mikey": "michael",
    "jen": "jennifer",
    "jenny": "jennifer",
    "joe": "joseph",
    "joey": "joseph",
    "tony": "anthony",
    "dave": "david",
    "steve": "steven",
    "chris": "christopher",
    "andy": "andrew",
}

WEIGHTS: Dict[str, float] = {
    "exact": 0.20,
    "token_sort": 0.25,
    "token_set": 0.20,
    "partial": 0.10,
    "phonetic": 0.15,
    "initial_last": 0.10,
}


@dataclass
class NameProfile:
    index: int
    raw: str
    normalized: str
    tokens: List[str]
    sorted_tokens_str: str
    initials: str
    first: str
    last: str
    phonetic_tokens: Set[str]


def _ascii_fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def normalize_name(name: str) -> str:
    if name is None:
        return ""
    value = _ascii_fold(str(name)).lower()
    value = re.sub(r"[,/._-]+", " ", value)
    value = re.sub(r"[^a-z0-9\s']", " ", value)
    tokens = [t.strip("'") for t in value.split() if t.strip("'")]
    tokens = [NICKNAME_MAP.get(t, t) for t in tokens]
    tokens = [t for t in tokens if t not in TITLE_TOKENS and t not in SUFFIX_TOKENS]
    return " ".join(tokens)


def soundex(token: str) -> str:
    if not token:
        return ""
    token = re.sub(r"[^a-z]", "", token.lower())
    if not token:
        return ""

    mapping = {
        "b": "1",
        "f": "1",
        "p": "1",
        "v": "1",
        "c": "2",
        "g": "2",
        "j": "2",
        "k": "2",
        "q": "2",
        "s": "2",
        "x": "2",
        "z": "2",
        "d": "3",
        "t": "3",
        "l": "4",
        "m": "5",
        "n": "5",
        "r": "6",
    }

    first_letter = token[0].upper()
    encoded = []
    previous = ""
    for char in token[1:]:
        code = mapping.get(char, "0")
        if code != "0" and code != previous:
            encoded.append(code)
        previous = code
    code = first_letter + "".join(encoded)
    return (code + "000")[:4]


def build_profile(index: int, raw_name: str) -> NameProfile:
    normalized = normalize_name(raw_name)
    tokens = normalized.split()
    sorted_tokens = " ".join(sorted(tokens))
    initials = "".join(t[0] for t in tokens if t)
    first = tokens[0] if tokens else ""
    last = tokens[-1] if tokens else ""
    phonetic_tokens = {soundex(t) for t in tokens if t}

    return NameProfile(
        index=index,
        raw=str(raw_name),
        normalized=normalized,
        tokens=tokens,
        sorted_tokens_str=sorted_tokens,
        initials=initials,
        first=first,
        last=last,
        phonetic_tokens=phonetic_tokens,
    )


def phonetic_score(a_codes: Set[str], b_codes: Set[str]) -> float:
    if not a_codes or not b_codes:
        return 0.0
    intersection = len(a_codes.intersection(b_codes))
    union = len(a_codes.union(b_codes))
    if union == 0:
        return 0.0
    return 100.0 * (intersection / union)


def initial_last_score(a: NameProfile, b: NameProfile) -> float:
    if not a.last or not b.last:
        return 0.0
    last_match = a.last == b.last
    first_initial_match = bool(a.first and b.first and a.first[0] == b.first[0])
    if last_match and first_initial_match:
        return 100.0
    if last_match:
        return 50.0
    return 0.0


def combined_score(a: NameProfile, b: NameProfile) -> Tuple[float, Dict[str, float]]:
    exact = 100.0 if a.normalized == b.normalized and a.normalized else 0.0
    token_sort = fuzz.token_sort_ratio(a.normalized, b.normalized)
    token_set = fuzz.token_set_ratio(a.normalized, b.normalized)
    partial = fuzz.partial_ratio(a.normalized, b.normalized)
    phon = phonetic_score(a.phonetic_tokens, b.phonetic_tokens)
    init_last = initial_last_score(a, b)

    components = {
        "exact": exact,
        "token_sort": float(token_sort),
        "token_set": float(token_set),
        "partial": float(partial),
        "phonetic": float(phon),
        "initial_last": float(init_last),
    }
    weighted_sum = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)
    total_weight = sum(WEIGHTS.values())
    return weighted_sum / total_weight, components


def generate_candidates(
    profile_a: NameProfile,
    profiles_b: List[NameProfile],
    normalized_choices: List[str],
    top_candidates: int,
) -> Set[int]:
    candidate_indices: Set[int] = set()

    # Layer 1: deterministic candidates
    for profile_b in profiles_b:
        if not profile_a.normalized or not profile_b.normalized:
            continue
        if (
            profile_a.normalized == profile_b.normalized
            or profile_a.sorted_tokens_str == profile_b.sorted_tokens_str
            or (
                profile_a.first == profile_b.first
                and profile_a.last
                and profile_a.last == profile_b.last
            )
        ):
            candidate_indices.add(profile_b.index)

    # Layer 2: fuzzy shortlist
    fuzzy_hits = process.extract(
        query=profile_a.normalized,
        choices=normalized_choices,
        scorer=fuzz.WRatio,
        limit=top_candidates,
    )
    for _, _, idx in fuzzy_hits:
        candidate_indices.add(idx)

    # Layer 3: phonetic + last-name candidate expansion
    if profile_a.last:
        last_code = soundex(profile_a.last)
        for profile_b in profiles_b:
            if profile_b.last and soundex(profile_b.last) == last_code:
                candidate_indices.add(profile_b.index)

    return candidate_indices


def layered_weighted_name_match(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    name_col_a: str,
    name_col_b: str,
    threshold: float = 75.0,
    top_k: int = 1,
    top_candidates: int = 20,
) -> pd.DataFrame:
    if name_col_a not in df_a.columns:
        raise ValueError(f"Column '{name_col_a}' not found in source A")
    if name_col_b not in df_b.columns:
        raise ValueError(f"Column '{name_col_b}' not found in source B")

    df_a = df_a.copy()
    df_b = df_b.copy()
    df_a[name_col_a] = df_a[name_col_a].fillna("").astype(str)
    df_b[name_col_b] = df_b[name_col_b].fillna("").astype(str)

    profiles_b: List[NameProfile] = [
        build_profile(idx, name) for idx, name in enumerate(df_b[name_col_b].tolist())
    ]
    normalized_choices = [p.normalized for p in profiles_b]

    results = []

    for idx_a, raw_a in enumerate(df_a[name_col_a].tolist()):
        profile_a = build_profile(idx_a, raw_a)
        if not profile_a.normalized:
            continue

        candidate_indices = generate_candidates(
            profile_a=profile_a,
            profiles_b=profiles_b,
            normalized_choices=normalized_choices,
            top_candidates=top_candidates,
        )
        if not candidate_indices:
            continue

        scored = []
        for idx_b in candidate_indices:
            profile_b = profiles_b[idx_b]
            score, components = combined_score(profile_a, profile_b)
            if score >= threshold:
                scored.append((score, profile_b, components))

        scored.sort(key=lambda x: x[0], reverse=True)
        for rank, (score, profile_b, components) in enumerate(scored[:top_k], start=1):
            results.append(
                {
                    "index_a": idx_a,
                    "name_a": profile_a.raw,
                    "normalized_a": profile_a.normalized,
                    "index_b": profile_b.index,
                    "name_b": profile_b.raw,
                    "normalized_b": profile_b.normalized,
                    "weighted_score": round(score, 2),
                    "rank": rank,
                    "score_exact": round(components["exact"], 2),
                    "score_token_sort": round(components["token_sort"], 2),
                    "score_token_set": round(components["token_set"], 2),
                    "score_partial": round(components["partial"], 2),
                    "score_phonetic": round(components["phonetic"], 2),
                    "score_initial_last": round(components["initial_last"], 2),
                }
            )

    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean/normalize names and perform layered weighted matching "
            "between two CSV sources."
        )
    )
    parser.add_argument("--source-a", required=True, help="Path to CSV for source A")
    parser.add_argument("--source-b", required=True, help="Path to CSV for source B")
    parser.add_argument(
        "--name-col-a",
        default="name",
        help="Name column in source A (default: name)",
    )
    parser.add_argument(
        "--name-col-b",
        default="name",
        help="Name column in source B (default: name)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=75.0,
        help="Minimum weighted score to keep a match (0-100, default: 75)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of matches to keep per name from source A (default: 1)",
    )
    parser.add_argument(
        "--top-candidates",
        type=int,
        default=20,
        help="Fuzzy shortlist size before weighted scoring (default: 20)",
    )
    parser.add_argument(
        "--output",
        default="matches_weighted.csv",
        help="Output CSV file (default: matches_weighted.csv)",
    )
    return parser.parse_args()


def read_csv_loose(path: str, name_col: str) -> pd.DataFrame:
    """
    Read CSV robustly, including rows where name fields contain unquoted commas.

    If a row has more fields than the header and `name_col` is the last header,
    the extra fields are merged into `name_col`.
    """
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return pd.DataFrame()

    header = [h.strip() for h in rows[0]]
    if name_col not in header:
        # Fall back to pandas default behavior if the requested name column
        # is not discoverable in the raw header.
        return pd.read_csv(path)

    name_idx = header.index(name_col)
    parsed_rows: List[List[str]] = []

    for row in rows[1:]:
        if not row:
            continue
        if len(row) > len(header) and name_idx == len(header) - 1:
            # Join overflow into the last column (typically name field).
            row = row[:name_idx] + [",".join(row[name_idx:])]
        elif len(row) > len(header):
            row = row[: len(header)]
        elif len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        parsed_rows.append(row)

    return pd.DataFrame(parsed_rows, columns=header)


def main() -> None:
    args = parse_args()
    df_a = read_csv_loose(args.source_a, args.name_col_a)
    df_b = read_csv_loose(args.source_b, args.name_col_b)

    matches = layered_weighted_name_match(
        df_a=df_a,
        df_b=df_b,
        name_col_a=args.name_col_a,
        name_col_b=args.name_col_b,
        threshold=args.threshold,
        top_k=args.top_k,
        top_candidates=args.top_candidates,
    )

    matches.to_csv(args.output, index=False)
    print(f"Found {len(matches)} matches with score >= {args.threshold}")
    print(f"Saved results to {args.output}")
    if not matches.empty:
        print(matches.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
