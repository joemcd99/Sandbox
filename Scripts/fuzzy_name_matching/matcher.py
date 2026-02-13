import argparse
import csv
import re
import sys
import unicodedata
from typing import Dict, List, Set, Tuple

import pandas as pd
from rapidfuzz import fuzz, process


class SmartMatcher:
    """
    Build-off of the notebook matcher:
    - robust normalization
    - phonetic support
    - weighted multi-metric scoring
    """

    def __init__(self) -> None:
        self.noise_words = {
            "inc",
            "corp",
            "llc",
            "ltd",
            "limited",
            "company",
            "mr",
            "mrs",
            "ms",
            "dr",
            "phd",
            "jr",
            "sr",
            "ii",
            "iii",
        }
        self.nickname_map = {
            "bob": "robert",
            "rob": "robert",
            "bobby": "robert",
            "bill": "william",
            "will": "william",
            "mike": "michael",
            "jen": "jennifer",
            "jenny": "jennifer",
            "kathy": "katherine",
            "kate": "katherine",
            "joe": "joseph",
            "dave": "david",
            "steve": "steven",
            "chris": "christopher",
        }
        self.weights = {
            "wratio": 0.30,
            "token_sort": 0.20,
            "token_set": 0.20,
            "partial": 0.10,
            "phonetic": 0.10,
            "initial_last": 0.10,
        }

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
        text = text.lower()
        text = re.sub(r"[,/._-]+", " ", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        tokens = [t for t in text.split() if t and t not in self.noise_words]
        tokens = [self.nickname_map.get(t, t) for t in tokens]
        return " ".join(tokens)

    def soundex(self, token: str) -> str:
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
        first = token[0].upper()
        out = []
        prev = ""
        for ch in token[1:]:
            code = mapping.get(ch, "0")
            if code != "0" and code != prev:
                out.append(code)
            prev = code
        return (first + "".join(out) + "000")[:4]

    def phonetic_score(self, a: str, b: str) -> float:
        ta = [t for t in a.split() if t]
        tb = [t for t in b.split() if t]
        if not ta or not tb:
            return 0.0
        sa = {self.soundex(t) for t in ta}
        sb = {self.soundex(t) for t in tb}
        inter = len(sa.intersection(sb))
        union = len(sa.union(sb))
        if union == 0:
            return 0.0
        return 100.0 * (inter / union)

    def initial_last_score(self, a: str, b: str) -> float:
        ta = a.split()
        tb = b.split()
        if not ta or not tb:
            return 0.0
        first_a, last_a = ta[0], ta[-1]
        first_b, last_b = tb[0], tb[-1]
        if last_a == last_b and first_a and first_b and first_a[0] == first_b[0]:
            return 100.0
        if last_a == last_b:
            return 50.0
        return 0.0

    def get_scores(self, name_a: str, name_b: str) -> Tuple[float, Dict[str, float]]:
        clean_a = self.normalize(name_a)
        clean_b = self.normalize(name_b)
        if not clean_a or not clean_b:
            return 0.0, {}

        components = {
            "wratio": float(fuzz.WRatio(clean_a, clean_b)),
            "token_sort": float(fuzz.token_sort_ratio(clean_a, clean_b)),
            "token_set": float(fuzz.token_set_ratio(clean_a, clean_b)),
            "partial": float(fuzz.partial_ratio(clean_a, clean_b)),
            "phonetic": float(self.phonetic_score(clean_a, clean_b)),
            "initial_last": float(self.initial_last_score(clean_a, clean_b)),
        }
        final = sum(self.weights[k] * components[k] for k in self.weights) / sum(self.weights.values())
        return final, components


def read_csv_loose(path: str, name_col: str) -> pd.DataFrame:
    """
    Handles common bad CSV rows where names contain unquoted commas.
    """
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return pd.DataFrame()
    header = [h.strip() for h in rows[0]]
    if name_col not in header:
        return pd.read_csv(path)

    name_idx = header.index(name_col)
    normalized_rows: List[List[str]] = []
    for row in rows[1:]:
        if not row:
            continue
        if len(row) > len(header) and name_idx == len(header) - 1:
            row = row[:name_idx] + [",".join(row[name_idx:])]
        elif len(row) > len(header):
            row = row[: len(header)]
        elif len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        normalized_rows.append(row)
    return pd.DataFrame(normalized_rows, columns=header)


def fuzzy_match(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    col_a: str,
    col_b: str,
    threshold: float = 80.0,
    output_file: str = "matches.csv",
    top_k: int = 1,
    top_candidates: int = 20,
) -> pd.DataFrame:
    if col_a not in df_a.columns:
        raise ValueError(f"Column '{col_a}' not found in DataFrame A")
    if col_b not in df_b.columns:
        raise ValueError(f"Column '{col_b}' not found in DataFrame B")

    matcher = SmartMatcher()
    names_a = df_a[col_a].fillna("").astype(str).tolist()
    names_b = df_b[col_b].fillna("").astype(str).tolist()
    normalized_b = [matcher.normalize(n) for n in names_b]

    results: List[Dict[str, object]] = []
    print(f"Matching {len(names_a)} names from A against {len(names_b)} names from B...")

    for idx_a, name_a in enumerate(names_a):
        clean_a = matcher.normalize(name_a)
        if not clean_a:
            continue

        candidates = process.extract(
            clean_a,
            normalized_b,
            scorer=fuzz.WRatio,
            limit=top_candidates,
        )

        scored = []
        for _, _, idx_b in candidates:
            score, components = matcher.get_scores(name_a, names_b[idx_b])
            if score >= threshold:
                scored.append((score, idx_b, components))

        scored.sort(key=lambda x: x[0], reverse=True)
        for rank, (score, idx_b, components) in enumerate(scored[:top_k], start=1):
            results.append(
                {
                    "Index A": idx_a,
                    "Original Name (A)": name_a,
                    "Index B": idx_b,
                    "Matched Name (B)": names_b[idx_b],
                    "Weighted Score": round(score, 2),
                    "Rank": rank,
                    "WRatio": round(components["wratio"], 2),
                    "Token Sort": round(components["token_sort"], 2),
                    "Token Set": round(components["token_set"], 2),
                    "Partial": round(components["partial"], 2),
                    "Phonetic": round(components["phonetic"], 2),
                    "Initial+Last": round(components["initial_last"], 2),
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Found {len(results_df)} matches with score >= {threshold}")
    print(f"Saved matches to {output_file}")
    if not results_df.empty:
        print(results_df.head(20).to_string(index=False))
    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart fuzzy name matching with normalization and weighted scoring."
    )
    parser.add_argument("--source-a", help="CSV path for source A")
    parser.add_argument("--source-b", help="CSV path for source B")
    parser.add_argument("--col-a", default="name", help="Name column in source A")
    parser.add_argument("--col-b", default="full_name", help="Name column in source B")
    parser.add_argument("--threshold", type=float, default=80.0, help="Weighted threshold (0-100)")
    parser.add_argument("--top-k", type=int, default=1, help="Top matches per input name")
    parser.add_argument("--top-candidates", type=int, default=20, help="Candidate shortlist size")
    parser.add_argument("--output", default="matches.csv", help="Output CSV filename")
    return parser.parse_args([] if "ipykernel" in sys.modules else None)


if __name__ == "__main__":
    args = parse_args()

    if args.source_a and args.source_b:
        df_a = read_csv_loose(args.source_a, args.col_a)
        df_b = read_csv_loose(args.source_b, args.col_b)
    else:
        # Backwards-compatible inline demo data
        df_a = pd.DataFrame(
            {"id": [1, 2, 3, 4], "name": ["John Smith", "Jane Doe", "Robert Johnson", "Michael Brown"]}
        )
        df_b = pd.DataFrame(
            {
                "id": [101, 102, 103, 104, 105],
                "full_name": ["Smith, John", "J. Doe", "Bob Johnson", "Mike Brown", "Unmatched Person"],
            }
        )

    fuzzy_match(
        df_a=df_a,
        df_b=df_b,
        col_a=args.col_a,
        col_b=args.col_b,
        threshold=args.threshold,
        output_file=args.output,
        top_k=args.top_k,
        top_candidates=args.top_candidates,
    )
