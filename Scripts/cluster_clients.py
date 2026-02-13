#!/usr/bin/env python3
"""Cluster clients by KYC / geo demographic attributes from a CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


KYC_GEO_KEYWORDS = (
    "kyc",
    "geo",
    "country",
    "city",
    "region",
    "state",
    "postal",
    "zip",
    "age",
    "income",
    "occupation",
    "industry",
    "nationality",
    "residence",
    "risk",
    "pep",
    "source",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster clients based on KYC/geo demographic attributes."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV.")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output CSV with cluster labels. Default: <input>_clustered.csv",
    )
    parser.add_argument(
        "--profile-output",
        default=None,
        help="Path to cluster profile CSV. Default: <input>_cluster_profiles.csv",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Comma-separated list of feature columns to use for clustering.",
    )
    parser.add_argument(
        "--categorical-cols",
        default=None,
        help="Comma-separated categorical columns. If omitted, inferred from dtypes.",
    )
    parser.add_argument(
        "--numeric-cols",
        default=None,
        help="Comma-separated numeric columns. If omitted, inferred from dtypes.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of clusters (ignored if --auto-k is set). Default: 5.",
    )
    parser.add_argument(
        "--auto-k",
        action="store_true",
        help="Automatically choose k via silhouette score.",
    )
    parser.add_argument(
        "--min-k", type=int, default=2, help="Minimum k for --auto-k. Default: 2."
    )
    parser.add_argument(
        "--max-k", type=int, default=10, help="Maximum k for --auto-k. Default: 10."
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed. Default: 42."
    )
    return parser.parse_args()


def parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [x.strip() for x in value.split(",")]
    return [x for x in items if x]


def infer_features(df: pd.DataFrame) -> list[str]:
    inferred = [
        c
        for c in df.columns
        if any(keyword in c.lower() for keyword in KYC_GEO_KEYWORDS)
    ]
    if inferred:
        return inferred
    # Fallback: keep non-ID columns.
    return [c for c in df.columns if "id" not in c.lower()]


def validate_columns(df: pd.DataFrame, cols: Iterable[str], arg_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{arg_name}: missing columns in input CSV: {missing}")


def choose_k_silhouette(
    transformed_x,
    min_k: int,
    max_k: int,
    random_state: int,
) -> tuple[int, float]:
    best_k = min_k
    best_score = -1.0
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = model.fit_predict(transformed_x)
        score = silhouette_score(transformed_x, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, best_score


def build_cluster_profiles(
    df: pd.DataFrame, labels: pd.Series, numeric_cols: list[str], categorical_cols: list[str]
) -> pd.DataFrame:
    work = df.copy()
    work["cluster"] = labels
    n = len(work)

    size = work.groupby("cluster", dropna=False).size().rename("client_count")
    size_pct = (size / n).rename("client_share")

    profile = pd.concat([size, size_pct], axis=1)

    if numeric_cols:
        numeric_means = (
            work.groupby("cluster", dropna=False)[numeric_cols]
            .mean(numeric_only=True)
            .add_prefix("mean_")
        )
        profile = profile.join(numeric_means)

    if categorical_cols:
        # Use first mode value per cluster for categorical quick profile.
        cat_modes = {}
        for col in categorical_cols:
            mode_series = work.groupby("cluster", dropna=False)[col].agg(
                lambda s: s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else pd.NA
            )
            cat_modes[f"mode_{col}"] = mode_series
        cat_profile = pd.DataFrame(cat_modes)
        profile = profile.join(cat_profile)

    return profile.reset_index().sort_values("cluster")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_clustered.csv")
    )
    profile_output_path = (
        Path(args.profile_output)
        if args.profile_output
        else input_path.with_name(f"{input_path.stem}_cluster_profiles.csv")
    )

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    feature_cols = parse_csv_list(args.features) or infer_features(df)
    validate_columns(df, feature_cols, "--features")

    model_df = df[feature_cols].copy()
    if model_df.shape[1] == 0:
        raise ValueError("No feature columns available for clustering.")

    explicit_num = parse_csv_list(args.numeric_cols)
    explicit_cat = parse_csv_list(args.categorical_cols)

    if explicit_num is not None:
        validate_columns(model_df, explicit_num, "--numeric-cols")
        numeric_cols = explicit_num
    else:
        numeric_cols = model_df.select_dtypes(include=["number", "bool"]).columns.tolist()

    if explicit_cat is not None:
        validate_columns(model_df, explicit_cat, "--categorical-cols")
        categorical_cols = explicit_cat
    else:
        categorical_cols = [c for c in model_df.columns if c not in numeric_cols]

    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable columns for preprocessing.")

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    transformed_x = preprocess.fit_transform(model_df)
    n_rows = len(model_df)
    if n_rows < 2:
        raise ValueError("Need at least 2 rows to cluster.")

    n_clusters = args.n_clusters
    if args.auto_k:
        upper = min(args.max_k, n_rows - 1)
        lower = max(2, args.min_k)
        if upper < lower:
            raise ValueError("Invalid auto-k bounds for this dataset size.")
        n_clusters, best_score = choose_k_silhouette(
            transformed_x=transformed_x,
            min_k=lower,
            max_k=upper,
            random_state=args.random_state,
        )
        print(f"Selected k={n_clusters} via silhouette score={best_score:.4f}")
    else:
        if n_clusters < 2 or n_clusters > n_rows:
            raise ValueError(
                f"--n-clusters must be between 2 and number of rows ({n_rows})."
            )

    kmeans = KMeans(n_clusters=n_clusters, random_state=args.random_state, n_init="auto")
    labels = pd.Series(kmeans.fit_predict(transformed_x), index=df.index, name="cluster")

    result = df.copy()
    result["cluster"] = labels
    result.to_csv(output_path, index=False)

    profile = build_cluster_profiles(
        df=df,
        labels=labels,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
    profile.to_csv(profile_output_path, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Features used ({len(feature_cols)}): {feature_cols}")
    print(f"Numeric cols ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical cols ({len(categorical_cols)}): {categorical_cols}")
    print(f"Clusters: {n_clusters}")
    print(f"Wrote labeled data to: {output_path}")
    print(f"Wrote cluster profiles to: {profile_output_path}")


if __name__ == "__main__":
    main()
