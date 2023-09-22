import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from constants import DATA_SUBSETS, RANDOM_SEED, LABEL_FEATURE, ROGUE_WAVE_THRESHOLD


def read_files(infiles):
    """Read all parquet input files into a single pandas DataFrame."""
    if isinstance(infiles, str):
        infiles = [infiles]

    df = []

    for f in tqdm(infiles):
        dfi = pd.read_parquet(f)
        df.append(dfi)

    df = pd.concat(df, copy=False)
    df.rename_axis(("aggregate_id_local", "meta_station_name"), inplace=True)
    return df


def drop_invalid(df, target_features):
    mask = np.all(np.isfinite(df[target_features]), axis=1)
    return df[mask]


def convert_log_features(df):
    for feature in df.columns:
        if not feature.endswith("_log10"):
            continue

        feature_stripped = feature[: -len("_log10")]
        df[feature_stripped] = 10 ** df[feature]

    for feature in df.columns:
        if not feature.endswith("directionality_index"):
            continue

        df[f"{feature}_log10"] = np.log10(df[feature])

    return df


def apply_constraints(df, constraints):
    """Remove all data violating constraints."""
    n_rows = df.shape[0]
    mask = np.ones(n_rows, dtype="bool")

    for feature, lower, upper in constraints:
        if lower is not None:
            mask &= df[feature] >= lower

        if upper is not None:
            mask &= df[feature] <= upper

    return df[mask]


def generate_subsets(df, features):
    out = {}

    for name, spec in DATA_SUBSETS.items():
        spec_constraints = [
            (key, *val) for key, val in spec.get("constraints", {}).items()
        ]
        df_sub = apply_constraints(df, spec_constraints)

        spec_stations = spec.get("stations", [])
        if spec_stations:
            allowed_stations = [f"CDIP_{s}p1" for s in spec_stations]
            station_mask = df.index.isin(allowed_stations, "meta_station_name")
            df_sub = df_sub[station_mask]

        out[name] = get_model_inputs(df_sub, features)

    return out


def train_test_split(df, chunk_length=1000, train_ratio=0.66):
    np.random.seed(RANDOM_SEED)

    overhang = len(df) % chunk_length
    df_idx = np.arange(len(df) - overhang).reshape(-1, chunk_length)
    np.random.shuffle(df_idx)
    df_idx = df_idx.flatten()

    train_cutoff = int(len(df_idx) * train_ratio)
    df_train = df.iloc[df_idx[:train_cutoff]]
    df_val = df.iloc[df_idx[train_cutoff:]]

    return df_train, df_val


def get_model_inputs(df, target_features):
    x = df[target_features]
    y = df[LABEL_FEATURE] > ROGUE_WAVE_THRESHOLD
    return x, y
