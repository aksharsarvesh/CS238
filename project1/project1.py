from math import lgamma
from itertools import product
import pandas as pd

def infer_state_names(df: pd.DataFrame):
    """
    Infer discrete state names for each column from the data.
    Ensures deterministic ordering.
    """
    state_names = {}
    for col in df.columns:
        # Use pandas Categorical if already set; otherwise unique values
        if isinstance(df[col].dtype, pd.api.types.CategoricalDtype):
            state_names[col] = list(df[col].cat.categories)
        else:
            state_names[col] = sorted(df[col].dropna().unique().tolist())
    return state_names

def ensure_categorical(df: pd.DataFrame, state_names: dict):
    """
    Return a copy of df with columns converted to Categorical using provided state_names.
    """
    out = df.copy()
    for col, states in state_names.items():
        out[col] = pd.Categorical(out[col], categories=states, ordered=False)
    return out

def k2_score(data, dag, state_names=None):
    if state_names is None:
        state_names = infer_state_names(data)
    df = ensure_categorical(data, state_names)
    log_score = 0.0

    for X, parents in dag.items():
        r_i = len(state_names[X])
        alpha_ijk = 1  # K2 prior

        parent_state_lists = [state_names[p] for p in parents] if parents else [()]
        parent_configs = list(product(*parent_state_lists)) if parents else [()]
        for j_cfg in parent_configs:
            subset = df
            if parents:
                for p, val in zip(parents, j_cfg):
                    subset = subset[subset[p] == val]
            counts = subset[X].value_counts().reindex(state_names[X], fill_value=0).astype(int)
            N_ij = int(counts.sum())
            term = lgamma(r_i * alpha_ijk) - lgamma(N_ij + r_i * alpha_ijk)
            term += sum(lgamma(n_k + alpha_ijk) - lgamma(alpha_ijk) for n_k in counts)
            log_score += term
    return float(log_score)

def build_dag_from_edges(
    text: str,
    data: pd.DataFrame = None,
) -> dict[str, list[str]]:
    
    g: dict[str, list[str]] = {}

    if data is not None:
        g = {col: [] for col in data.columns}

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for i, line in enumerate(lines, start=1):
        if "," in line:
            parts = [p.strip() for p in line.split(",")]
        else:
            parts = [p.strip() for p in line.split()]
        if len(parts) != 2:
            raise ValueError(f"Invalid edge format on line {i!r}: {line!r} (expected 'parent,child')")

        parent, child = parts

        if data is not None:
            cols = set(data.columns)
            if parent not in cols or child not in cols:
                raise ValueError(
                    f"Edge references unknown node(s) on line {i}: "
                    f"{parent!r}->{child!r} not found in dataset columns {sorted(cols)}"
                )
                
        g.setdefault(parent, [])
        g.setdefault(child, [])

        if parent not in g[child]:
            g[child].append(parent)

    return g

def k2_search(case: str):
    data = pd.read_csv(f'data/{case}.csv')
    best_path = f'{case}.gph'
    test_path = 'test.gph'
    file = open(test_path, 'r')
    g = build_dag_from_edges(file.read(), data)
    score = k2_score(data, g)
    print(score)
    
    
# ------- Example usage -------
if __name__ == "__main__":
    # Toy dataset: all variables are discrete
    case = 'small'
    #
    k2_search(case)
