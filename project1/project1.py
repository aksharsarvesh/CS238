from math import lgamma
from itertools import product
import pandas as pd
import random

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

def save_graph(g: dict[str, list[str]], path: str):
    lines = []
    for child, parents in g.items():
        for parent in parents:
            lines.append(f"{parent},{child}")

    # Write all edges to file
    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"[save_g] Saved {len(lines)} edges to {path}")

def local_k2_score(data, child: str, parents: list[str], state_names: dict[str, list]):
    
    log_score = 0.0
    r_i = len(state_names[child])
    alpha_ijk = 1  # K2 prior

    parent_state_lists = [state_names[p] for p in parents] if parents else [()]
    parent_configs = list(product(*parent_state_lists)) if parents else [()]
    for j_cfg in parent_configs:
        subset = data
        if parents:
            for p, val in zip(parents, j_cfg):
                subset = subset[subset[p] == val]
        counts = subset[child].value_counts().reindex(state_names[child], fill_value=0).astype(int)
        N_ij = int(counts.sum())
        term = lgamma(r_i * alpha_ijk) - lgamma(N_ij + r_i * alpha_ijk)
        term += sum(lgamma(n_k + alpha_ijk) - lgamma(alpha_ijk) for n_k in counts)
        log_score += term
    return float(log_score)

def recompute_k2(data, g, parent, child, state_names) -> float:
    old = local_k2_score(data, child, g[child], state_names)
    new = local_k2_score(data, child, g[child] + [parent], state_names)
    return new - old

def k2_score(df_cat: pd.DataFrame, g: dict[str, list[str]], state_names: dict[str, list]) -> float:
    return sum(local_k2_score(df_cat, child, g[child], state_names) for child in g)

def build_graph_from_edges(
    text: str,
    data: pd.DataFrame = None,
) -> dict[str, list[str]]:
    
    g: dict[str, list[str]] = {}

    if data is not None:
        g = {col: [] for col in data.columns}

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for i, line in enumerate(lines, start=1):
        parts = [p.strip() for p in line.split(",")]

        parent, child = parts
        
        if parent not in g[child]:
            g[child].append(parent)

    return g


def k2_search(case: str):
    data = pd.read_csv(f'data/{case}.csv')
    state_names = infer_state_names(data)
    data = ensure_categorical(data, state_names)
    
    initial_path = f'{case}.gph'
    file = open(initial_path, 'w+')
    g = build_graph_from_edges(file.read(), data)
    
    best_score = k2_score(data, g, state_names)
    best_path = f'{case}.gph'
    
    for seed in range(20):
        print(f'Trying seed {seed}')
        random.seed(seed)
        order = list(data.columns)
        random.shuffle(order)

        shouldContinue = True
        g = build_graph_from_edges("", data)        
        score = k2_score(data, g, state_names)                
        
        while shouldContinue:
            best_edge_score = 0
            shouldContinue = False
            best_edge = None
            for p_idx, parent in enumerate(order):
                for c_idx in range(p_idx + 1, len(order)):
                    child = order[c_idx]

                    if parent in g[child]:
                        continue

                    candidate = recompute_k2(data, g, parent, child, state_names)
                    
                    
                    if candidate > best_edge_score:
                        best_edge_score = candidate
                        best_edge = (parent, child)
                        

            if best_edge is not None and best_edge_score > 0:
                score += best_edge_score
                shouldContinue = True
                print(f'New Best Score: {score}')
                p, c = best_edge
                g[c].append(p)

        print(f'Best score this iteration: {score}')
        if(score > best_score):
            save_graph(g, best_path)
            best_score = score
        
        
# ------- Example usage -------
if __name__ == "__main__":
    # Toy dataset: all variables are discrete
    cases = ['small', 'medium', 'large']
    for case in cases:
        k2_search(case)
    # data = pd.read_csv("data/small.csv")
    # file = open('small.gph', 'r')
    # g = build_graph_from_edges(file.read(), data)
   
    # state_names = infer_state_names(data)
    # data = ensure_categorical(data, state_names)
    # print(g)
    # score = k2_score(data, g, state_names)
    # print(score)
    

    
