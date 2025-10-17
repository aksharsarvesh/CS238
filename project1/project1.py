from math import lgamma
from itertools import product
import pandas as pd
import random

def state_names(df: pd.DataFrame):
    
    state_names = {}
    for col in df.columns:
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

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"[save_g] Saved {len(lines)} edges to {path}")

def local_k2_score(data, child: str, parents: list[str], state_names: dict[str, list]):
    
    log_score = 0.0
    r_i = len(state_names[child])
    alpha_ijk = 1 

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

def recompute_k2(data, g, parent, child, state_names, score_components) -> float:
    old = score_components[child]
    new = local_k2_score(data, child, g[child] + [parent], state_names)
    return new - old

def k2_score(df_cat: pd.DataFrame, g: dict[str, list[str]], state_names: dict[str, list], score_components: dict[str, float]) -> float:
    out = 0
    for child in g:
        score_components[child] = local_k2_score(df_cat, child, g[child], state_names)
        out += score_components[child]
    return out

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
    state_names = state_names(data)
    data = ensure_categorical(data, state_names)

    with open(f'{case}.gph', 'r') as f:
        g = build_graph_from_edges(f.read(), data)

    vars_list = list(data.columns)
    idx = {v: i for i, v in enumerate(vars_list)}
    n = len(vars_list)
    canReach = [[False]*n for _ in range(n)]  

    for child, parents in g.items():
        c = idx[child]
        for parent in parents:
            p = idx[parent]
            canReach[p][c] = True

    for k in range(n):
        for i in range(n):
            if canReach[i][k]:
                row_i = canReach[i]
                row_k = canReach[k]
                for j in range(n):
                    if row_k[j]:
                        row_i[j] = True

    def would_cycle(parent: str, child: str) -> bool:
        u, v = idx[parent], idx[child]
        if u == v:
            return True
        return canReach[v][u]  

    def commit_edge_and_update(parent: str, child: str):
        u, v = idx[parent], idx[child]
        # ancestors of parent 
        anc = [a for a in range(n) if a == u or canReach[a][u]]
        # descendants of child 
        desc = [d for d in range(n) if d == v or canReach[v][d]]
        for a in anc:
            row = canReach[a]
            for d in desc:
                row[d] = True
        g[child].append(parent)

    best_path = f'{case}.gph'
    score_components = {}
    best_score = k2_score(data, g, state_names, score_components)
    print(best_score)

    score = best_score

    children_best_delta  = {c: 0.0  for c in vars_list}   
    children_best_parent = {c: None for c in vars_list}   

    recompute = set(vars_list)

    while True:
        best_edge = None
        best_edge_score = 0.0
        
        for child in vars_list:
            if child in recompute:
                local_best_delta = 0.0
                local_best_parent = None

                for parent in vars_list:
                    if parent == child or parent in g[child] or would_cycle(parent, child):
                        continue
                    
                    candidate = recompute_k2(data, g, parent, child, state_names, score_components)
                    if candidate > local_best_delta:
                        local_best_delta = candidate
                        local_best_parent = parent

                children_best_delta[child]  = local_best_delta
                children_best_parent[child] = local_best_parent

                print(f'Scanned child {child}. Best edge delta: {children_best_delta[child]:.6f}')
            
            # pick global best using cached (or just-updated) values
            if children_best_delta[child] > best_edge_score:
                best_edge_score = children_best_delta[child]
                best_parent = children_best_parent[child]
                best_edge = (best_parent, child) if best_parent is not None else None


        if best_edge is not None and best_edge_score > 0:
            p, c = best_edge

            if would_cycle(p, c):
                # This can happen if constraints changed between compute and commit.
                # Invalidate this child's cached choice and recompute next loop.
                children_best_delta[c] = 0.0
                children_best_parent[c] = None
                # Recompute only this child next iteration  
            else:
                commit_edge_and_update(p, c)
                score_components[c] += best_edge_score
                score += best_edge_score
                print(f'New Best Score: {score}')
                save_graph(g, 'large_cache.gph')
            recompute = {c}

        else:
            break

    print(f'Final best score: {score}')
    if score > best_score:
        save_graph(g, best_path)


        
        
if __name__ == "__main__":
    cases = ['small', 'medium', 'large']
    for case in cases:
        k2_search(case)
    # data = pd.read_csv("data/medium.csv")
    # file = open('medium.gph', 'r')
    # g = build_graph_from_edges(file.read(), data)
   
    # state_names = infer_state_names(data)
    # data = ensure_categorical(data, state_names)
    # print(g)
    # score_components = dict()
    # score = k2_score(data, g, state_names, score_components)
    # print(score)
    

    
