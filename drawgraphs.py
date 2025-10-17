import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from project1.project1 import build_graph_from_edges

def load_graph(path):
    
    graph_file = open(f'{path}.gph', 'r')
    data = pd.read_csv(f'data/{path}.csv')
    
    graph_data = graph_file.read()
    gDict = build_graph_from_edges(graph_data, data)
    
    G = nx.DiGraph()
    G.add_nodes_from(gDict.keys())
    
    for child in gDict.keys():
        for parent in gDict[child]:
            G.add_edge(parent, child)
    
    return G

def draw_graph(path):
    G = load_graph(path)
    nx.write_latex(G, f'{path}.tex', caption=f'{path}', as_document=True)
    # nx.draw_circular(G, with_labels=True, font_weight='bold')
    # plt.show()
    
if __name__ == "__main__":
    cases = ['small', 'medium','large']
    for case in cases:
        draw_graph(case)
    
