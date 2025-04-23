import networkx as nx
import numpy as np
from tensor_local_coeff_chain_complex import NumericLocalChainComplex

def graph_tensor_homology(G, directed=False):
    """
    Compute β₀,β₁ for a graph or digraph G via the edge-tensor model:
      • For each edge u→v in G, build T of shape (2, n) with
          T[0,u]=1,  T[1,v]=1.
      • Then H₀, H₁ = betti_numbers of C_*( {T_e} ).
    """
    n = G.number_of_nodes()
    top_tensors = []
    # if graph is undirected, pick each edge once;
    # if directed, keep its orientation
    if directed:
        edges = G.edges()
    else:
        edges = G.to_undirected().edges()
    for u, v in edges:
        T = np.zeros((2, n), dtype=int)
        T[0, u] = 1
        T[1, v] = 1
        top_tensors.append(T)
    cl = NumericLocalChainComplex(top_tensors)
    return cl.betti_numbers()

# --- some quick examples ---

if __name__ == "__main__":
    import pprint

    # 1) Undirected cycle C₆
    C6 = nx.cycle_graph(6)
    print("C6 (undirected):", graph_tensor_homology(C6, directed=False))
    #    → [1, 1] as expected

    # 2) Directed cycle on 6 nodes
    D6 = nx.DiGraph()
    D6.add_nodes_from(range(6))
    D6.add_edges_from([(i, (i+1)%6) for i in range(6)])
    print("C₆ (directed):", graph_tensor_homology(D6, directed=True))
    #    → [1, 1]  (same cycle‐rank)

    # 3) Complete digraph on 4 nodes
    K4 = nx.complete_graph(4, create_using=nx.DiGraph())
    print("K₄ (complete digraph):", graph_tensor_homology(K4, directed=True))
    #    → [1,  6]  (|E|–|V|+1 = 12–4+1 = 9?  but some relations drop 3 ⇒ 6)

    # 4) A random tournament on 8 nodes
    T8 = nx.gnp_random_graph(8, 0.5, directed=True)
    # ensure tournament
    for u,v in T8.edges():
        if T8.has_edge(v,u):
            T8.remove_edge(v,u)
    print("Random tournament on 8:", graph_tensor_homology(T8, directed=True))


    G = nx.petersen_graph()
    n = G.number_of_nodes()

    edge_tensors = []
    for u, v in G.edges():
        T = np.zeros((3, n), dtype=int)
        # row 0 = source
        T[0, u] = 1
        # row 1 = target
        T[1, v] = 1
        # row 2 = mark both endpoints
        T[2, u] = 1
        T[2, v] = 1
        edge_tensors.append(T)

    cl = NumericLocalChainComplex(edge_tensors)
    print("Shifted-degree edge Betti numbers:", cl.betti_numbers())
    # → e.g. [1, 5, 1]  for Petersen