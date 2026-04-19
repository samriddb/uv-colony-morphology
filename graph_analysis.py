import numpy as np
import networkx as nx

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False


def build_graph(df, knn_idxs, knn_dists):
    """
    weighted undirected graph - nodes are colonies, edges are knn connections
    computes degree, clustering coeff, betweenness centrality, communities
    """

    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_node(
            int(row["label"]),
            pos=(float(row["cx"]), float(row["cy"])),
            area=float(row["area_px"]),
            diam=float(row["equiv_diam"]),
        )

    labels = df["label"].values
    for i, (nbrs, dists) in enumerate(zip(knn_idxs, knn_dists)):
        u = int(labels[i])
        for j, d in zip(nbrs, dists):
            v = int(labels[j])
            if not G.has_edge(u, v) and d > 0:
                G.add_edge(u, v, weight=float(1.0 / d), dist=float(d))

    degree     = dict(G.degree())
    clustering = nx.clustering(G, weight="weight")

    try:
        between = nx.betweenness_centrality(G, weight="dist")
    except Exception:
        between = {n: 0.0 for n in G.nodes()}

    communities = None
    if HAS_LOUVAIN and len(G.nodes) > 2:
        try:
            communities = community_louvain.best_partition(G, weight="weight")
        except Exception:
            pass

    stats = {
        "nodes":        G.number_of_nodes(),
        "edges":        G.number_of_edges(),
        "avg_degree":   np.mean(list(degree.values())),
        "avg_clust":    np.mean(list(clustering.values())),
        "n_components": len(list(nx.connected_components(G))),
        "density":      nx.density(G),
    }

    return G, degree, clustering, between, communities, stats
