from pyvis.network import Network
import networkx as nx
import json
from typing import Union
from pgmpy.models import BayesianNetwork, DynamicBayesianNetwork

def network_visualisation(
    model: Union[BayesianNetwork, DynamicBayesianNetwork],
    html_file: str = "bayesian_network.html",
    notebook: bool = False,
    slice_colors: tuple = ("#97C2FC", "#FB7E81"),
    physics: str = "barnes_hut"
) -> None:
    """
    Create an interactive Pyvis visualization of a (Dynamic)BayesianNetwork.

    Each pgmpy node (tuple or DynamicNode) is stringified to use as the Pyvis ID.
    Time‐slice 0 vs. 1 get different colours in the DBN case.
    """
    # 1) Build a NetworkX graph of the structure
    G = nx.DiGraph(model.edges())

    # 2) Prepare a string‐ID map for every node
    id_map = {}
    for n in G.nodes():
        if hasattr(n, "node") and hasattr(n, "time_slice"):
            vid = f"{n.node} (t={n.time_slice})"
        elif isinstance(n, tuple) and len(n) == 2 and isinstance(n[1], int):
            vid = f"{n[0]} (t={n[1]})"
        else:
            vid = str(n)
        id_map[n] = vid

    # 3) Set up the Pyvis Network
    net = Network(height="750px", width="100%", directed=True)
    if physics == "barnes_hut":
        net.barnes_hut()
    else:
        net.force_atlas_2based()

    # 4) Add nodes with labels and colours
    for n in G.nodes():
        label = id_map[n]
        color = "#ADCBE3"
        if "(t=0)" in label:
            color = slice_colors[0]
        elif "(t=1)" in label:
            color = slice_colors[1]
        net.add_node(label, label=label, title=label, color=color)

    # 5) Add directed edges using the string IDs
    for u, v in G.edges():
        net.add_edge(id_map[u], id_map[v], arrows="to")

    # 6) Build and set JSON options (physics is in scope here)
    options = {
        "physics": {"solver": physics},
        "edges": {"arrows": {"to": {"enabled": True}}},
        "layout": {"hierarchical": {"enabled": False}}
    }
    net.set_options(json.dumps(options))

    # 7) Save or show
    if notebook:
        net.show(html_file)
    else:
        net.save_graph(html_file)
        print(f"Interactive network saved to {html_file}")



