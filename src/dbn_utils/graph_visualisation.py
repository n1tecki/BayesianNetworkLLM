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
) -> None:
    """
    Create an interactive Pyvis visualization of a (Dynamic)BayesianNetwork,
    using a gentle “repulsion” physics setup.

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

    # 3) Set up the Pyvis Network (no physics Helper called here)
    net = Network(height="750px", width="100%", directed=True)

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

    # 6) Apply “weak repulsion” physics options
    def _apply_weak_physics(network: Network):
        """
        Use a light “repulsion” solver with a low spring constant and mild damping.
        Nodes will still repel each other gently, but won’t oscillate too violently.
        """
        opts = {
            "physics": {
                "enabled": True,
                "solver": "repulsion",
                "repulsion": {
                    "nodeDistance": 100,      # distance at which repulsion pushes
                    "springLength": 100,      # rest length of springs
                    "springConstant": 0.002,  # very weak pull-back force
                    "damping": 0.08,          # mild friction to damp oscillations
                },
                "stabilization": {
                    "enabled": False         # turn off automatic stabilization
                },
            },
            "edges": {
                "arrows": {
                    "to": {"enabled": True}
                }
            },
            "layout": {
                "hierarchical": {
                    "enabled": False
                }
            },
        }
        network.set_options(json.dumps(opts))

    # 7) Disable any built-in physics helpers, then apply gentle “repulsion”
    net.toggle_physics(False)
    _apply_weak_physics(net)

    # 8) Save or show
    if notebook:
        net.show(html_file)
    else:
        net.save_graph(html_file)
        print(f"Interactive network saved to {html_file}")



from typing import Union
import networkx as nx
from pyvis.network import Network
from pgmpy.models import BayesianNetwork, DynamicBayesianNetwork


def network_slice_visualisations(
    model: Union[BayesianNetwork, DynamicBayesianNetwork],
    base_html_file: str = "dbn_slice",
    slice_colors: tuple[str, str] = ("#97C2FC", "#FB7E81"),
) -> None:
    """
    Write three interactive HTML graphs, with physics entirely turned off 
    so nodes can be dragged and will remain where you drop them:

      <base>_t0.html    – only time-slice-0 nodes and edges
      <base>_t1.html    – only time-slice-1 nodes and edges
      <base>_inter.html – only edges that cross t=0 ➜ t=1, with the two
                          slices laid out on parallel rows.
    """

    # 1) build the raw DiGraph from your model’s edges
    G = nx.DiGraph(model.edges())

    # 2) helper to extract “label” and “time_slice” from each node
    def _label(node):
        if hasattr(node, "node") and hasattr(node, "time_slice"):
            return f"{node.node} (t={node.time_slice})", node.time_slice
        if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], int):
            return f"{node[0]} (t={node[1]})", node[1]
        return str(node), None

    id_map: dict[object, str] = {}
    slice0: list[object] = []
    slice1: list[object] = []

    for n in G.nodes():
        label, ts = _label(n)
        id_map[n] = label
        if ts == 0:
            slice0.append(n)
        elif ts == 1:
            slice1.append(n)

    # 3) Build a fresh Pyvis Network **without** invoking any physics helper.
    #    We’ll disable physics explicitly in the JSON options.
    def _build_net() -> Network:
        net = Network(height="750px", width="100%", directed=True)
        return net

    # 4) Every graph (t0 and t1) will have physics turned OFF in its options.
    def _apply_json_options_gentle(net: Network):
        opts = {
            "physics": {
                "enabled": True,
                "solver": "repulsion",
                "repulsion": {
                    "nodeDistance": 70,        # spacing between nodes
                    "springLength": 70,        # rest length of springs
                    "springConstant": 0.002,    # how weak the “pull back” is
                    "damping": 0.04,            # some friction so it settles
                },
                "stabilization": {"enabled": False},
            },
            "edges": {"arrows": {"to": {"enabled": True}}},
            "layout": {"hierarchical": {"enabled": False}},
        }
        net.set_options(json.dumps(opts))

    # 5a) Build and save t=0 slice with physics disabled
    net0 = _build_net()
    for n in slice0:
        net0.add_node(id_map[n], label=id_map[n], color=slice_colors[0])
    for u, v in G.edges(slice0):
        if v in slice0:
            net0.add_edge(id_map[u], id_map[v], arrows="to")

    # Disable physics _after_ all nodes+edges are in place,
    # just to be extra sure nothing re-enables it:
    net0.toggle_physics(False)
    _apply_json_options_gentle(net0)
    net0.save_graph(f"{base_html_file}_t0.html")
    print(f"Saved → {base_html_file}_t0.html")

    # 5b) Build and save t=1 slice with physics disabled
    net1 = _build_net()
    for n in slice1:
        net1.add_node(id_map[n], label=id_map[n], color=slice_colors[1])
    for u, v in G.edges(slice1):
        if v in slice1:
            net1.add_edge(id_map[u], id_map[v], arrows="to")

    net1.toggle_physics(False)
    _apply_json_options_gentle(net1)
    net1.save_graph(f"{base_html_file}_t1.html")
    print(f"Saved → {base_html_file}_t1.html")

    # 5c) Inter-slice view: fixed positions (physics off by design)
    net_inter = Network(height="750px", width="100%", directed=True)
    net_inter.toggle_physics(False)  # no physics, nodes will be permanently fixed

    h_gap, v_gap = 120, 100
    for idx, n in enumerate(sorted(slice0, key=id_map.get)):
        net_inter.add_node(
            id_map[n],
            label=id_map[n],
            color=slice_colors[0],
            x=idx * h_gap,
            y= v_gap,
            fixed={"x": True, "y": True},
        )
    for idx, n in enumerate(sorted(slice1, key=id_map.get)):
        net_inter.add_node(
            id_map[n],
            label=id_map[n],
            color=slice_colors[1],
            x=idx * h_gap,
            y=-v_gap,
            fixed={"x": True, "y": True},
        )

    for u, v in G.edges():
        if (u in slice0 and v in slice1) or (u in slice1 and v in slice0):
            net_inter.add_edge(id_map[u], id_map[v], arrows="to")

    # no need to call _apply_json_options here since physics is already off
    net_inter.save_graph(f"{base_html_file}_inter.html")
    print(f"Saved → {base_html_file}_inter.html")