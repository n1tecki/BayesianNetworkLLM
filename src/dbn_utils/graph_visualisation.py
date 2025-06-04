from typing import Union
import json
import networkx as nx
from pyvis.network import Network
from pgmpy.models import BayesianNetwork, DynamicBayesianNetwork


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Single-view visualiser
# ──────────────────────────────────────────────────────────────────────────────
def network_visualisation(
    model: Union[BayesianNetwork, DynamicBayesianNetwork],
    html_file: str = "bayesian_network.html",
    notebook: bool = False,
    slice_colors: tuple = ("#97C2FC", "#FB7E81"),
) -> None:
    """
    Create an interactive Pyvis visualisation of a (Dynamic)BayesianNetwork.

    • Time slice 0 nodes are shown as “(t)”        (colour slice_colors[0])
    • Time slice 1 nodes are shown as “(t+1)”      (colour slice_colors[1])
    """

    # 1) Build a NetworkX view of the structure
    G = nx.DiGraph(model.edges())

    # 2) Map every pgmpy node → visible-ID string
    id_map = {}
    for n in G.nodes():
        # DynamicBayesianNetwork nodes --------------------------
        if hasattr(n, "node") and hasattr(n, "time_slice"):
            ts, base = n.time_slice, n.node
        # Tuple-style DBN nodes (('X', 0) / ('X', 1)) -----------
        elif isinstance(n, tuple) and len(n) == 2 and isinstance(n[1], int):
            ts, base = n[1], n[0]
        # Static BN nodes ---------------------------------------
        else:
            id_map[n] = str(n)
            continue

        # Human-friendly labels
        if ts == 0:
            vid = f"{base} (t)"
        elif ts == 1:
            vid = f"{base} (t+1)"
        else:                          # any other slice number
            vid = f"{base} (t={ts})"
        id_map[n] = vid

    # 3) Set up the Pyvis network
    net = Network(height="750px", width="100%", directed=True)

    # 4) Add nodes with colours / labels
    for n in G.nodes():
        label = id_map[n]
        # default colour for static-only BNs
        color = "#ADCBE3"
        if label.endswith("(t)"):
            color = slice_colors[0]
        elif label.endswith("(t+1)"):
            color = slice_colors[1]
        net.add_node(label, label=label, title=label, color=color)

    # 5) Add edges
    for u, v in G.edges():
        net.add_edge(id_map[u], id_map[v], arrows="to")

    # 6) Gentle repulsion physics
    def _apply_weak_physics(network: Network):
        opts = {
            "physics": {
                "enabled": True,
                "solver": "repulsion",
                "repulsion": {
                    "nodeDistance": 100,
                    "springLength": 100,
                    "springConstant": 0.002,
                    "damping": 0.08,
                },
                "stabilization": {"enabled": False},
            },
            "edges": {"arrows": {"to": {"enabled": True}}},
        }
        network.set_options(json.dumps(opts))

    net.toggle_physics(False)
    _apply_weak_physics(net)

    # 7) Save / show
    if notebook:
        net.show(html_file)
    else:
        net.save_graph(html_file)
        print(f"Interactive network saved to {html_file}")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Three-view “slice” visualiser
# ──────────────────────────────────────────────────────────────────────────────
def network_slice_visualisations(
    model: Union[BayesianNetwork, DynamicBayesianNetwork],
    base_html_file: str = "dbn_slice",
    slice_colors: tuple[str, str] = ("#97C2FC", "#FB7E81"),
) -> None:
    """
    Produce three HTML files:

      <base>_t0.html    – time-slice-0 nodes labelled “… (t)”
      <base>_t1.html    – time-slice-1 nodes labelled “… (t+1)”
      <base>_inter.html – only edges that cross t ➜ t+1, with rows for each slice
    """

    G = nx.DiGraph(model.edges())

    # helper: stringify node & extract slice index
    def _label(node):
        if hasattr(node, "node") and hasattr(node, "time_slice"):
            ts, base = node.time_slice, node.node
        elif isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], int):
            ts, base = node[1], node[0]
        else:                                   # static BN node
            return str(node), None

        if ts == 0:
            return f"{base} (t)", 0
        if ts == 1:
            return f"{base} (t+1)", 1
        return f"{base} (t={ts})", ts

    id_map = {}
    slice0, slice1 = [], []
    for n in G.nodes():
        label, ts = _label(n)
        id_map[n] = label
        if ts == 0:
            slice0.append(n)
        elif ts == 1:
            slice1.append(n)

    # build a blank (physics-off) Network instance
    def _build_net() -> Network:
        net = Network(height="750px", width="100%", directed=True)
        return net

    # apply gentle options (still physics-off for these slice views)
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

    # ---------------  slice t  -----------------
    net0 = _build_net()
    for n in slice0:
        net0.add_node(id_map[n], label=id_map[n], color=slice_colors[0])
    for u, v in G.edges(slice0):
        if v in slice0:
            net0.add_edge(id_map[u], id_map[v], arrows="to")
    _apply_json_options_gentle(net0)
    net0.save_graph(f"{base_html_file}_t0.html")
    print(f"Saved → {base_html_file}_t0.html")

    # ---------------  slice t+1  -----------------
    net1 = _build_net()
    for n in slice1:
        net1.add_node(id_map[n], label=id_map[n], color=slice_colors[1])
    for u, v in G.edges(slice1):
        if v in slice1:
            net1.add_edge(id_map[u], id_map[v], arrows="to")
    _apply_json_options_gentle(net1)
    net1.save_graph(f"{base_html_file}_t1.html")
    print(f"Saved → {base_html_file}_t1.html")



    # ---------------  inter-slice view  -----------------
    net_inter = Network(height="750px", width="100%", directed=True)
    net_inter.toggle_physics(False)          # fixed positions, no physics

    h_gap, v_gap = 200, 100

    # lower row  (t) ─ labels stay *below* the nodes (PyVis default)
    for idx, n in enumerate(sorted(slice0, key=id_map.get)):
        net_inter.add_node(
            id_map[n],
            label=id_map[n],
            color=slice_colors[0],
            x=idx * h_gap,
            y=+v_gap,
            fixed={"x": True, "y": True},
            # default label placement → below the node, so no font “vadjust” needed
        )

    # upper row  (t+1) ─ labels go *above* the nodes
    for idx, n in enumerate(sorted(slice1, key=id_map.get)):
        net_inter.add_node(
            id_map[n],
            label=id_map[n],
            color=slice_colors[1],
            x=idx * h_gap,
            y=-v_gap,
            fixed={"x": True, "y": True},
            font={"vadjust": -80},   # negative shift moves label upward
        )

    # edges that cross slices (t → t+1 or t+1 → t)
    for u, v in G.edges():
        if (u in slice0 and v in slice1) or (u in slice1 and v in slice0):
            net_inter.add_edge(id_map[u], id_map[v], arrows="to")

    net_inter.save_graph(f"{base_html_file}_inter.html")
    print(f"Saved → {base_html_file}_inter.html")

