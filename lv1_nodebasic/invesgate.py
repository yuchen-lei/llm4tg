# %%
import asyncio
import json
import os
import random
import sys
from collections import defaultdict
from glob import glob

import networkx as nx
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(".")

basd8_root = r"E:\Files\llm4tg\basd-8"

subgraphs = glob(basd8_root + r"\**\*.graphml", recursive=True)

addr = "12grkTfkHXH2qsRzGWGGKSKyrbHwLXykfP"
node = "n206"

subgraph = [x for x in subgraphs if addr in x][0]
G: nx.DiGraph = nx.read_graphml(subgraph)
depth = nx.single_source_shortest_path_length(
    G.to_undirected(as_view=True), source="n0"
)
node_info = {
    n: (
        {
            "in_degree": G.nodes[n]["tx_inputs_count"],
            "out_degree": G.nodes[n]["tx_outputs_count"],
            "in_value": G.nodes[n]["tx_inputs_value"],
            "out_value": G.nodes[n]["tx_outputs_value"],
        }
        if (depth[n] & 1)
        else {
            "in_degree": G.in_degree(n),
            "out_degree": G.out_degree(n),
            "in_value": sum([e["value"] for (u, v, e) in G.in_edges(n, data=True)]),
            "out_value": sum([e["value"] for (u, v, e) in G.out_edges(n, data=True)]),
        }
    )
    for n in G.nodes
}
node_info = {
    n: {
        **v,
        "diff_degree": abs(v["in_degree"] - v["out_degree"]),
        "diff_value": abs(v["in_value"] - v["out_value"]),
    }
    for n, v in node_info.items()
}

print(G.nodes[node])
print(node_info[node])
# %%
G.in_edges(node, data=True)
# %%
# fix ground truth
curdir = os.path.dirname(__file__)
import cchardet

with open(f"{curdir}/lv1_results.json", "r", encoding="ISO-8859-3") as f:
    result = json.load(f)
result
# %%
