# %%
import math
import os
import random
from glob import glob
from io import StringIO

import networkx
import numpy as np
import tiktoken
from networkx import DiGraph
from tqdm.auto import tqdm

tokenizer = tiktoken.encoding_for_model("gpt-4")

subgraph_files = glob("basd-8/**/*.graphml", recursive=True)


def summary_info_for_address(G: DiGraph, node):
    times = [x for _, _, x in G.to_undirected().edges(node, data="time")]
    return {
        "in_degree": G.in_degree(node),
        "out_degree": G.out_degree(node),
        "in_value": sum([x["value"] for _, _, x in G.in_edges(node, data=True)]),
        "out_value": sum([x["value"] for _, _, x in G.out_edges(node, data=True)]),
        "time_range": max(times) - min(times),
        # "out_nodes": set([t for _, t in G.edges(node)]),
    }


def summary_info_for_transaction(G: DiGraph, node):
    nodeview = G.nodes[node]
    return {
        "in_degree": nodeview["tx_inputs_count"],
        "out_degree": nodeview["tx_outputs_count"],
        "in_value": nodeview["tx_inputs_value"],
        "out_value": nodeview["tx_outputs_value"],
        "in_nodes": [f for f,_ in G.in_edges(node)],
        "out_nodes": [t for _, t in G.out_edges(node)],
    }


def sample_single_graph(subgraph_file):
    with open(subgraph_file, "r") as f:
        graphml_txt = f.read()
    subgraph_filename = os.path.splitext(os.path.basename(subgraph_file))[0]
    if os.path.exists(f"graph_sampled/{subgraph_filename}.txt"):
        return
    graphml_tokens = len(tokenizer.encode(graphml_txt))
    # in transaction graph, theres two types of nodes
    # 1. transaction nodes
    # 2. address nodes
    # transaction nodes have much more information
    # address nodes have only address
    # let's say we preserve most of the information of transaction nodes
    # but only caculate the in_ out_ info for address nodes
    graph: DiGraph = networkx.read_graphml(subgraph_file)
    sio = StringIO()
    dist = networkx.single_source_shortest_path_length(graph.to_undirected(), "n0")
    dist_values = list(dist.values())
    dist_count = {v: dist_values.count(v) for v in set(dist_values)}
    last_dist = -1
    # we consider that nodes with larger value are more important based on our previous work
    # so we apply weighted sampling from nodes
    node_info = {
        node: summary_info_for_transaction(graph, node)
        if (dist[node] & 1)
        else summary_info_for_address(graph, node)
        for node in graph.nodes
    }
    # importance: (log(in_value + out_value + 1) + 2 * log(in_degree + out_degree + 1)) / (dist + 1)
    node_weights = [
        (
            math.log1p(node_info[node]["in_value"] + node_info[node]["out_value"])
            + 2
            * math.log1p(node_info[node]["in_degree"] + node_info[node]["out_degree"])
        )
        / (dist[node] + 1)
        for node in graph.nodes
    ]
    node_weights = 1 / np.array(node_weights)
    node_weights[0] = 0
    node_weights = node_weights / node_weights.sum()
    removed_nodes = np.random.choice(
        list(graph.nodes()),
        # to reduce tokens into 3000, we have to keep only about 75 nodes
        # the parameter can be changed here based on the demand
        max(graph.number_of_nodes() - 60, 0),
        replace=False,
        p=node_weights,
    )
    removed_nodes = set(removed_nodes)
    node_keeps = set(graph.nodes()) - removed_nodes
    for node in list(node_keeps):
        path = networkx.shortest_path(graph.to_undirected(as_view=True), "n0", node)
        for node in path:
            node_keeps.add(node)
    removed_nodes = set(graph.nodes()) - node_keeps
    graph.remove_nodes_from(removed_nodes)
    print(graph)
    for node in graph.nodes:
        node_type = "transaction" if (dist[node] & 1) else "address"
        if last_dist != dist[node]:
            print(
                f"Layer {dist[node]}: {dist_count[dist[node]]} {node_type} nodes",
                file=sio,
            )
            last_dist = dist[node]
        print(
            f"{node} {node_type}:",
            node_info[node],
            file=sio,
        )
    # method1(graph)
    graph_repr = sio.getvalue().replace("'", "")
    # print(graph_repr)
    with open(f"graph_sampled/{subgraph_filename}.txt", "w") as f:
        f.write(graph_repr)
    print(graphml_tokens, "->", len(tokenizer.encode(graph_repr)))


from multiprocessing import Pool
import pandas as pd
def split_category_address(path):
    category, address = path.split("\\")[-2:]
    address = os.path.splitext(address)[0]
    return category, address


address_category_map = {
    split_category_address(x)[1]: split_category_address(x)[0] for x in subgraph_files
}


df = pd.read_csv("babd13-slim.csv")
df["label"] = df["account"].apply(lambda x: address_category_map[x])
df_ref=df.groupby("label").sample(n=3,random_state=42,replace=False)[["account","label"]]

test_graphs = df.sample(40, random_state=114514)
accounts=df_ref["account"].tolist()
test_accounts=test_graphs["account"].tolist()
print(accounts)
print(test_accounts)
print(len(accounts+test_accounts))
subgraph_files1 = [x for x in subgraph_files if os.path.splitext(os.path.basename(x))[0] in accounts]
subgraph_files2 = [x for x in subgraph_files if os.path.splitext(os.path.basename(x))[0] in test_accounts]
print(subgraph_files2)
print(len(subgraph_files1))
print(len(subgraph_files2))
print(len(subgraph_files1+subgraph_files2))
if __name__ == "__main__":
    with Pool(8) as p:
        p.map(sample_single_graph, tqdm(subgraph_files1+subgraph_files2))
# %%
