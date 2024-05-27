# %%
import math
import os
from glob import glob
from io import StringIO

import networkx
import numpy as np
import tiktoken
from networkx import DiGraph

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
        "in_nodes": [f for f, _ in G.in_edges(node)],
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
    # to reduce tokens into 3000, we have to keep only about 75 nodes
    # we assume that nodes with larger value are more important
    # so we apply weighted sampling from nodes
    node_info = {
        node: (
            summary_info_for_transaction(graph, node)
            if (dist[node] & 1)
            else summary_info_for_address(graph, node)
        )
        for node in graph.nodes
    }
    # importance: log(in_value + out_value + 1) / (dist + 1)
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
    print(graphml_tokens, "->", len(tokenizer.encode(graph_repr)))


def graph_full_repr(subgraph_file):
    with open(subgraph_file, "r") as f:
        graphml_txt = f.read()
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
    # to reduce tokens into 3000, we have to keep only about 75 nodes
    # we assume that nodes with larger value are more important
    # so we apply weighted sampling from nodes
    node_info = {
        node: (
            summary_info_for_transaction(graph, node)
            if (dist[node] & 1)
            else summary_info_for_address(graph, node)
        )
        for node in graph.nodes
    }
    # importance: log(in_value + out_value + 1) / (dist + 1)
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
    return graph.number_of_nodes(), graphml_tokens, len(tokenizer.encode(graph_repr))


def calc_tokens_per_node(subgraph_file):
    with open(subgraph_file, "r") as f:
        graphml_txt = f.read()
    graphml_tokens = len(tokenizer.encode(graphml_txt))
    graph: DiGraph = networkx.read_graphml(subgraph_file)
    return graph.number_of_nodes(), graphml_tokens


def main2():
    big_graph = next(
        x for x in subgraph_files if "11DzAdeW2BwxRsraZpRQQbY45msFzL4Sf" in x
    )
    big_graph: DiGraph = networkx.read_graphml(big_graph)
    print(big_graph.number_of_nodes())


def main2():
    from tqdm.auto import tqdm
    import joblib
    from joblib import Parallel, delayed

    nodes = []
    # for subgraph_file in tqdm(subgraph_files):
    #     graph: DiGraph = networkx.read_graphml(subgraph_file)
    #     nodes.append(graph.number_of_nodes())
    nodes = Parallel(n_jobs=16)(
        delayed(lambda f: (f, networkx.read_graphml(f).number_of_nodes()))(x)
        for x in tqdm(subgraph_files)
    )
    import matplotlib.pyplot as plt

    joblib.dump(nodes, "nodes.joblib")


import joblib

nodes = joblib.load("nodes.joblib")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

good_x = [x[0] for x in nodes if x[1] < 3000]
dics = []
for file in tqdm(good_x):
    nodes, graphml_tokens, llm4tg = graph_full_repr(file)
    dic = {
        "address": os.path.basename(file),
        "nodes": nodes,
        "tokens": graphml_tokens,
        "llm4tg": llm4tg,
    }
    print(dic)
    dics.append(dic)


# %%

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("nodes.csv")
plt.rcParams.update({"font.size": 15})


plt.figure(figsize=(8, 7))


df.sort_values("nodes", inplace=True)

plt.plot(df["nodes"], df["tokens"], label="GraphML", linewidth=3)

plt.plot(df["nodes"], df["llm4tg"], label="LLM4TG", linewidth=3)

max_nodes = df["nodes"].max()

plt.plot([0, max_nodes * 1.05], [128000, 128000], label="GPT-4 Limit", linewidth=2)
plt.plot([0, max_nodes * 1.05], [16284, 16384], label="GPT-3.5 Limit", linewidth=2)
# new added
plt.plot([0, max_nodes * 1.05], [8192, 8192], label="Llama 3 Limit", linewidth=2)

# plt.title('Comparison of Token and LLM4TG over Nodes')
plt.xlabel("Number of Nodes")
plt.ylabel("Token Consumption")
plt.xlim(left=0, right=max_nodes * 1.05)
plt.ylim(bottom=1)
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, pos: f"{x / 1e5:.0f}$\\times 10^5$")
)

# set ylabel as n*(10^5) n=0,1,2,3,4,5,6


plt.legend()

plt.grid(True)
plt.savefig("tokens-node.png", bbox_inches="tight")
plt.show()
# %%
