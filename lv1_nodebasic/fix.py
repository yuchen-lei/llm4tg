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


from aux_querygpt import aquery_once, get_llm_model, token_count


def extract_addr(pth):
    return os.path.splitext(os.path.basename(pth))[0]


basd8_root = r"E:\Files\llm4tg\basd-8"

subgraphs = glob(basd8_root + r"\**\*.graphml", recursive=True)

df_tokens = pd.read_csv("tokens.csv")
applicable_graphset = set(df_tokens[df_tokens[".llm4tg"] < 125000]["addr"].to_list())
applicable_graphs = [x for x in subgraphs if extract_addr(x) in applicable_graphset]

applicable_graphs2 = []
for subgraph in applicable_graphs:
    addr = extract_addr(subgraph)
    G: nx.DiGraph = nx.read_graphml(subgraph)
    nodes = G.number_of_nodes()
    if nodes > 10:
        applicable_graphs2.append(subgraph)

random.seed(114514)
selected_subgraphs = random.sample(applicable_graphs2, 50)

summary = []
for subgraph in selected_subgraphs:
    addr = extract_addr(subgraph)
    G: nx.DiGraph = nx.read_graphml(subgraph)
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    tokens = df_tokens[df_tokens["addr"] == addr][".llm4tg"].values[0]
    summary.append(
        {
            "addr": addr,
            "nodes": nodes,
            "edges": edges,
            "tokens": tokens,
        }
    )
curdir = os.path.dirname(__file__)
pd.DataFrame(summary).to_csv(f"{curdir}/selected_summary.csv", index=False)

gpt4t = get_llm_model("gpt-4-turbo")
gpt4o = get_llm_model("gpt-4o")

with open(f"{curdir}/lv1_results.json", "r", encoding="ISO-8859-3") as f:
    orig_result = json.load(f)


async def main():
    results = {}
    for subgraph in tqdm(selected_subgraphs):
        addr, _type = os.path.splitext(os.path.basename(subgraph))
        assert addr in orig_result
        tqdm.write(f"{addr=}")
        G: nx.DiGraph = nx.read_graphml(subgraph)
        nodes = list(G.nodes)
        depth = nx.single_source_shortest_path_length(
            G.to_undirected(as_view=True), source="n0"
        )
        sorted_depth = {k: depth[k] for k in nodes}
        rev_dict = defaultdict(list)
        [rev_dict[v].append(k) for k, v in sorted_depth.items()]
        # selpoints = set(random.choice(l) for l in rev_dict.values())
        selpoints = orig_result[addr]["selected_nodes"]
        while len(selpoints) < 10:
            selpoints.add(random.choice(nodes))
        with open(f"repr_llm4tg/{addr}.llm4tg") as f:
            graph_repr = f.read()
        tqdm.write(f"{token_count(graph_repr)=}")
        selpoints = sorted(selpoints)
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
                    "in_value": sum(
                        [e["value"] for (u, v, e) in G.in_edges(n, data=True)]
                    ),
                    "out_value": sum(
                        [e["value"] for (u, v, e) in G.out_edges(n, data=True)]
                    ),
                }
            )
            for n in nodes
        }
        node_info = {
            n: {
                **v,
                "diff_degree": abs(v["in_degree"] - v["out_degree"]),
                "diff_value": abs(v["in_value"] - v["out_value"]),
            }
            for n, v in node_info.items()
        }

        def make_questions(n):
            if depth[n] & 1:  # transaction
                in_nodes = [f for f, _ in G.in_edges(n)]
                out_nodes = [t for _, t in G.out_edges(n)]
                should_exist = (ord(n[-1]) - ord("0")) & 1
                conn = set(in_nodes + out_nodes)
                rest = set(nodes) - conn
                chknode = random.choice(list(conn if should_exist else rest))
                return (
                    f"node {n}: answer Input/Output Degree and Amount. And check if {n} is connected with {chknode}, answer in special_info field by true/false.",
                    (chknode, should_exist),
                )
            else:  # address
                times = [
                    x for _, _, x in G.to_undirected(as_view=True).edges(n, data="time")
                ]
                return (
                    f"node {n}: answer Input/Output Degree and Amount. And find out Time range field, answer in special_info.",
                    max(times) - min(times),
                )

        questions = [make_questions(n) for n in selpoints]
        ground_truth = {
            "global": {
                f"max_{key}_node": max(node_info, key=lambda x: node_info[x][key])
                for key in [
                    "in_degree",
                    "out_degree",
                    "in_value",
                    "out_value",
                    "diff_degree",
                    "diff_value",
                ]
            },
            "nodes": {
                n: {
                    "in_degree": node_info[n]["in_degree"],
                    "out_degree": node_info[n]["out_degree"],
                    "in_value": node_info[n]["in_value"],
                    "out_value": node_info[n]["out_value"],
                    "special_info": questions[i][1],
                }
                for i, n in enumerate(selpoints)
            },
        }

        questions_prompt = "\n".join([q[0] for q in questions])

        # task4t = aquery_once(
        #     gpt4t, "query_basic", nodes=questions_prompt, graph=graph_repr
        # )
        # task4o = aquery_once(
        #     gpt4o, "query_basic", nodes=questions_prompt, graph=graph_repr
        # )
        # ans4t, ans4o = await asyncio.gather(task4t, task4o)
        ans4t = orig_result[addr]["gpt4t"]
        ans4o = orig_result[addr]["gpt4o"]
        results[addr] = {
            "selected_nodes": selpoints,
            "ground_truth": ground_truth,
            "gpt4t": ans4t,
            "gpt4o": ans4o,
        }
        with open(f"{curdir}/results_fix2.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


asyncio.run(main())
