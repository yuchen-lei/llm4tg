import json
import os
import sys
from glob import glob
from io import StringIO

import networkx as nx
import pandas as pd

sys.path.append(".")

from aux_querygpt import aquery_once, get_llm_model, token_count, get_azure_model


tokens = pd.read_csv("tokens.csv")

small_graphs = tokens[tokens[".gexf"] < 125000]
small_graphs_addr = small_graphs["addr"].tolist()
small_graphs_addr

basd8_root = r"E:\Files\llm4tg\basd-8"

subgraphs = glob(basd8_root + r"\**\*.graphml", recursive=True)


def extract_addr(pth):
    return os.path.splitext(os.path.basename(pth))[0]


applicable_graphs = [x for x in subgraphs if extract_addr(x) in small_graphs_addr]
applicable_graphs
applicable_graphs2 = []
for subgraph in applicable_graphs:
    addr = extract_addr(subgraph)
    G: nx.DiGraph = nx.read_graphml(subgraph)
    nodes = G.number_of_nodes()
    if nodes > 10:
        applicable_graphs2.append(subgraph)
applicable_graphs = applicable_graphs2
applicable_graphs

import asyncio
import random
from collections import defaultdict

from tqdm.auto import tqdm


async def main():
    results = {}
    for subgraph in tqdm(applicable_graphs):
        addr, _type = os.path.splitext(os.path.basename(subgraph))
        tqdm.write(f"{addr=}")
        G: nx.DiGraph = nx.read_graphml(subgraph)
        nodes = list(G.nodes)
        depth = nx.single_source_shortest_path_length(
            G.to_undirected(as_view=True), source="n0"
        )
        sorted_depth = {k: depth[k] for k in nodes}
        rev_dict = defaultdict(list)
        [rev_dict[v].append(k) for k, v in sorted_depth.items()]
        selpoints = set(random.choice(l) for l in rev_dict.values())
        while len(selpoints) < 10:
            selpoints.add(random.choice(nodes))
        with open(f"repr_llm4tg/{addr}.llm4tg") as f:
            graph_repr_llm4tg = f.read()
        graph_repr_gexf = "\n".join(nx.generate_gexf(G))
        graph_repr_gml = "\n".join(nx.generate_gml(G))
        with open(subgraph, "r") as f:
            graph_repr_graphml = f.read()
        # tqdm.write(f"{token_count(graph_repr_llm4tg)=}")
        selpoints = sorted(selpoints)
        node_info = {
            n: {
                "in_degree": G.in_degree(n),
                "out_degree": G.out_degree(n),
                "in_value": sum([e["value"] for (u, v, e) in G.in_edges(n, data=True)]),
                "out_value": sum(
                    [e["value"] for (u, v, e) in G.out_edges(n, data=True)]
                ),
            }
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
        llm = get_llm_model("gpt-4o-2024-05-13")
        # llm = get_azure_model("Llama-3.3-70B-Instruct")
        # llm = get_azure_model("DeepSeek-V3")
        task_llm4tg = aquery_once(
            llm, "query_basic", nodes=questions_prompt, graph=graph_repr_llm4tg
        )
        task_gexf = aquery_once(
            llm, "query_basic", nodes=questions_prompt, graph=graph_repr_gexf
        )
        task_gml = aquery_once(
            llm, "query_basic", nodes=questions_prompt, graph=graph_repr_gml
        )
        task_graphml = aquery_once(
            llm, "query_basic", nodes=questions_prompt, graph=graph_repr_graphml
        )
        # ans4t, ans4o = await asyncio.gather(task4t, task4o)
        result_llm4tg, result_gexf, result_gml, result_graphml = await asyncio.gather(
            task_llm4tg, task_gexf, task_gml, task_graphml
        )
        results[addr] = {
            "selected_nodes": selpoints,
            "ground_truth": ground_truth,
            "llm4tg": result_llm4tg,
            "gexf": result_gexf,
            "gml": result_gml,
            "graphml": result_graphml,
        }
        with open(f"lv9_www/exexp2_res_gpt4o.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


asyncio.run(main())
