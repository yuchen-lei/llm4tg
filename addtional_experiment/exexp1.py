import os
import random
import sys
from glob import glob

import networkx as nx
import pandas as pd

sys.path.append(".")

from aux_querygpt import get_llm_model, query_once, aquery_once


def get_addr(graph_pth):
    addr, _type = os.path.splitext(os.path.basename(graph_pth))
    return addr


sampled_graphs = glob("sampled_100/*.llm4tg", recursive=True)
sampled_graphs = {
    get_addr(sampled_graph): os.path.abspath(sampled_graph)
    for sampled_graph in sampled_graphs
}

basd8_root = "../llm4tg/basd-8"
subgraphs = glob(basd8_root + r"/**/*.graphml", recursive=True)

df = pd.DataFrame(
    [
        {
            "address": os.path.splitext(os.path.basename(subgraph))[0],
            "label": os.path.basename(os.path.dirname(subgraph)),
        }
        for subgraph in subgraphs
    ]
)
print(df)

curdir = os.path.dirname(__file__)
print(curdir)

df.to_csv(f"{curdir}/basd8.csv", index=False)

df_ref = df.groupby("label").sample(n=2, random_state=114514)

df_ref.to_csv(f"{curdir}/basd8_ref.csv")

ref_graph_builder = []
for idx, current_graph in df_ref.iterrows():
    account = current_graph["address"]
    ref_graph_builder.append(f"graph {idx}:")
    with open(sampled_graphs[account], "r") as f:
        ref_graph_builder.append(f.read())
    ref_graph_builder.append("")
ref_graphs = "\n".join(ref_graph_builder)

rest_graphs = set(df["address"]) - set(df_ref["address"])

random.seed(114514)

test_graphs = random.sample(list(rest_graphs), 40)

llm4t = get_llm_model("gpt-4-turbo")
llm4o = get_llm_model("gpt-4o")

from tqdm.auto import tqdm
import json

results = {}


async def main():
    for test_graph in tqdm(test_graphs):
        with open(sampled_graphs[test_graph], "r") as f:
            test_graph_repr = f.read()
        res4t = aquery_once(
            llm4t, "query_graph_raw", graph_ref=ref_graphs, graph=test_graph_repr
        )
        res4o = aquery_once(
            llm4o, "query_graph_raw", graph_ref=ref_graphs, graph=test_graph_repr
        )
        res4t, res4o = await asyncio.gather(res4t, res4o)
        results[test_graph] = {
            "gpt-4-turbo": res4t["result"],
            "gpt-4o": res4o["result"],
        }
        with open(f"{curdir}/results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
