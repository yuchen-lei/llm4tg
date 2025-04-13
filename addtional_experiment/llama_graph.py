import json
import os
import pickle
import sys
from glob import glob

import pandas as pd
from tqdm.auto import tqdm

sys.path.append(".")

from aux_querygpt import aquery_once, get_llm_model, get_azure_model

subgraph_files = glob("../llm4tg/basd-8/**/*.graphml", recursive=True)


def split_category_address(path):
    category, address = path.split("\\")[-2:]
    address = os.path.splitext(address)[0]
    return category, address


address_category_map = {
    address: category
    for x in subgraph_files
    for category, address in [split_category_address(x)]
}

import asyncio
import joblib

curdir = os.path.dirname(__file__)


async def main():
    df = pd.read_csv("babd13-slim.csv")
    df["label"] = df["account"].apply(lambda x: address_category_map[x])
    # df_dup = df[df.duplicated("account")].sort_values("account")
    # print(df_dup)
    # return
    df_ref = df.groupby("label").sample(n=3, random_state=42, replace=False)[
        ["account", "label"]
    ]
    ref_graph_builder = []
    for _, current_graph in df_ref.iterrows():
        account = current_graph["account"]
        ref_graph_builder.append(f"graph of {current_graph['label']}:")
        with open(f"graph_sampled/{account}.txt", "r") as f:
            ref_graph_builder.append(f.read())
        ref_graph_builder.append("")
    ref_graphs = "\n".join(ref_graph_builder)
    df_rest = df[~df["account"].isin(df_ref["account"])]
    # test_graphs = df_rest.groupby("label").sample(
    #     n=8, random_state=114514, replace=False
    # )
    prev_results = joblib.load("../llm4tg/4-categorize-by-graph-gpt4-p3.pkl")

    test_graphs = df[df["account"].isin(prev_results.keys())].drop_duplicates("account")
    # print(df_ref[df_ref["account"].isin(prev_results.keys())])
    # return
    # llm = get_llm_model("gpt-4o")
    llm = get_azure_model("Llama-3.3-70B-Instruct")

    results = {}
    # if os.path.exists(f"{curdir}/result_gpt4o_graph.json"):
    #     with open(f"{curdir}/result_gpt4o_graph.json", "r") as f:
    #         results = json.load(f)

    sem = asyncio.Semaphore(4)
    progbar1 = tqdm(total=len(test_graphs))
    progbar2 = tqdm(total=len(test_graphs))

    async def single_graph(account, label, current_graph_repr):
        async with sem:
            progbar1.update()
            if account not in results:
                result = await aquery_once(
                    llm,
                    "categorize_by_graph",
                    graph_ref=ref_graphs,
                    graph=current_graph_repr,
                )
                results[account] = {
                    "ground_truth": label,
                    "ds_graph": {
                        "result": result["result"],
                        "explain": result["reason"],
                    },
                }
            with open(f"{curdir}/result_llama_graph.json", "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            progbar2.update()

    tasks = []
    for _, current_graph in test_graphs.iterrows():
        account = current_graph["account"]
        with open(f"graph_sampled/{account}.txt", "r") as f:
            current_graph_repr = f.read()
        tasks.append(single_graph(account, current_graph["label"], current_graph_repr))
    await asyncio.gather(*tasks)


asyncio.run(main())
