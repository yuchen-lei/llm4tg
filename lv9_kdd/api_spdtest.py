import json
import os
import pickle
import sys
from glob import glob

import pandas as pd
from tqdm.auto import tqdm

sys.path.append(".")

from aux_querygpt import aquery_once, get_llm_model, instantiate_prompt

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

curdir = os.path.dirname(__file__)

meaningful_columns = [
    "label",
    "S2-2",
    "S1-6",
    "S1-2",
    "S3",
    "PAIa21-1",
    "PTIa41-2",
    "S6",
    "S5",
    "CI3a32-2",
    "S7",
]


def main_feature():
    df = pd.read_csv("babd13-slim.csv")
    df["label"] = df["account"].apply(lambda x: address_category_map[x])
    df_ref = df.groupby("label").sample(n=5, random_state=514, replace=False)
    ref_graphs = "\n".join(
        [
            f"graph {idx+1:02d}: {json.dumps(val)}"
            for idx, val in enumerate(
                df_ref[meaningful_columns].to_dict(orient="records")
            )
        ]
    )

    df_rest = df[~df["account"].isin(df_ref["account"])]
    test_graphs_p1 = df_rest.groupby("label").sample(
        n=8, random_state=114514, replace=False
    )
    test_graphs_p2 = df_rest[
        ~df_rest["account"].isin(test_graphs_p1["account"])
    ].sample(n=500 - len(test_graphs_p1), random_state=1919810, replace=False)
    test_graphs = pd.concat([test_graphs_p1, test_graphs_p2])
    llm = get_llm_model("gpt-4o-mini")
    results = {}
    sem = asyncio.Semaphore(8)

    async def single_graph(account, label, current_graph_repr):
        async with sem:
            result = await aquery_once(
                llm,
                "categorize_by_features",
                features=ref_graphs,
                graph=current_graph_repr,
            )
            results[account] = {
                "ground_truth": label,
                "gpt35_feature": {
                    "result": result["result"],
                    "explain": result["reason"],
                },
            }

    from langchain_core.messages import convert_to_openai_messages
    from openai import OpenAI

    client = OpenAI()
    results = {}
    for _, current_graph in test_graphs.sample(10).iterrows():
        account = current_graph["account"]
        current_graph_repr = current_graph[meaningful_columns[1:]].to_dict()
        # tasks.append(single_graph(account, current_graph["label"], current_graph_repr))
        prompt = instantiate_prompt(
            "categorize_by_features",
            features=ref_graphs,
            graph=current_graph_repr,
        )
        msg = convert_to_openai_messages(prompt)
        from pprint import pprint

        pprint(convert_to_openai_messages(prompt))

        import time

        def first_token(model):
            st = time.time()
            res = client.chat.completions.create(
                model=model,
                messages=msg,
                max_completion_tokens=1,
            )
            ed = time.time()
            # return ed - st, res.usage.model_dump()
            return {
                "time": ed - st,
                "content": res.choices[0].message.content,
                "usage": res.usage.model_dump(),
            }

        def total_time(model):
            st = time.time()
            res = client.chat.completions.create(
                model=model,
                messages=msg,
            )
            ed = time.time()
            return {
                "time": ed - st,
                "content": res.choices[0].message.content,
                "usage": res.usage.model_dump(),
            }

        results[account] = {}
        for models in ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]:
            results[account][models] = {
                "first_token": first_token(models),
                "total_time": total_time(models),
            }
        with open(f"{curdir}/result_api_spd_feature.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def main_graph():
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
    # prev_results = joblib.load("../llm4tg/4-categorize-by-graph-gpt4-p3.pkl")

    # test_graphs = df[df["account"].isin(prev_results.keys())].drop_duplicates("account")
    test_graphs_p1 = df_rest.groupby("label").sample(
        n=8, random_state=114514, replace=False
    )
    test_graphs_p2 = df_rest[
        ~df_rest["account"].isin(test_graphs_p1["account"])
    ].sample(n=500 - len(test_graphs_p1), random_state=1919810, replace=False)
    test_graphs = pd.concat([test_graphs_p1, test_graphs_p2])
    # print(df_ref[df_ref["account"].isin(prev_results.keys())])
    # return
    gpt4o = get_llm_model("gpt-4o")

    results = {}
    # if os.path.exists(f"{curdir}/result_gpt4o_graph.json"):
    #     with open(f"{curdir}/result_gpt4o_graph.json", "r") as f:
    #         results = json.load(f)

    sem = asyncio.Semaphore(4)

    async def single_graph(account, label, current_graph_repr):
        async with sem:
            if account not in results:
                result = await aquery_once(
                    gpt4o,
                    "categorize_by_graph",
                    graph_ref=ref_graphs,
                    graph=current_graph_repr,
                )
                results[account] = {
                    "ground_truth": label,
                    "gpt4o_graph": {
                        "result": result["result"],
                        "explain": result["reason"],
                    },
                }
            with open(f"{curdir}/result_gpt4o_graph.json", "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    from langchain_core.messages import convert_to_openai_messages
    from openai import OpenAI

    client = OpenAI()
    results = {}
    for _, current_graph in test_graphs.sample(10).iterrows():
        account = current_graph["account"]
        with open(f"graph_sampled/{account}.txt", "r") as f:
            current_graph_repr = f.read()

        prompt = instantiate_prompt(
            "categorize_by_graph",
            graph_ref=ref_graphs,
            graph=current_graph_repr,
        )
        msg = convert_to_openai_messages(prompt)
        from pprint import pprint

        pprint(convert_to_openai_messages(prompt))

        import time

        def first_token(model):
            st = time.time()
            res = client.chat.completions.create(
                model=model,
                messages=msg,
                max_completion_tokens=1,
            )
            ed = time.time()
            return {
                "time": ed - st,
                "content": res.choices[0].message.content,
                "usage": res.usage.model_dump(),
            }

        def total_time(model):
            st = time.time()
            res = client.chat.completions.create(
                model=model,
                messages=msg,
            )
            ed = time.time()
            return {
                "time": ed - st,
                "content": res.choices[0].message.content,
                "usage": res.usage.model_dump(),
            }

        results[account] = {}
        for models in ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]:
            results[account][models] = {
                "first_token": first_token(models),
                "total_time": total_time(models),
            }
        with open(f"{curdir}/result_api_spd_graph.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import threading

    t1 = threading.Thread(target=main_feature)
    t2 = threading.Thread(target=main_graph)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
