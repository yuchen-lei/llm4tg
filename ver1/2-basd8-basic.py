# %%
from io import StringIO
import networkx
from networkx import DiGraph
from networkx.classes.coreviews import AtlasView
from glob import glob

import aux_querygpt
import importlib
import asyncio
import tiktoken

importlib.reload(aux_querygpt)


def dict_no_id(d):
    return {k: v for k, v in d.items() if k != "id"}


graph_files = glob("basd-8/**/*.graphml", recursive=True)
from tqdm.auto import tqdm
import pickle
import os

if os.path.exists("basd-8-results-gpt4.pkl"):
    results = pickle.load(open("basd-8-results-gpt4.pkl", "rb"))
else:
    results = []

progress1 = tqdm(total=len(graph_files))
progress2 = tqdm(total=len(graph_files))
progress3 = tqdm(total=len(graph_files))
progress4 = tqdm(total=len(graph_files))
sem = asyncio.Semaphore(10)
import time


async def query_single_graph(graph_file):
    async with sem:
        progress1.update()
        start_time = time.time()
        graph: DiGraph = networkx.read_graphml(graph_file)
        first_node = "n0"
        if any(x["address"] == graph.nodes[first_node]["address"] for x in results):
            progress2.update()
            return
        # first_node = next(iter(graph.nodes))
        current_result = {
            "address": graph.nodes[first_node]["address"],
            "ground_truth": {
                "in_degree": graph.in_degree(first_node),
                "out_degree": graph.out_degree(first_node),
                "in_value": sum(
                    [d["value"] for (u, v, d) in graph.in_edges(first_node, data=True)]
                ),
                "out_value": sum(
                    [d["value"] for (u, v, d) in graph.out_edges(first_node, data=True)]
                ),
                "first_time": min(
                    [
                        d["time"]
                        for (u, v, d) in graph.to_undirected().edges(
                            first_node, data=True
                        )
                    ]
                ),
            },
            "llm_result": {
                "gpt-4-1106-preview": [],
                "gpt-3.5-turbo-1106": [],
            },
        }

        view: AtlasView = graph.to_undirected().adj[first_node]
        s = StringIO()
        for key in view:
            direction = "to" if graph.get_edge_data(first_node, key) else "from"
            print(direction, key, ":", dict_no_id(view.get(key)), file=s)
        edge_repr = s.getvalue()
        s.truncate(0)
        tasks = []
        tokens = len(tiktoken.get_encoding("cl100k_base").encode(edge_repr))
        if tokens > 15000:
            progress3.set_postfix_str(f"{tokens=}")
            progress3.update()
            return
        models = ["gpt-4-1106-preview"]
        for llm_model in models:
            llm = aux_querygpt.get_llm_model(llm_model)
            tasks.append(
                asyncio.gather(
                    *[aux_querygpt.query_node_info(edge_repr, llm) for i in range(3)]
                )
            )
        llm_results = await asyncio.gather(*tasks)
        progress4.update()
        for llm_model, result in zip(models, llm_results):
            current_result["llm_result"][llm_model] = result
        s.truncate(0)
        results.append(current_result)
        pickle.dump(results, open("basd-8-results-gpt4.pkl", "wb"))
        end_time = time.time()
        await asyncio.sleep(
            (tokens + 150) * 3 * 10 / (160000 / 60) - (end_time - start_time)
        )
        return current_result


async def main():
    tasks = asyncio.gather(
        *[query_single_graph(graph_file) for graph_file in graph_files]
    )
    results = await tasks
    print(results)


await main()
# %%
results = pickle.load(open("basd-8-results.pkl", "rb"))
import json
# results = {k: json.loads(v) for k, v in results.items()}
json.dump(results, open("basd-8-results.json", "w"), indent=2)
# %%
