# %%
import importlib
import json
import os
import pickle
from glob import glob

import pandas as pd
from tqdm.auto import tqdm

import aux_querygpt

importlib.reload(aux_querygpt)

subgraph_files = glob("basd-8/**/*.graphml", recursive=True)


def split_category_address(path):
    category, address = path.split("\\")[-2:]
    address = os.path.splitext(address)[0]
    return category, address


address_category_map = {
    split_category_address(x)[1]: split_category_address(x)[0] for x in subgraph_files
}


async def main():
    df = pd.read_csv("babd13-slim.csv")
    # select each 10 from every category by column 'label'
    df["label"] = df["account"].apply(lambda x: address_category_map[x])
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
    print(ref_graphs)
    test_graphs = df.groupby("label").sample(n=8, replace=False)
    llm = aux_querygpt.get_llm_model("gpt-4-1106-preview")
    results = {}
    for _, current_graph in tqdm(test_graphs.iterrows(), total=len(test_graphs)):
        account = current_graph["account"]
        with open(f"graph_sampled/{account}.txt", "r") as f:
            current_graph_repr = f.read()
        result = await aux_querygpt.categorize_by_graph(
            ref_graphs,
            current_graph_repr,
            llm,
        )
        results[account] = {
            "ground_truth": current_graph["label"],
            "llm_result": result["result"],
            "llm_explain": result["reason"],
        }
        with open("4-categorize-by-graph-gpt4-p3.pkl", "wb") as f:
            pickle.dump(results, f)
        with open("4-categorize-by-graph-gpt4-p3.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import asyncio

    # asyncio.run(main())

# %%
import pickle
import pandas as pd

p1 = pickle.load(open("4-categorize-by-graph-gpt4.pkl", "rb"))
p2 = pickle.load(open("4-categorize-by-graph-gpt4-p2.pkl", "rb"))
p3 = {**p1, **p2}
print(len(p1), len(p2), len(p3))
for p in [p1, p2, p3]:
    print(f"--------")
    df = pd.DataFrame(
        [
            {
                "address": k,
                "label": address_category_map[k],
            }
            for k in p.keys()
        ]
    )
    for grp in df.groupby("label"):
        print(grp[0],len(grp[1]))
# %%
