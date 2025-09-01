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

meaningful_columns = [
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
    "label",
]
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
    df_ref = (
        df.groupby("label")
        .sample(n=5)[["label"] + meaningful_columns]
        .to_dict(orient="records")
    )
    ref_graphs = "\n".join(
        [f"graph {idx+1:02d}: {json.dumps(val)}" for idx, val in enumerate(df_ref)]
    )
    # print(ref_graphs)
    test_graphs = df.sample(500, random_state=42)
    llm = aux_querygpt.get_llm_model("gpt-4-1106-preview")
    results = {}
    for _, current_graph in tqdm(test_graphs.iterrows(), total=len(test_graphs)):
        account = current_graph["account"]
        current_graph_repr = current_graph[meaningful_columns].to_dict()
        result = await aux_querygpt.categorize_by_features(
            ref_graphs,
            current_graph_repr,
            llm,
        )
        results[account] = {
            "ground_truth": current_graph["label"],
            "llm_result": result["result"],
            "llm_explain": result["reason"],
        }
    with open("4-categorize-by-table-gpt4.pkl", "wb") as f:
        pickle.dump(results, f)


await main()

# %%
