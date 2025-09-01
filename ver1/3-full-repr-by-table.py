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
import pandas as pd

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
import json
import aux_querygpt
import importlib

importlib.reload(aux_querygpt)
import pickle
from tqdm.auto import tqdm
import os
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
    df_ref =df.groupby("label").sample(n=5,replace=False,random_state=42)
    df_ref[["account"]+meaningful_columns].to_csv("3-full-repr-by-table-reftables.csv",index=False)
    ref_graphs = "\n".join(
        [f"graph {idx+1:02d}: {json.dumps(val)}" for idx, val in enumerate(df_ref[meaningful_columns].to_dict(orient="records"))]
    )
    test_graphs = df.sample(40, random_state=114514)
    test_graphs[["account"]+meaningful_columns].to_csv("3-full-repr-by-table-testtables.csv",index=False)
    # llm = aux_querygpt.get_llm_model("gpt-3.5-turbo-1106")
    llm = aux_querygpt.get_llm_model("gpt-4-1106-preview")
    results = {}
    for _, current_graph in tqdm(test_graphs.iterrows(), total=len(test_graphs)):
        account = current_graph["account"]
        current_graph_repr = current_graph[meaningful_columns].to_dict()
        result = await aux_querygpt.query_graph_characteristics(
            ref_graphs,
            current_graph_repr,
            llm,
        )
        results[account] = result
    with open("3-full-repr-by-table-4.0.pkl", "wb") as f:
        pickle.dump(results, f)
    with open("3-full-repr-by-table-4.0.json", "w") as f:
        json.dump(results, f, indent=2,ensure_ascii=False)


await main()
# %%
results = pickle.load(open("3-full-repr-by-table.pkl", "rb"))
import json

results = {k: json.loads(v) for k, v in results.items()}
json.dump(results, open("3-full-repr-by-table.json", "w"), indent=2)
# %%
