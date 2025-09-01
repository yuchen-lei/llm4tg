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
    df_ref =df.groupby("label").sample(n=3,replace=False,random_state=42)
    df_ref.to_csv("3-full-repr-by-graph-4.0-ref.csv",index=False)
    ref_graph_builder=[]
    for idx,current_graph in df_ref.iterrows():
        account = current_graph["account"]
        ref_graph_builder.append(f"graph {idx}:")
        with open(f"graph_sampled/{account}.txt","r") as f:
            ref_graph_builder.append(f.read())
        ref_graph_builder.append("")
    ref_graphs = "\n".join(ref_graph_builder)
    print(len(tiktoken.encoding_for_model("gpt-4").encode(ref_graphs)))
    test_graphs = df.sample(40, random_state=114514)
    test_graphs.to_csv("3-full-repr-by-graph-4.0-test.csv",index=False)
    # llm = aux_querygpt.get_llm_model("gpt-3.5-turbo-1106")
    llm = aux_querygpt.get_llm_model("gpt-4-1106-preview")
    results = {}
    for _, current_graph in tqdm(test_graphs.iterrows(), total=len(test_graphs)):
        account = current_graph["account"]
        with open(f"graph_sampled/{account}.txt","r") as f:
            current_graph_repr=f.read()
        result = await aux_querygpt.query_graph_characteristics_raw(
            ref_graphs,
            current_graph_repr,
            llm,
        )
        results[account] = result
        with open("3-full-repr-by-graph-4.0.pkl", "wb") as f:
            pickle.dump(results, f)
        with open("3-full-repr-by-graph-4.0.json", "w") as f:
            json.dump(results, f, indent=2,ensure_ascii=False)


await main()
# %%
