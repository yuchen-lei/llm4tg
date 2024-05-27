import asyncio
import json
import os
import random
from glob import glob

import pandas as pd

from aux_querygpt import *

random.seed(42)

subgraph_files = glob("basd-8/**/*.graphml", recursive=True)


def split_category_address(path):
    category, address = path.split("\\")[-2:]
    address = os.path.splitext(address)[0]
    return category, address


address_category_map = {
    split_category_address(x)[1]: split_category_address(x)[0] for x in subgraph_files
}

df = pd.read_csv("babd13-slim.csv")
address_has_features = set(df["account"].to_list())
laundering_addresses = [
    x
    for x in address_category_map
    if address_category_map[x] == "money laundering" and x in address_has_features
]
tumbler_address = [
    x
    for x in address_category_map
    if address_category_map[x] == "tumbler" and x in address_has_features
]

laundering_addresses = random.sample(laundering_addresses, 10)
tumbler_address = random.sample(tumbler_address, 10)

ref_graphs = laundering_addresses + tumbler_address

df["label"] = df["account"].apply(lambda x: address_category_map[x])
df = df[df["account"].isin(ref_graphs)]
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
graph_features = {
    row["account"]: json.dumps(row[meaningful_columns].to_dict()).replace('"', "")
    for _, row in df.iterrows()
}
graph_prompt = []
for idx, address in enumerate(ref_graphs):
    graph_prompt.append(f"# Graph {idx+1}: features {graph_features[address]}")
    with open(f"graph_sampled/{address}.txt") as f:
        graph_prompt.append(f.read())
graph_prompt_repr = "\n".join(graph_prompt)

with open("5-distingush-2-graph-prompt.txt", "w") as f:
    f.write(graph_prompt_repr)
with open("5-distingush-2-graph-index.txt", "w") as f:
    f.write(
        "\n".join([f"{idx+1}: {address}" for idx, address in enumerate(ref_graphs)])
    )

print(token_count(graph_prompt_repr))


async def main():
    llm = get_llm_model()
    task = await distingush_graphs(graph_prompt_repr, llm, "distingush_graphs_type2")
    with open("5-distingush-2.json", "w") as f:
        f.write(json.dumps(task, indent=2))


asyncio.run(main())
