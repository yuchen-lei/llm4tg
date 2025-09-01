import json
import os
from glob import glob

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
curdir = os.path.dirname(__file__)
resultfiles = glob(f"{curdir}/*.json")
for result in resultfiles:
    with open(result, "r") as f:
        results = json.load(f)
    for addr in results:
        results[addr]["ground_truth"] = address_category_map[addr]
    with open(result, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
