import pandas as pd
from glob import glob
import os

graphs_from_file = glob("basd-8/**/*.graphml", recursive=True)
graphs_from_file = [os.path.basename(x) for x in graphs_from_file]
graphs_from_file = [os.path.splitext(x)[0] for x in graphs_from_file]


df = pd.read_csv("basd-8/subgraph_summary.csv")
l1 = df["address"].to_list()
assert all(x == y for x, y in zip(graphs_from_file, l1))

babd13 = pd.read_csv("BABD-13.csv")
babd13_slim = babd13[babd13["account"].isin(l1)]
babd13_slim.to_csv("babd13-slim.csv", index=False)

