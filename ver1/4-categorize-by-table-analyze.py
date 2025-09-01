# %%
import asyncio
import importlib
import json
import pickle
from glob import glob
from io import StringIO

import networkx
import numpy as np
import pandas as pd
import tiktoken
from networkx import DiGraph
from networkx.classes.coreviews import AtlasView

import aux_querygpt

importlib.reload(aux_querygpt)


with open("4-categorize-by-table-gpt4.pkl", "rb") as f:
    results = pickle.load(f)
# with open("4-categorize-by-table.json", "w") as f:
#     json.dump(results, f)
# groupby a dict by value
from collections import defaultdict

acc = sum([v["ground_truth"] == v["llm_result"][0] for k, v in results.items()])
acctop3 = sum([v["ground_truth"] in v["llm_result"] for k, v in results.items()])
print(acc / len(results))
print(acctop3 / len(results))
categories = set([v["ground_truth"] for k, v in results.items()])
print(categories)
category2num = {v: k for k, v in enumerate(categories)}
print(category2num)
y_true = [category2num[v["ground_truth"]] for k, v in results.items()]
y_pred = [category2num[v["llm_result"][0]] for k, v in results.items()]
print(y_true)
print(y_pred)
import sklearn.metrics

f1 = sklearn.metrics.f1_score(y_true, y_pred, average=None)
recall = sklearn.metrics.recall_score(y_true, y_pred, average=None)
precision = sklearn.metrics.precision_score(y_true, y_pred, average=None)
# accu = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize="true").diagonal()
accu = []
y_true = np.array(y_true)
y_pred = np.array(y_pred)
accu = [
    sum((y_true == i) == (y_pred == i)) / len(y_true) for i in range(len(categories))
]
print(recall)
print(precision)
print(accu)
result_df = pd.DataFrame(
    {
        "label": category2num.keys(),
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "accu": accu,
    }
)
print(result_df)

f1 = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
recall = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
precision = sklearn.metrics.precision_score(y_true, y_pred, average="macro")
print(f"{f1=}, {recall=}, {precision=}")

# %%
