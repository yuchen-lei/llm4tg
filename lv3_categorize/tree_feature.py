import json
import os
import pickle
import sys
from glob import glob

import pandas as pd
import sklearn.ensemble
import sklearn.multiclass
import sklearn.neural_network
import sklearn.random_projection
import sklearn.svm
from tqdm.auto import tqdm

sys.path.append(".")

from aux_querygpt import aquery_once, get_llm_model

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
import sklearn.tree
import catboost


async def main():
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
    # print(df_ref[meaningful_columns].to_dict(orient="records"))
    model = sklearn.tree.DecisionTreeClassifier()
    # model = catboost.CatBoostClassifier()
    # model = sklearn.svm.LinearSVC(dual=True, max_iter=100000)
    # model = sklearn.neural_network.MLPClassifier(
    #     hidden_layer_sizes=(100, 100, 100), max_iter=1000
    # )
    # model = sklearn.ensemble.RandomForestClassifier()
    models = {
        "dt": sklearn.tree.DecisionTreeClassifier(),
        "svm": sklearn.svm.LinearSVC(dual=True),
        "mlp": sklearn.neural_network.MLPClassifier(
            hidden_layer_sizes=(100, 100, 100), max_iter=1000
        ),
        "forest": sklearn.ensemble.RandomForestClassifier(),
        "catboost": catboost.CatBoostClassifier(),
    }
    model_results = {}
    for model_name, model in models.items():
        model.fit(df_ref[meaningful_columns[1:]], df_ref["label"])
        y_pred = model.predict(test_graphs[meaningful_columns[1:]])
        y_true = test_graphs["label"]
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        f1 = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
        recall = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
        precision = sklearn.metrics.precision_score(y_true, y_pred, average="macro")
        marco_results = {
            "acc": acc,
            "f1": f1,
            "recall": recall,
            "precision": precision,
        }
        model_results[model_name] = marco_results
        print(model_name, marco_results)
    import joblib

    joblib.dump(models, f"{curdir}/models.pkl")
    joblib.dump(model_results, f"{curdir}/model_results.pkl")
    df = pd.DataFrame.from_dict(model_results, orient="index")
    df.to_csv(f"{curdir}/model_results.csv")
    return
    llm = get_llm_model("gpt-3.5-turbo")
    results = {}
    sem = asyncio.Semaphore(8)
    progbar1 = tqdm(total=len(test_graphs))
    progbar2 = tqdm(total=len(test_graphs))

    async def single_graph(account, label, current_graph_repr):
        async with sem:
            progbar1.update()
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
            with open(f"{curdir}/result_gpt35_feature_rs514_rd3.json", "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            progbar2.update()

    tasks = []
    for _, current_graph in test_graphs.iterrows():
        account = current_graph["account"]
        current_graph_repr = current_graph[meaningful_columns[1:]].to_dict()
        tasks.append(single_graph(account, current_graph["label"], current_graph_repr))
    await asyncio.gather(*tasks)


asyncio.run(main())
