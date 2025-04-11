import json

llms = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
dataset = []
with open("lv9_www/result_api_spd_graph.json") as f:
    graph = json.load(f)
    for addr, info in graph.items():
        for llm in llms:
            tm = info[llm]["total_time"]
            tokens = tm["usage"]["completion_tokens"]
            sec = tm["time"]
            prom = tm["usage"]["prompt_tokens"]
            dataset.append(
                {
                    "llm": llm,
                    "tokens": tokens,
                    "sec": sec,
                    "prom": prom,
                }
            )
import pandas as pd

df = pd.DataFrame(dataset)

df["tokens_per_sec"] = df["tokens"] / df["sec"]
for llm in llms:
    print(llm, df[df["llm"] == llm]["tokens_per_sec"].mean())
