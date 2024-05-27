# LLM4TG

This project includes the source code for our paper "**Advancing Transaction Graph Analysis with Large Language Models: A Case Study on Bitcoin Networks**". 

## Contents

- [Dataset and Preprocess](#dataset-and-preprocess)
- [Experiment](#experiment)
- [LLM4TG](#llm4tg)

## Dataset and Preprocess
The primary dataset used in the experiments is the **Bitcoin Address Subgraph Dataset** - [BASD-8](https://www.kaggle.com/datasets/lemonx/basd8). In addition, to explore the feature differences between graph-level and graph feature-level, we apply the addresses same to BASD-8 in the **Bitcoin Address Behavior Dataset** - [BABD-13](https://www.kaggle.com/datasets/lemonx/babd13), implemented via `1-extract.py`. 

## Experiment
First, our experiments on BASD-8 cover transaction graph understanding levels 1 to 4, details can be found in `2-basd8-basic.py`, where `aux_querygpt` function is used for querying through OpenAI APIs.

## LLM4TG
The structure of the Bitcoin transaction graph inputting to LLMs is shown in ...


