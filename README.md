# LLM4TG

This project includes the source code for our work **[Large Language Models for Cryptocurrency Transaction Analysis: A Bitcoin Case Study](https://arxiv.org/abs/2501.18158)**. If you find our work helpful for your research, please consider citing it as:

    @article{lei2025large,
      title={Large Language Models for Cryptocurrency Transaction Analysis: A Bitcoin Case Study},
      author={Lei, Yuchen and Xiang, Yuexin and Wang, Qin and Dowsley, Rafael and Yuen, Tsz Hon and Yu, Jiangshan},
      journal={arXiv preprint arXiv:2501.18158},
      year={2025}
    }

If you have any questions please feel free to contact me by e-mail at Yuexin.Xiang@monash.edu.

## Contents

- [Dataset and Preprocess](#dataset-and-preprocess)
- [Layered Framework](#layered-framework)
- [LLM4TG Format](#llm4tg-format)
- [CETraS](#cetras-algorithm)
- [Experiment](#experiment)
- [Acknowledgment](#acknowledgment)

## Dataset and Preprocess
## 1. Bitcoin Address Subgraph Dataset ([BASD-8](https://www.kaggle.com/datasets/lemonx/basd8))  
The **BASD-8** serves as our primary dataset for experiments. It captures **transactional subgraph structures** of Bitcoin addresses and was introduced in our **IEEE Big Data'22** paper, accessible on [Kaggle](https://ieeexplore.ieee.org/abstract/document/10020980).

## 2. Bitcoin Address Behavior Dataset ([BABD-13](https://www.kaggle.com/datasets/lemonx/babd13))  
To explore the differences between **raw graph structures** and **graph-derived features**, we also use the **BABD-13**. This dataset contains the same addresses as BASD-8 but focuses on **behavioral patterns** derived from Bitcoin transactions. It was introduced in our **IEEE TIFS'24** paper, available on [Kaggle](https://ieeexplore.ieee.org/abstract/document/10375557).


## 


## LLM4TG Format
(purpose and which file) The structure of the Bitcoin transaction graph inputting to LLMs is shown in ...
The data could be found in BASD-8 update.


## CETraS Algorithm
(purpose and which file) This algorithm compressed the ... subgraphs 


## Experiment
(how to reproduce the results shown in Tables and Figures from levels 1 - 3) First, our experiments on BASD-8 cover transaction graph understanding levels 1 to 3, details can be found in `2-basd8-basic.py`, where `aux_querygpt` function is used for querying through OpenAI APIs.


## Acknowledgment
This research is supported by **OpenAI Researcher Access Program**.


