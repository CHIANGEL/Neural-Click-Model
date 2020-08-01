# A Pytorch Implementation of Neural Click Model (NCM)

## Introduction

This is an non-official implementation of Neural Click Model proposed in the paper: [A Neural Click Model for Web Search. WWW 2016](https://dl.acm.org/doi/10.1145/2872427.2883033).

## Requirements

- python 3.7
- pytorch 1.4.0
- tensorboardx 2.0
- tqdm 4.44.1

## Dataset

I use [TianGong-ST dataset](http://www.thuir.cn/tiangong-st/) provided by Sougou and Tsinghua University. This Chinese-centric TianGong-ST dataset is provided to support researches in a wide range of session-level Information Retrieval (IR) tasks. It consists of 147,155 refined Web search sessions, 40,596 unique queries, 297,597 Web pages, six kinds of weak relevance labels assessed by click models, and also a subset of 2,000 sessions with 5-level human relevance labels for documents of the last queries in them. In order to align with the experiment conducted on [CACM](https://github.com/CHIANGEL/Context-Aware-Click-Model), division of the datset is:

| Attribute           |   Train |  Dev   |   Test |
| :---: | :---: | :---: | :---: |
| Sessions            |  117431 | 13154  |  26570 |
| Queries             | 35903 | 9373 | 11391 |
| Avg Session Len     |    2.4099 |  2.4012  |   2.4986 |

## Input Formats & Data Preparation

I provide python files for data pre-processing:

- [TianGong-ST-NCM.py](TianGong-ST-NCM.py)
- [TianGong_HumanLabel_Parser.py](TianGong_HumanLabel_Parser.py)

After data pre-processing, we can put all the generated files into ```./data``` folder as input files for NCM. Demo input files are available under the ```./data``` directory. 

The format of train & dev & test input files is as follows:

- Each line: ```<session id><tab><query id><tab><zero><tab><zero><tab>[<document ids>]<tab>[<vtype ids>]<tab>[<clicks infos>]```
- The ```<zero>``` is a meaningless alignment term, which can be removed from the program.

The format of human_label.txt remains the same in TianGong-ST dataset:

- Each line: ```<sample id><tab><session id><tab><query id><tab><document id><tab><relevance label><tab><page validation>```

## Recommended Hyperparameter for TianGong-ST

I provide a [example_run.sh](example_run.sh) for training. The hyperparamemter is:

- optim: adam
- eval_freq: 100
- check_point: 100
- learning_rate: 0.001
- lr_decay: 0.5
- weight_decay: 1e-5
- dropout_rate: 0.5
- num_steps: 40000
- embed_size: 32
- hidden_size: 64
- batch_size: 32
- patience: 5
