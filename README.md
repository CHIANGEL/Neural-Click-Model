# Neural-Click-Model
A Neural Click Model for Web Search. WWW 2016

python ./TREC2014.py --dict_and_list --data_txt --embedding

doc_embedding_QD not supported: not enough memory

Directly concatenate doc_embedding and interaction_embedding (binary vector of size 1)

LSTM + QD+Q+D

## Usage

First use scripts in scripts_for_dataset to generate useful data files from xml files.

Then run the NCM model to train & test

In dev & test period, batch_size must be 1. Otherwise there are paddings in data batches, leading to repeated computation of some samples.