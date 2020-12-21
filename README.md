## A Pytorch Implementation of Neural Click Model (NCM)

### Introduction

This is an non-official implementation of Neural Click Model proposed in the paper: [A Neural Click Model for Web Search. WWW 2016](https://dl.acm.org/doi/10.1145/2872427.2883033).

**NOTE**: There are several modification in my implementation compared to the original paper:

1. The original paper use one-hot encoding for query/doc embeddings. This makes the embedding dimensionality up to 10240, which is too large for training on GPU efficiently. So here I use nn.embedding provided by PyTorch to constrain the dimensionality.
2. Vertical type is added as side information. Vertical type means the representation style of a displayed document (e.g., organic vertical, the illustrated vertical, the encyclopediavertical. To the best of my knowledge, up to now, only TianGong-ST dataset provides this side information.
3. I input the query information at each RNN time step, instead of only using query for RNN state initialization.

### Requirements

- python 3.7
- pytorch 1.6.0+cu101
- torchvision 0.7.0+cu101
- tensorboard 2.1

### Input Formats & Data Preparation

After data pre-processing, we can put all the generated files into ```./data/dataset/``` folder as input files for NCM. Demo input files are available under the ```./data/demo/``` directory. 

The format of train & dev & test & label input files is as follows:

- Each line: ```<session id><tab><query id><tab>[<document ids>]<tab>[<vtype ids>]<tab>[<clicks infos>]<tab>[<relevance>]```

### Recommended Hyperparameter

I provided some recommended hyperparameters below. The example training shell can be found at ```./example_run.sh```

- optim: adam
- eval_freq: 100
- check_point: 100
- learning_rate: 0.001
- lr_decay: 0.5
- weight_decay: 1e-5
- dropout_rate: 0.5
- num_steps: 40000
- embed_size: 64
- hidden_size: 64
- batch_size: 128
- patience: 5

### Important Tips

For brevity and legibility of the code, I remove many non-core code:

1. Data-preprocessing for TianGong-ST dataset
2. Sequence-independent ranking for relevance estimation
3. Synthetic dataset generation

However, you can checkout these codes at the commit, (but the code might lack comments): 

```
commit 5b114cbdab0da4de1c43ebc3d666a7cac1818a23
Author: LinJianghao <chiangel.ljh@gmail.com>
Date:   Thu Nov 19 10:31:32 2020 +0800
```
