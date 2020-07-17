# encoding:utf-8
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import logging
# from layers import DotAttention, Summ, PointerNet

use_cuda = torch.cuda.is_available()
INF = 1e30

class Network(nn.Module):
    def __init__(self, args, query_size, doc_size, vtype_size):
        super(Network, self).__init__()
        self.args = args
        self.logger = logging.getLogger("NCM")
        self.embed_size = args.embed_size   # 300 as default
        self.hidden_size = args.hidden_size # 150 as default
        self.dropout_rate = args.dropout_rate
        self.encode_gru_num_layer = 1
        self.query_size = query_size
        self.doc_size = doc_size
        self.vtype_size = vtype_size

        self.query_embedding = nn.Embedding(query_size, self.embed_size)
        self.doc_embedding = nn.Embedding(doc_size, self.embed_size)
        self.vtype_embedding = nn.Embedding(vtype_size, self.embed_size//2)
        self.action_embedding = nn.Embedding(2, self.embed_size//2)

        self.gru = nn.GRU(self.embed_size*3, self.hidden_size,
                          batch_first=True, dropout=self.dropout_rate, num_layers=self.encode_gru_num_layer)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.output_linear = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, query, doc, vtype, action):
        batch_size = query.size()[0]
        max_doc_num = doc.size()[1]

        query_embed = self.query_embedding(query)  # batch_size, 11, embed_size
        doc_embed = self.doc_embedding(doc)  # batch_size, 11, embed_size
        vtype_embed = self.vtype_embedding(vtype)  # batch_size, 11, embed_size//2
        action_embed = self.action_embedding(action)  # batch_size, 11, embed_size//2

        gru_input = torch.cat((query_embed, doc_embed, vtype_embed, action_embed), dim=2)
        init_gru_state = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            init_gru_state = init_gru_state.cuda()
        outputs, _ = self.gru(gru_input, init_gru_state)
        outputs = self.dropout(outputs)
        logits = self.sigmoid(self.output_linear(outputs)).view(batch_size, max_doc_num)[:,1:]
        # print logits.size()
        return logits

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    class config():
        def __init__(self):
            self.embed_size = 300
            self.hidden_size = 150
            self.dropout_rate = 0.2

    args = config()
    model = Network(args, 10, 20, 30)
    q = Variable(torch.zeros(8, 11).long())
    d = Variable(torch.zeros(8, 11).long())
    v = Variable(torch.zeros(8, 11).long())
    a = Variable(torch.zeros(8, 11).long())
    model.forward(q, d, v, a)  # ？
    print (count_parameters(model))
