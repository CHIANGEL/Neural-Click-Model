import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import logging

use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')
INF = 1e30

class NCM(nn.Module):
    def __init__(self, args, query_size, doc_size, vtype_size):
        super(NCM, self).__init__()
        self.args = args
        self.logger = logging.getLogger("NCM")
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.dropout_rate = args.dropout_rate
        self.query_size = query_size
        self.doc_size = doc_size
        self.vtype_size = vtype_size

        self.query_embedding = nn.Embedding(query_size, self.embed_size)
        self.doc_embedding = nn.Embedding(doc_size, self.embed_size)
        self.vtype_embedding = nn.Embedding(vtype_size, self.embed_size // 2)
        self.action_embedding = nn.Embedding(2, self.embed_size // 2)

        self.gru = nn.GRU(self.embed_size * 3, self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.output_linear = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, query, doc, vtype, action, gru_state=None):
        batch_size = query.size()[0]
        max_doc_num = doc.size()[1]

        query_embed = self.query_embedding(query)  # [batch_size, 11, embed_size]
        doc_embed = self.doc_embedding(doc)  # [batch_size, 11, embed_size]
        vtype_embed = self.vtype_embedding(vtype)  # [batch_size, 11, embed_size // 2]
        action_embed = self.action_embedding(action)  # [batch_size, 11, embed_size // 2]

        gru_input = torch.cat((query_embed, doc_embed, vtype_embed, action_embed), dim=2)
        if gru_state == None:
            gru_state = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            gru_state = gru_state.cuda()
        outputs, gru_state = self.gru(gru_input, gru_state)
        outputs = self.dropout(outputs)
        logits = self.sigmoid(self.output_linear(outputs)).view(batch_size, max_doc_num)
        if logits.shape[1] > 1:
            logits = logits[:, 1:]
        return logits, gru_state

