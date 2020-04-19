# coding: utf8

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import logging

use_cuda = torch.cuda.is_available()

class NCM(nn.Module):
    def __init__(self, args, query_size, doc_size, input_size):
        super(NCM, self).__init__()
        self.args = args

        # Get logger
        self.logger = logging.getLogger("NCM")

        # Config setting
        self.input_size = input_size
        self.hidden_size = args.hidden_size
        self.dropout_rate = args.dropout_rate
        self.query_size = query_size
        self.doc_size = doc_size
        self.best_eval_loss = 2333
        
        # create network components
        if self.args.model_type.lower() == 'rnn':
            self.rnn = nn.RNN(input_size=self.input_size,
                              hidden_size=self.hidden_size,
                              nonlinearity='relu',
                              dropout=self.dropout_rate)
            # print(self.rnn)
        elif self.args.model_type.lower() == 'lstm':
            self.lstm = nn.LSTM(input_size=self.input_size, 
                                hidden_size=self.hidden_size,
                                dropout=self.dropout_rate)
            # print(self.lstm)
        else:
            raise NotImplementedError('Unsupported model_type: {}'.format(self.args.model_type))
        self.output_linear = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, query_embed, doc_embed, clicks):
        # assert self.args.batch_size == query.size(0)
        batch_size = query_embed.size(0)
        max_doc_num = doc_embed.size(1)
        clicks = clicks.view(batch_size, -1, 1).float() # [batch_size, 10, 1]

        # print('query_embed: {}\n{}\n'.format(query_embed.size(), query_embed))
        # print('doc_embed: {}\n{}\n'.format(doc_embed.size(), doc_embed))
        # print('clicks: {}\n{}\n'.format(clicks.size(), clicks))

        if self.args.model_type.lower() == 'rnn':
            rnn_input = torch.cat((clicks, doc_embed), 2) # [batch_size, 10, 1 + doc_embed_size]
        elif self.args.model_type.lower() == 'lstm':
            # generate the first input for LSTM: q + 0_i + 0_d
            tmp = torch.zeros(batch_size, 1, 1 + doc_embed.size(2)).cuda()
            first_input = torch.cat((query_embed, tmp), 2) # [batch_size, 1, query_embed_size + 1 + doc_embed_size]
            # generate the rest inputs: 0_q + i + d
            tmp = torch.zeros(batch_size, max_doc_num, query_embed.size(2)).cuda()
            rest_input = torch.cat((tmp, clicks, doc_embed), 2) # [batch_size, max_doc_num, query_embed_size + 1 + doc_embed_size]
            # generate lstm_input 
            lstm_input = torch.cat((first_input, rest_input), 1) # [batch_size, 11, query_embed_size + 1 + doc_embed_size]
            lstm_input = lstm_input.transpose(0, 1) # [11, batch_size, query_embed_size + 1 + doc_embed_size]
            # Initialize h_0, c_0 if needed. The default values are all zeros.
            outputs, _ = self.lstm(lstm_input) # outputs: [11, batch_size, hidden_size]
        # print('outputs: {}\n{}\n'.format(outputs.size(), outputs))
        outputs = outputs.transpose(0, 1) # [batch_size, 11, hidden_size]
        # print('outputs: {}\n{}\n'.format(outputs.size(), outputs))
        outputs = self.output_linear(outputs).view(batch_size, -1)[:, 1:] # [batch_size, max_doc_num]
        # print('outputs: {}\n{}\n'.format(outputs.size(), outputs))
        logits = self.sigmoid(outputs) # [batch_size, max_doc_num]
        # print('logits: {}\n{}\n'.format(logits.size(), logits))
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
    model = NCM(args, 10, 20, 30)
    q = Variable(torch.zeros(8, 11).long())
    d = Variable(torch.zeros(8, 11).long())
    v = Variable(torch.zeros(8, 11).long())
    a = Variable(torch.zeros(8, 11).long())
    model(q, d, v, a)
    print(count_parameters(model))
