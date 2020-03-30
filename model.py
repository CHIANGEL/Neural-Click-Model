# coding: utf8

import os
import time
import logging
import json
import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter

from NCM import NCM

use_cuda = torch.cuda.is_available()

MINF = 1e-30

class Model(object):
    """
     Model class performs as an interface layer for NCM model.
    """
    def __init__(self, args, query_size, doc_size):
        self.args = args

        # Get logger
        self.logger = logging.getLogger("NCM")

        # Config setting
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.eval_freq = args.eval_freq
        self.global_step = args.load_model if args.load_model > -1 else 0
        self.patience = args.patience
        self.max_doc_num = args.max_doc_num
        self.writer = SummaryWriter(self.args.summary_dir) if args.train else None
        
        # create the NCM model instance
        self.model = NCM(self.args, query_size, doc_size)
        self.optimizer = self.create_train_optim()
        self.criterion = nn.MSELoss()
        if use_cuda:
            self.model = self.model.cuda()
        if args.data_parallel:
            self.model = nn.DataParallel(self.model)
       
    def create_train_optim(self):
        """
         Create the optimizer according to the args
        """
        if self.optim_type == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.optim_type == 'adadelta':
            optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.optim_type == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.optim_type == 'rprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.optim_type == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        return optimizer

    def adjust_learning_rate(self, decay_rate=0.5):
        '''
         Decay the learning rate once reaching the patience threshold
        '''
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate

    def compute_loss(self, pred_scores, target_scores):
        """
         Compute the loss function
        """
        total_loss = 0.
        loss_list = []
        cnt = 0
        for batch_idx, scores in enumerate(target_scores):
            cnt += 1
            loss = 0.
            for position_idx, score in enumerate(scores):
                if score == 0:
                    loss -= torch.log(1. - pred_scores[batch_idx][position_idx].view(1) + MINF)
                else:
                    loss -= torch.log(pred_scores[batch_idx][position_idx].view(1) + MINF)
            loss_list.append(loss.data[0])
            total_loss += loss
        total_loss /= cnt
        return total_loss, loss_list

    def _train_epoch(self, train_batches, data, min_eval_loss, patience, step_pbar):
        """
         Train the model for a single epoch.
        """
        evaluate = True
        exit_tag = False
        num_steps = self.args.num_steps
        check_point, batch_size = self.args.check_point, self.args.batch_size
        save_dir, save_prefix = self.args.model_dir, self.args.algo

        for bitx, batch in enumerate(train_batches):
            self.global_step += 1
            step_pbar.update(1)
            QIDS = Variable(torch.from_numpy(np.array(batch['qids'], dtype=np.int64)))
            UIDS = Variable(torch.from_numpy(np.array(batch['uids'], dtype=np.int64)))
            CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64))[:, 0: -1])
            tmp = torch.zeros(CLICKS.size(0), 1).long()
            CLICKS = torch.cat((tmp, CLICKS), 1) # CLICKS stands for interaction representation in NCM paper

            if use_cuda:
                QIDS, UIDS, CLICKS = QIDS.cuda(), UIDS.cuda(), CLICKS.cuda()
            
            # print('QIDS: {}\n{}\n'.format(QIDS.size(), QIDS))
            # print('UIDS: {}\n{}\n'.format(UIDS.size(), UIDS))
            # print('CLICKS: {}\n{}\n'.format(CLICKS.size(), CLICKS))

            self.model.train()
            self.optimizer.zero_grad()
            pred_logits = self.model(QIDS, UIDS, CLICKS)
            loss, loss_list = self.compute_loss(pred_logits, batch['clicks'])
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar('train/loss', loss.data[0], self.global_step)

            if evaluate and self.global_step % self.eval_freq == 0:
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, shuffle=False)
                    eval_loss = self.evaluate(eval_batches, data, 
                                              result_dir=self.args.result_dir,
                                              result_prefix='train_dev.predicted.{}.{}'.format(self.args.algo, self.global_step), 
                                              t=-1)
                    self.writer.add_scalar("dev/loss", eval_loss, self.global_step)

                    if eval_loss < min_eval_loss:
                        min_eval_loss = eval_loss
                        patience = 0
                    else:
                        patience += 1
                    if patience >= self.patience:
                        self.adjust_learning_rate(self.args.lr_decay)
                        self.learning_rate *= self.args.lr_decay
                        self.writer.add_scalar('train/lr', self.learning_rate, self.global_step)
                        # reset the saved min eval loss (make it a bit larger). Because lr is decayed.
                        min_eval_loss = eval_loss
                        patience = 0
                        self.patience += 1
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            if check_point > 0 and self.global_step % check_point == 0:
                self.save_model(save_dir, save_prefix)
            if self.global_step >= num_steps:
                exit_tag = True

        return exit_tag, min_eval_loss, patience

    def train(self, data):
        '''
         Training of the model starts here.
        '''
        epoch, patience, min_eval_loss = 0, 0, 1e10
        step_pbar = tqdm(total=self.args.num_steps)
        exit_tag = False
        self.writer.add_scalar('train/lr', self.learning_rate, self.global_step)
        while not exit_tag:
            epoch += 1
            train_batches = data.gen_mini_batches('train', self.args.batch_size, shuffle=True)
            exit_tag, min_eval_loss, patience = self._train_epoch(train_batches, data, min_eval_loss, patience, step_pbar)

    def evaluate(self, eval_batches, dataset, result_dir=None, result_prefix=None, t=-1):
        eval_outputs = []
        total_loss, total_num = 0., 0
        self.logger.info('Evaluation at global_step {}.'.format(self.global_step))
        with torch.no_grad():
            for b_itx, batch in enumerate(eval_batches):
                if b_itx == t:
                    break

                QIDS = Variable(torch.from_numpy(np.array(batch['qids'], dtype=np.int64)))
                UIDS = Variable(torch.from_numpy(np.array(batch['uids'], dtype=np.int64)))
                CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64))[:, 0: -1])
                tmp = torch.zeros(CLICKS.size(0), 1).long()
                CLICKS = torch.cat((tmp, CLICKS), 1) # CLICKS stands for interaction representation in NCM paper

                if use_cuda:
                    QIDS, UIDS, CLICKS = QIDS.cuda(), UIDS.cuda(), CLICKS.cuda()

                self.model.eval()
                pred_logits = self.model(QIDS, UIDS, CLICKS)
                loss, loss_list = self.compute_loss(pred_logits, batch['clicks'])

                for eval_loss, data, pred_logit in zip(loss_list, batch['raw_data'], pred_logits.data.cpu().numpy().tolist()):
                    eval_outputs.append([data['qids'], 
                                         data['uids'],
                                         data['clicks'],
                                         pred_logit, 
                                         eval_loss])
                total_loss += loss
                total_num += 1

            if result_dir is not None and result_prefix is not None:
                result_file = os.path.join(result_dir, result_prefix + '.txt')
                with open(result_file, 'w') as fout:
                    for sample in eval_outputs:
                        fout.write('\t'.join(map(str, sample)) + '\n')

                self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))
            ave_span_loss = 1.0 * total_loss / total_num
        return ave_span_loss

    def save_model(self, model_dir, model_prefix):
        """
         Save the NCM model and optimizer into model_dir with model_prefix as the model indicator
        """
        torch.save(self.model.state_dict(), os.path.join(model_dir, model_prefix+'_{}.model'.format(self.global_step)))
        torch.save(self.optimizer.state_dict(), os.path.join(model_dir, model_prefix + '_{}.optimizer'.format(self.global_step)))
        self.logger.info('Model and optimizer saved in {}, with prefix {} and global step {}.'.format(model_dir, model_prefix, self.global_step))

    def load_model(self, model_dir, model_prefix, global_step):
        """
         Load the NCM model and optimizer from model_dir with model_prefix as the model indicator
        """
        # Load the optimizer
        optimizer_path = os.path.join(model_dir, model_prefix + '_{}.optimizer'.format(global_step))
        if not os.path.isfile(optimizer_path):
            optimizer_path = os.path.join(model_dir, model_prefix + '_best_{}.optimizer'.format(global_step))
        if os.path.isfile(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path))
            self.logger.info('Optimizer restored from {}, with prefix {} and global step {}.'.format(model_dir, model_prefix, global_step))
        
        # Load the NCM model
        model_path = os.path.join(model_dir, model_prefix + '_{}.model'.format(global_step))
        if not os.path.isfile(model_path):
            model_path = os.path.join(model_dir, model_prefix + '_best_{}.model'.format(global_step))
        if use_cuda:
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        self.logger.info('Model restored from {}, with prefix {} and global step {}.'.format(model_dir, model_prefix, global_step))