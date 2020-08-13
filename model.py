import os
import time
import logging
import json
import numpy as np
import torch
import copy
import math
import random
from torch.autograd import Variable
from tqdm import tqdm
from network import Network
from tensorboardX import SummaryWriter
from torch import nn
from ndcg import RelevanceEstimator
from TianGong_HumanLabel_Parser import TianGong_HumanLabel_Parser
from utils import *

use_cuda = torch.cuda.is_available()

MINF = 1e-30

class Model(object):
    """
    Implements the main reading comprehension model.
    """
    def __init__(self, args, query_size, doc_size, vtype_size):
        # logging
        self.logger = logging.getLogger("NCM")

        # basic config
        self.args = args
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.eval_freq = args.eval_freq
        self.global_step = args.load_model if args.load_model > -1 else 0
        self.patience = args.patience
        self.max_d_num = args.max_d_num
        self.writer = None
        if args.train:
            self.writer = SummaryWriter(self.args.summary_dir)

        # network
        self.model = Network(self.args, query_size, doc_size, vtype_size)
        if args.data_parallel:
            self.model = nn.DataParallel(self.model)
        if use_cuda:
            self.model = self.model.cuda()
        self.optimizer = self.create_train_op()
        self.criterion = nn.MSELoss()
        
        # for NDCG@k
        self.relevance_queries = TianGong_HumanLabel_Parser().parse(args.human_label_dir)
        self.relevance_estimator = RelevanceEstimator(args.minimum_occurrence)
        self.trunc_levels = [1, 3, 5, 10]

    def compute_loss(self, pred_scores, target_scores):
        """
        The loss function
        """
        total_loss = 0.
        loss_list = []
        cnt = 0
        for batch_idx, scores in enumerate(target_scores):
            cnt += 1
            loss = 0.
            for position_idx, score in enumerate(scores[2:]):
                if score == 0:
                    loss -= torch.log(1. - pred_scores[batch_idx][position_idx].view(1) + MINF)
                else:
                    loss -= torch.log(pred_scores[batch_idx][position_idx].view(1) + MINF)
            loss_list.append(loss.data[0])
            total_loss += loss
        total_loss /= cnt
        return total_loss, loss_list

    def create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
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
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        return optimizer

    def adjust_learning_rate(self, decay_rate=0.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate

    def _train_epoch(self, train_batches, data, max_metric_value, metric_save, patience, step_pbar):
        """
        Trains the model for a single epoch.
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
            VIDS = Variable(torch.from_numpy(np.array(batch['vids'], dtype=np.int64)))
            CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64))[:,:-1])
            if use_cuda:
                QIDS, UIDS, VIDS, CLICKS = QIDS.cuda(), UIDS.cuda(), VIDS.cuda(), CLICKS.cuda()

            self.model.train()
            self.optimizer.zero_grad()
            pred_logits, _ = self.model(QIDS, UIDS, VIDS, CLICKS)
            loss, loss_list = self.compute_loss(pred_logits, batch['clicks'])
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar('train/loss', loss.data[0], self.global_step)

            if evaluate and self.global_step % self.eval_freq == 0:
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, shuffle=False)
                    eval_loss, perplexity, perplexity_at_rank = self.evaluate(eval_batches, data, result_dir=self.args.result_dir, t=-1,
                                                                                result_prefix='train_dev.predicted.{}.{}'.format(self.args.algo,
                                                                                                                                self.global_step))
                    eval_batches1 = data.gen_mini_batches('test', batch_size, shuffle=False)
                    eval_loss1, perplexity1, perplexity_at_rank1 = self.evaluate(eval_batches1, data, result_dir=self.args.result_dir, t=-1,
                                                                                result_prefix='train_test.predicted.{}.{}'.format(self.args.algo,
                                                                                                                                self.global_step))
                    self.writer.add_scalar("dev/loss", eval_loss, self.global_step)
                    self.writer.add_scalar("dev/perplexity", perplexity, self.global_step)
                    self.writer.add_scalar("test/loss", eval_loss1, self.global_step)
                    self.writer.add_scalar("test/perplexity", perplexity1, self.global_step)

                    for trunc_level in self.trunc_levels:
                        ndcg_version1, ndcg_version2 = self.relevance_estimator.evaluate(self, data, self.relevance_queries, trunc_level)
                        self.writer.add_scalar("NDCG_version1/{}".format(trunc_level), ndcg_version1, self.global_step)
                        self.writer.add_scalar("NDCG_version2/{}".format(trunc_level), ndcg_version2, self.global_step)

                    if eval_loss < metric_save:
                        metric_save = eval_loss
                        patience = 0
                    else:
                        patience += 1
                    if patience >= self.patience:
                        self.adjust_learning_rate(self.args.lr_decay)
                        self.learning_rate *= self.args.lr_decay
                        self.writer.add_scalar('train/lr', self.learning_rate, self.global_step)
                        metric_save = eval_loss
                        patience = 0
                        self.patience += 1
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            if check_point > 0 and self.global_step % check_point == 0:
                self.save_model(save_dir, save_prefix)
            if self.global_step >= num_steps:
                exit_tag = True

        return max_metric_value, exit_tag, metric_save, patience

    def train(self, data):
        max_metric_value, epoch, patience, metric_save = 0., 0, 0, 1e10
        step_pbar = tqdm(total=self.args.num_steps)
        exit_tag = False
        self.writer.add_scalar('train/lr', self.learning_rate, self.global_step)
        while not exit_tag:
            epoch += 1
            train_batches = data.gen_mini_batches('train', self.args.batch_size, shuffle=True)
            max_metric_value, exit_tag, metric_save, patience = self._train_epoch(train_batches, data,
                                                                                max_metric_value, metric_save,
                                                                                patience, step_pbar)

    def compute_perplexity(self, pred_scores, target_scores):
        '''
        Compute the perplexity
        '''
        perplexity_at_rank = [0.0] * 10 # 10 docs per query
        total_num = 0
        for batch_idx, scores in enumerate(target_scores):
            total_num += 1
            for position_idx, score in enumerate(scores[2:]):
                if score == 0:
                    perplexity_at_rank[position_idx] += torch.log2(1. - pred_scores[batch_idx][position_idx].view(1) + MINF)
                else:
                    perplexity_at_rank[position_idx] += torch.log2(pred_scores[batch_idx][position_idx].view(1) + MINF)
        return total_num, perplexity_at_rank

    def evaluate(self, eval_batches, dataset, result_dir=None, result_prefix=None, t=-1):
        eval_ouput = []
        total_loss, total_num = 0., 0
        perplexity_num = 0
        perplexity_at_rank = [0.0] * 10 # 10 docs per query
        for b_itx, batch in enumerate(eval_batches):
            if b_itx == t:
                break
            if b_itx % 5000 == 0:
                self.logger.info('Evaluation step {}.'.format(b_itx))
            QIDS = Variable(torch.from_numpy(np.array(batch['qids'], dtype=np.int64)))
            UIDS = Variable(torch.from_numpy(np.array(batch['uids'], dtype=np.int64)))
            VIDS = Variable(torch.from_numpy(np.array(batch['vids'], dtype=np.int64)))
            CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64))[:,:-1])
            if use_cuda:
                QIDS, UIDS, VIDS, CLICKS = QIDS.cuda(), UIDS.cuda(), VIDS.cuda(), CLICKS.cuda()

            self.model.eval()
            pred_logits, _ = self.model(QIDS, UIDS, VIDS, CLICKS)
            loss, loss_list = self.compute_loss(pred_logits, batch['clicks'])
            tmp_num, tmp_perplexity_at_rank = self.compute_perplexity(pred_logits, batch['clicks'])
            perplexity_num += tmp_num
            perplexity_at_rank = [perplexity_at_rank[i] + tmp_perplexity_at_rank[i] for i in range(10)]
            # total_loss_list += loss_list
            # pred_logits_list = pred_logits.data.cpu().numpy().tolist()
            for pred_metric, data, pred_logit in zip(loss_list, batch['raw_data'], pred_logits.data.cpu().numpy().tolist()):
                eval_ouput.append([data['session_id'], data['query'],
                                    data['urls'][1:], data['vtypes'][1:], data['clicks'][2:], pred_logit, pred_metric])
            total_loss += loss.data[0] * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.txt')
            with open(result_file, 'w') as fout:
                for sample in eval_ouput:
                    fout.write('\t'.join(map(str, sample)) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

        assert total_num == perplexity_num
        ave_span_loss = 1.0 * total_loss / total_num
        perplexity_at_rank = [2 ** (-x / perplexity_num) for x in perplexity_at_rank]
        perplexity = sum(perplexity_at_rank) / len(perplexity_at_rank)
        return ave_span_loss, perplexity, perplexity_at_rank
    
    def predict_relevance(self, qid, uid, vid):
        qids = [[qid, qid]]
        uids = [[0, uid]]
        vids = [[0, vid]]
        clicks = [[0, 0, 0]]
        QIDS = Variable(torch.from_numpy(np.array(qids, dtype=np.int64)))
        UIDS = Variable(torch.from_numpy(np.array(uids, dtype=np.int64)))
        VIDS = Variable(torch.from_numpy(np.array(vids, dtype=np.int64)))
        CLICKS = Variable(torch.from_numpy(np.array(clicks, dtype=np.int64))[:,:-1])
        if use_cuda:
            QIDS, UIDS, VIDS, CLICKS = QIDS.cuda(), UIDS.cuda(), VIDS.cuda(), CLICKS.cuda()
        self.model.eval()
        pred_logits, _ = self.model(QIDS, UIDS, VIDS, CLICKS)
        return pred_logits[0][0]
    
    def ndcg_cheat(self, label_batches, data, result_dir=None, result_prefix=None, stop=-1):
        trunc_levels = [1, 3, 5, 10]
        ndcg_version1, ndcg_version2 = {}, {}
        useless_session, cnt_version1, cnt_version2 = {}, {}, {}
        for k in trunc_levels:
            ndcg_version1[k] = 0.0
            ndcg_version2[k] = 0.0
            useless_session[k] = 0
            cnt_version1[k] = 0
            cnt_version2[k] = 0
        with torch.no_grad():
            for b_itx, batch in enumerate(label_batches):
                if b_itx == stop:
                    break

                QIDS = Variable(torch.from_numpy(np.array(batch['qids'], dtype=np.int64)))
                UIDS = Variable(torch.from_numpy(np.array(batch['uids'], dtype=np.int64)))
                VIDS = Variable(torch.from_numpy(np.array(batch['vids'], dtype=np.int64)))
                CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64))[:, :-1])
                true_relevances = batch['relevances'][0]
                if use_cuda:
                    QIDS, UIDS, VIDS, CLICKS = QIDS.cuda(), UIDS.cuda(), VIDS.cuda(), CLICKS.cuda()

                self.model.eval()
                pred_logits, _ = self.model(QIDS, UIDS, VIDS, CLICKS)
                relevances = pred_logits.data.cpu().numpy().reshape(-1).tolist()
                pred_rels = {}
                for idx, relevance in enumerate(relevances):
                    pred_rels[idx] = relevance
                
                #print('{}: \n{}'.format('relevances', relevances))
                #print('{}: \n{}'.format('true_relevances', true_relevances))
                #print('{}: \n{}'.format('pred_rels', pred_rels))
                for k in trunc_levels:
                    #print('\n{}: {}'.format('trunc_level', k))
                    ideal_ranking_relevances = sorted(true_relevances, reverse=True)[:k]
                    ranking = sorted([idx for idx in pred_rels], key = lambda idx : pred_rels[idx], reverse=True)
                    ranking_relevances = [true_relevances[idx] for idx in ranking[:k]]
                    #print('{}: {}'.format('ideal_ranking_relevances', ideal_ranking_relevances))
                    #print('{}: {}'.format('ranking', ranking))
                    #print('{}: {}'.format('ranking_relevances', ranking_relevances))
                    dcg = self.dcg(ranking_relevances)
                    idcg = self.dcg(ideal_ranking_relevances)
                    if dcg > idcg:
                        pprint.pprint(ranking_relevances)
                        pprint.pprint(ideal_ranking_relevances)
                        pprint.pprint(dcg)
                        pprint.pprint(idcg)
                        pprint.pprint(info_per_query)
                        assert 0
                    ndcg = dcg / idcg if idcg > 0 else 1.0
                    if idcg == 0:
                        useless_session[k] += 1
                        cnt_version2[k] += 1
                        ndcg_version2[k] += ndcg
                    else:
                        ndcg = dcg / idcg
                        cnt_version1[k] += 1
                        cnt_version2[k] += 1
                        ndcg_version1[k] += ndcg
                        ndcg_version2[k] += ndcg
                    #print('{}: {}'.format('dcg', dcg))
                    #print('{}: {}'.format('idcg', idcg))
                    #print('{}: {}'.format('ndcg', ndcg))
            '''for k in trunc_levels:
                print()
                print('{}: {}'.format('cnt_version1[{}]'.format(k), cnt_version1[k]))
                print('{}: {}'.format('useless_session[{}]'.format(k), useless_session[k]))
                print('{}: {}'.format('cnt_version2[{}]'.format(k), cnt_version2[k]))'''
            for k in trunc_levels:
                assert cnt_version1[k] + useless_session[k] == 2000
                assert cnt_version2[k] == 2000
                ndcg_version1[k] /= cnt_version1[k]
                ndcg_version2[k] /= cnt_version2[k]
        return ndcg_version1, ndcg_version2

    def dcg(self, ranking_relevances):
        """
        Computes the DCG for a given ranking_relevances
        """
        return sum([(2 ** relevance - 1) / math.log(rank + 2, 2) for rank, relevance in enumerate(ranking_relevances)])

    def generate_click_seq(self, eval_batches, file_path, file_name):
        logit_list_for_print = []
        click_list_for_print = []
        true_click_list_for_print = []
        check_path(file_path)
        data_path = os.path.join(file_path, file_name)
        file = open(data_path, 'w')
        for b_itx, batch in enumerate(eval_batches):
            if b_itx % 5000 == 0:
                self.logger.info('Generating click sequence at step: {}.'.format(b_itx))
            QIDS = Variable(torch.from_numpy(np.array(batch['qids'], dtype=np.int64)))
            UIDS = Variable(torch.from_numpy(np.array(batch['uids'], dtype=np.int64)))
            VIDS = Variable(torch.from_numpy(np.array(batch['vids'], dtype=np.int64)))
            CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64)))
            if use_cuda:
                QIDS, UIDS, VIDS, CLICKS = QIDS.cuda(), UIDS.cuda(), VIDS.cuda(), CLICKS.cuda()

            self.model.eval()
            gru_state = Variable(torch.zeros(1, self.args.batch_size, self.hidden_size))
            CLICK_ = torch.zeros(self.args.batch_size, 1, dtype=CLICKS.dtype)
            if use_cuda:
                gru_state, CLICK_ = gru_state.cuda(), CLICK_.cuda()
            logit_list = []
            click_list = []
            for i in range(self.max_d_num + 1):
                logit, gru_state = self.model(QIDS[:, i:i+1], UIDS[:, i:i+1], VIDS[:, i:i+1], CLICK_ , gru_state=gru_state)
                if i > 0:
                    CLICK_ = (logit > 0.5).type(CLICKS.dtype)
                    logit_list.append(logit)
                    click_list.append(CLICK_)
            
            logits = torch.cat(logit_list, dim=1).cpu().detach().numpy().tolist()
            CLICKS_ = torch.cat(click_list, dim=1).cpu().numpy().tolist()
            CLICKS = CLICKS[:, 2:].cpu().numpy().tolist()
            assert len(CLICKS[0]) == 10
            
            for logit, CLICK_, CLICK in zip(logits, CLICKS_, CLICKS):
                file.write('{}\t{}\t{}\n'.format(str(logit), str(CLICK_), str(CLICK)))

    def generate_click_seq_cheat(self, eval_batches, file_path, file_name):
        logit_list_for_print = []
        click_list_for_print = []
        true_click_list_for_print = []
        check_path(file_path)
        data_path = os.path.join(file_path, file_name)
        file = open(data_path, 'w')
        for b_itx, batch in enumerate(eval_batches):
            if b_itx % 5000 == 0:
                self.logger.info('Generating click sequence at step: {}.'.format(b_itx))
            QIDS = Variable(torch.from_numpy(np.array(batch['qids'], dtype=np.int64)))
            UIDS = Variable(torch.from_numpy(np.array(batch['uids'], dtype=np.int64)))
            VIDS = Variable(torch.from_numpy(np.array(batch['vids'], dtype=np.int64)))
            CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64))[:, :-1])
            true_clicks = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64)))
            if use_cuda:
                QIDS, UIDS, VIDS, CLICKS, true_clicks = QIDS.cuda(), UIDS.cuda(), VIDS.cuda(), CLICKS.cuda(), true_clicks.cuda()

            self.model.eval()
            pred_logits, _ = self.model(QIDS, UIDS, VIDS, CLICKS)
            pred_clicks = (pred_logits > 0.5).type(true_clicks.dtype).cpu().numpy().tolist()
            pred_logits = pred_logits.detach().cpu().numpy().tolist()
            true_clicks = true_clicks[:, 2:].cpu().numpy().tolist()
            
            for logit, pred_click, true_click in zip(pred_logits, pred_clicks, true_clicks):
                file.write('{}\t{}\t{}\n'.format(str(logit), str(pred_click), str(true_click)))

    def save_model(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        torch.save(self.model.state_dict(), os.path.join(model_dir, model_prefix+'_{}.model'.format(self.global_step)))
        torch.save(self.optimizer.state_dict(), os.path.join(model_dir, model_prefix + '_{}.optimizer'.format(self.global_step)))
        self.logger.info('Model and optimizer saved in {}, with prefix {} and global step {}.'.format(model_dir, model_prefix, self.global_step))

    def load_model(self, model_dir, model_prefix, global_step):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        optimizer_path = os.path.join(model_dir, model_prefix + '_{}.optimizer'.format(global_step))
        if not os.path.isfile(optimizer_path):
            optimizer_path = os.path.join(model_dir, model_prefix + '_best_{}.optimizer'.format(global_step))
        if os.path.isfile(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path))
            self.logger.info('Optimizer restored from {}, with prefix {} and global step {}.'.format(model_dir, model_prefix, global_step))
        model_path = os.path.join(model_dir, model_prefix + '_{}.model'.format(global_step))
        if not os.path.isfile(model_path):
            model_path = os.path.join(model_dir, model_prefix + '_best_{}.model'.format(global_step))
        if use_cuda:
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        self.logger.info('Model restored from {}, with prefix {} and global step {}.'.format(model_dir, model_prefix, global_step))
