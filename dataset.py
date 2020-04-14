# coding: utf8

import glob
import os
import json
import logging
import math
import numpy as np
import pprint

from utils import *

class Dataset(object):
    """
     This module implements the APIs for loading the dataset
    """
    def __init__(self, args, train_dirs=[], dev_dirs=[], test_dirs=[], isRank=False):
        # config settings
        self.logger = logging.getLogger("neural_click_model")
        self.max_doc_num = args.max_doc_num
        self.gpu_num = args.gpu_num
        self.args = args
        self.num_train_files = args.num_train_files
        self.num_dev_files = args.num_dev_files
        self.num_test_files = args.num_test_files

        # load pre-proccessed dicts & lists
        self.query_qid = load_dict('data/' + args.dataset, 'query_qid.dict')
        self.url_uid = load_dict('data/' + args.dataset, 'url_uid.dict')
        self.infos_per_query = load_list('data/' + args.dataset, 'infos_per_query.list')

        # load train & dev & test data
        self.train_set, self.dev_set, self.test_set = [], [], []
        if isRank:
            if train_dirs:
                for train_dir in train_dirs:
                    self.train_set += self.load_dataset_rank(train_dir, num=self.num_train_files, mode='train')
                self.logger.info('Train set size: {} queries.'.format(len(self.train_set)))
            if dev_dirs:
                for dev_dir in dev_dirs:
                    self.dev_set += self.load_dataset_rank(dev_dir, num=self.num_dev_files, mode='dev')
                self.logger.info('Dev set size: {} queries.'.format(len(self.dev_set)))
            if test_dirs:
                for test_dir in test_dirs:
                    self.test_set += self.load_dataset_rank(test_dir, num=self.num_test_files, mode='test')
                self.logger.info('Test set size: {} queries.'.format(len(self.test_set)))
        else:
            if train_dirs:
                for train_dir in train_dirs:
                    self.train_set += self.load_dataset(train_dir, num=self.num_train_files, mode='train')
                self.logger.info('Train set size: {} queries.'.format(len(self.train_set)))
            if dev_dirs:
                for dev_dir in dev_dirs:
                    self.dev_set += self.load_dataset(dev_dir, num=self.num_dev_files, mode='dev')
                self.logger.info('Dev set size: {} queries.'.format(len(self.dev_set)))
            if test_dirs:
                for test_dir in test_dirs:
                    self.test_set += self.load_dataset(test_dir, num=self.num_test_files, mode='test')
                self.logger.info('Test set size: {} queries.'.format(len(self.test_set)))

    def load_dataset(self, file_path, num, mode):
        """
         Load the dataset for training
        """
        data_set = []
        file = open(file_path, 'r')
        query_infos = file.read().strip().split('\n\n')
        for query_info in query_infos:
            qid = 0
            uids, clicks = [], []
            lines = query_info.strip().split('\n')
            for line in lines:
                attrs = line.strip().split('\t')
                qids = eval(attrs[0].strip()) # qids is a list
                doc_info = eval(attrs[1].strip())
                uids.append(doc_info[0])
                clicks.append(doc_info[2])
            data_set.append({'qids': qids, 
                             'uids': uids,
                             'clicks': clicks})
        return data_set

    def load_dataset_rank(self, data_path, num, mode):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        data_set = []
        files = sorted(glob.glob(data_path + '/part-*'))
        if num > 0:
            files = files[0:num]
        for fn in files:
            # print fn
            lines = open(fn).readlines()
            for line in lines:
                attrs = line.strip().split('\t')
                session_id = attrs[0]
                if session_id not in self.train_session_id:
                    continue
                query = attrs[1].strip().lower()
                if query not in self.query_qid:
                    self.query_qid[query] = len(self.query_qid)
                    self.qid_query[self.query_qid[query]] = query
                qid = self.query_qid[query]

                urls = [url.encode('utf-8', 'ignore') for url in json.loads(attrs[4])]
                if len(urls) < self.max_d_num:
                    continue
                urls = urls[:self.max_d_num]
                vtypes = [vtype.encode('utf-8', 'ignore') for vtype in json.loads(attrs[5])][:self.max_d_num]
                # clicks = json.loads(attrs[6])[:self.max_d_num]
                for curr_url, curr_vtype in zip(urls, vtypes):
                    clicks = [0, 0, 0]
                    qids = [qid, qid]
                    uids = [0]
                    if curr_url not in self.url_uid:
                        self.url_uid[curr_url] = len(self.url_uid)
                        self.uid_url[self.url_uid[curr_url]] = curr_url
                    uids.append(self.url_uid[curr_url])
                    vids = [0]
                    if curr_vtype not in self.vtype_vid:
                        self.vtype_vid[curr_vtype] = len(self.vtype_vid)
                        self.vid_vtype[self.vtype_vid[curr_vtype]] = curr_vtype
                    vids.append(self.vtype_vid[curr_vtype])

                    if qid not in self.qid_uid_set:
                        self.qid_uid_set[qid] = {}
                    if uids[-1] not in self.qid_uid_set[qid]:
                        self.qid_uid_set[qid][uids[-1]] = 0
                    else:
                        continue
                    data_set.append({'session_id': session_id,
                                     'qids':qids, 'query': query,
                                     'uids': uids, 'urls': ['', curr_url],
                                     'vids': vids, 'vtypes': ['', curr_vtype],
                                     'clicks': clicks})
        return data_set

    def _one_mini_batch(self, data, indices):
        """
         Return one mini batch
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'qids': [],
                      'uids': [],
                      'vids': [],
                      'clicks': []}
        for idx, sample in enumerate(batch_data['raw_data']):
            batch_data['qids'].append(sample['qids'])
            batch_data['uids'].append(sample['uids'])
            batch_data['clicks'].append(sample['clicks'])
        return batch_data

    def gen_mini_batches(self, set_name, batch_size, shuffle=True):
        """
         Generate data batches for a specific dataset (train/dev/test)
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
            assert batch_size == 1
        elif set_name == 'test':
            data = self.test_set
            assert batch_size == 1
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        indices = indices.tolist()
        # Because batch_size % gpu_num == 0
        # So that len(data) % batch_size == 0  <=> len(data) % gpu_num == 0
        indices += indices[0: (batch_size - data_size % batch_size) % batch_size]
        assert len(indices) % batch_size == 0
        for batch_start in np.arange(0, len(list(indices)), batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices)
