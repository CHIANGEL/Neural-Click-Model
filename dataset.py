# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements data process strategies.
"""
import glob
import os
import json
import logging
import math
import numpy as np

class Dataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, args, train_dirs=[], dev_dirs=[], test_dirs=[], isRank=False):
        self.logger = logging.getLogger("NCM")
        self.max_d_num = args.max_d_num
        self.gpu_num = args.gpu_num
        self.args = args
        self.num_train_files = args.num_train_files
        self.num_dev_files = args.num_dev_files
        self.num_test_files = args.num_test_files
        self.qid_query = {}
        self.uid_url = {}
        self.uid_vid = {}

        self.qid_query, self.query_qid = {0: ''}, {'': 0}
        self.uid_url, self.url_uid = {0: ''}, {'': 0}
        self.vid_vtype, self.vtype_vid = {0: ''}, {'': 0}

        self.qid_uid_set = {}
        self.train_set, self.dev_set, self.test_set = [], [], []
        if isRank:
            if train_dirs:
                for train_dir in train_dirs:
                    self.train_set += self.load_dataset_rank(train_dir, num=self.num_train_files, mode='train')
                self.logger.info('Train set size: {} sessions.'.format(len(self.train_set)))
            if dev_dirs:
                for dev_dir in dev_dirs:
                    self.dev_set += self.load_dataset_rank(dev_dir, num=self.num_dev_files, mode='dev')
                self.logger.info('Dev set size: {} sessions.'.format(len(self.dev_set)))
            if test_dirs:
                for test_dir in test_dirs:
                    self.test_set += self.load_dataset_rank(test_dir, num=self.num_test_files, mode='test')
                self.logger.info('Test set size: {} sessions.'.format(len(self.test_set)))
        else:
            if train_dirs:
                for train_dir in train_dirs:
                    self.train_set += self.load_dataset(train_dir, num=self.num_train_files, mode='train')
                self.logger.info('Train set size: {} sessions.'.format(len(self.train_set)))
            if dev_dirs:
                for dev_dir in dev_dirs:
                    self.dev_set += self.load_dataset(dev_dir, num=self.num_dev_files, mode='dev')
                self.logger.info('Dev set size: {} sessions.'.format(len(self.dev_set)))
            if test_dirs:
                for test_dir in test_dirs:
                    self.test_set += self.load_dataset(test_dir, num=self.num_test_files, mode='test')
                self.logger.info('Test set size: {} sessions.'.format(len(self.test_set)))

    def load_dataset(self, data_path, num, mode):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        data_set = []
        # files = sorted(glob.glob(data_path + '/part-*'))
        files = [data_path]
        if num > 0:
            files = files[0:num]
        for fn in files:
            # print fn
            lines = open(fn).readlines()
            for line in lines:
                attr = line.strip().split('\t')
                session_id = attr[0]
                query = int(attr[1].strip())

                urls = [url for url in json.loads(attr[4])]
                if len(urls) < self.max_d_num:
                    continue
                urls = urls[:self.max_d_num]
                vtypes = [vtype for vtype in json.loads(attr[5])][:self.max_d_num]
                clicks = json.loads(attr[6])[:self.max_d_num]
                clicks = [0, 0] + clicks
                if query not in self.query_qid:
                    self.query_qid[query] = len(self.query_qid)
                    self.qid_query[self.query_qid[query]] = query
                if query in self.query_qid:
                    qid = self.query_qid[query]
                else:
                    continue
                qids = [qid for _ in range(self.max_d_num+1)]
                uids = [0]
                for url in urls:
                    if url not in self.url_uid:
                        self.url_uid[url] = len(self.url_uid)
                        self.uid_url[self.url_uid[url]] = url
                    if url in self.url_uid:
                        uids.append(self.url_uid[url])
                    else:
                        uids.append(0)
                vids = [0]
                for vtype in vtypes:
                    if vtype not in self.vtype_vid:
                        self.vtype_vid[vtype] = len(self.vtype_vid)
                        self.vid_vtype[self.vtype_vid[vtype]] = vtype
                    if vtype in self.vtype_vid:
                        vids.append(self.vtype_vid[vtype])
                    else:
                        vids.append(0)
                for url, vtype in zip(urls, vtypes):
                    uid = self.url_uid[url]
                    vid = self.vtype_vid[vtype]
                    self.uid_vid[uid] = vid
                data_set.append({'session_id': session_id,
                                 'qids': qids, 'query': query,
                                 'uids': uids, 'urls': [''] + urls,
                                 'vids': vids, 'vtypes': [''] + vtypes,
                                 'clicks': clicks})
        return data_set

    def load_dataset_rank(self, data_path, num, mode):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        data_set = []
        # files = sorted(glob.glob(data_path + '/part-*'))
        files = [data_path]
        if num > 0:
            files = files[0:num]
        for fn in files:
            # print fn
            lines = open(fn).readlines()
            for line in lines:
                attr = line.strip().split('\t')
                session_id = attr[0]
                query = attr[1].strip()
                if query not in self.query_qid:
                    self.query_qid[query] = len(self.query_qid)
                    self.qid_query[self.query_qid[query]] = query
                qid = self.query_qid[query]

                urls = [url for url in json.loads(attr[4])]
                if len(urls) < self.max_d_num:
                    continue
                urls = urls[:self.max_d_num]
                vtypes = [vtype for vtype in json.loads(attr[5])][:self.max_d_num]
                # clicks = json.loads(attr[6])[:self.max_d_num]
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
                                     'qids': qids, 'query': query,
                                     'uids': uids, 'urls': ['', curr_url],
                                     'vids': vids, 'vtypes': ['', curr_vtype],
                                     'clicks': clicks})
        return data_set

    def _one_mini_batch(self, data, indices):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'qids': [],
                      'uids': [],
                      'vids': [],
                      'clicks': []}
        for sidx, sample in enumerate(batch_data['raw_data']):
            batch_data['qids'].append(sample['qids'])
            batch_data['uids'].append(sample['uids'])
            batch_data['vids'].append(sample['vids'])
            batch_data['clicks'].append(sample['clicks'])
        return batch_data

    def gen_mini_batches(self, set_name, batch_size, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)

        # data format
        # {'session_id': session_id, 'qids': qids, 'query': query,'uids': uids,
        # 'urls': [''] + urls,'vids': vids, 'vtypes': [''] + vtypes, 'clicks': clicks}

        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        indices = indices.tolist()

        # 多GPU并行的样子？厉害了。。。
        indices += indices[:(self.gpu_num - data_size % self.gpu_num)%self.gpu_num]
        for batch_start in np.arange(0, len(list(indices)), batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            # yield 返回一个迭代器
            yield self._one_mini_batch(data, batch_indices)
