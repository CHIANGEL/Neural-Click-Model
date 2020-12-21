import os
import json
import logging
import numpy as np
import utils

class Dataset(object):
    """
    This module implements the APIs for loading dataset and providing batch data
    """
    def __init__(self, args):
        self.logger = logging.getLogger("NCM")
        self.max_d_num = args.max_d_num
        self.gpu_num = args.gpu_num
        self.dataset = args.dataset
        self.data_dir = os.path.join('data', self.dataset)
        self.args = args
        
        self.train_set = self.load_dataset(os.path.join('data', self.dataset, 'train_per_query_quid.txt'), mode='train')
        self.valid_set = self.load_dataset(os.path.join('data', self.dataset, 'valid_per_query_quid.txt'), mode='valid')
        self.test_set = self.load_dataset(os.path.join('data', self.dataset, 'test_per_query_quid.txt'), mode='test')
        self.label_set = self.load_dataset(os.path.join('data', self.dataset, 'human_label_for_NCM_per_query_quid.txt'), mode='label')
        self.trainset_size = len(self.train_set)
        self.validset_size = len(self.valid_set)
        self.testset_size = len(self.test_set)
        self.labelset_size = len(self.label_set)

        self.query_qid = utils.load_dict(self.data_dir, 'query_qid.dict')
        self.url_uid = utils.load_dict(self.data_dir, 'url_uid.dict')
        self.vtype_vid = utils.load_dict(self.data_dir, 'vtype_vid.dict')
        self.query_size = len(self.query_qid)
        self.doc_size = len(self.url_uid)
        self.vtype_size = len(self.vtype_vid)

        self.logger.info('Train set size: {} queries.'.format(len(self.train_set)))
        self.logger.info('Dev set size: {} queries.'.format(len(self.valid_set)))
        self.logger.info('Test set size: {} queries.'.format(len(self.test_set)))
        self.logger.info('Label set size: {} queries.'.format(len(self.label_set)))
        self.logger.info('Unique query num, including zero vector: {}'.format(self.query_size))
        self.logger.info('Unique doc num, including zero vector: {}'.format(self.doc_size))
        self.logger.info('Unique vtype num, including zero vector: {}'.format(self.vtype_size))

    def load_dataset(self, data_path, mode):
        """
        Loads the dataset
        """
        data_set = []
        lines = open(data_path).readlines()
        for line in lines:
            # WARNING: '\t' is necessary, because there are spaces in the string-list that should not be splited.
            attr = line.strip().split('\t')
            sid = int(attr[0])
            qid = int(attr[1].strip())
            qids = [qid for _ in range(self.max_d_num + 1)]
            uids = [0] + json.loads(attr[2].strip())
            vids = [0] + json.loads(attr[3].strip())
            clicks = [0, 0] + json.loads(attr[4].strip())
            relevances = json.loads(attr[5].strip()) if mode == 'label' else [0 for _ in range(self.max_d_num)]
            data_set.append({'sid': sid,
                            'qids': qids,
                            'uids': uids,
                            'vids': vids,
                            'clicks': clicks, 
                            'relevances': relevances,})
        return data_set

    def _one_mini_batch(self, data, indices):
        """
        Get one mini batch data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                        'qids': [],
                        'uids': [],
                        'vids': [],
                        'clicks': [],
                        'relevances': [],
                        'true_clicks': [],}
        for sidx, sample in enumerate(batch_data['raw_data']):
            batch_data['qids'].append(sample['qids'])
            batch_data['uids'].append(sample['uids'])
            batch_data['vids'].append(sample['vids'])
            batch_data['clicks'].append(sample['clicks'])
            batch_data['relevances'].append(sample['relevances'])
            batch_data['true_clicks'].append(sample['clicks'][2:])
        return batch_data

    def gen_mini_batches(self, set_name, batch_size, shuffle=True):
        """
        Generate data batches for a specific dataset (train/valid/test)
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'valid':
            data = self.valid_set
        elif set_name == 'test':
            data = self.test_set
        elif set_name == 'label':
            data = self.label_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)

        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        indices = indices.tolist()

        # alignment for multi-gpu cases
        indices += indices[:(self.gpu_num - data_size % self.gpu_num) % self.gpu_num]
        for batch_start in np.arange(0, len(list(indices)), batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices)