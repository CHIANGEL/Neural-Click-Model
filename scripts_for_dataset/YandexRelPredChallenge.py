# !/usr/bin/python
# coding: utf8

from xml.dom.minidom import parse
import xml.dom.minidom
import pprint
import string
import argparse
import re
import os
import numpy as np
import torch
import torch.nn as nn
from utils import *

def generate_dict_list(args):
    query_qid = {}
    url_uid = {}
    sessions_file = open(args.dataset, "r")
        
    # generate infos_per_session
    print('  - {}'.format('generating infos_per_session...'))
    infos_per_session = []
    interaction_infos = []
    interaction_info = {}
    info_per_session = {}
    current_session_id = -1
    first_query = True
    junk_interation_num = 0
    for line in sessions_file: 
        entry_array = line.strip().split("\t")
        # new session, clear list and dict
        if int(entry_array[0]) != current_session_id:
            if current_session_id > -1:
                interaction_infos.append(interaction_info)
                info_per_session['interactions'] = interaction_infos
                infos_per_session.append(info_per_session)
            info_per_session = {} 
            interaction_infos = []
            current_session_id = int(entry_array[0])
            info_per_session['session_id'] = current_session_id
            first_query =  True
        # If the entry has 6 or more elements, it is a query
        if len(entry_array) >= 6 and entry_array[2] == 'Q':
            if first_query == False:
                interaction_infos.append(interaction_info)
            first_query = False
            interaction_info = {}
            # process query_qid
            query = int(entry_array[3])
            if not (query in query_qid):
                query_qid[query] = len(query_qid)
            interaction_info['query'] = query
            interaction_info['qid'] = query_qid[query]
            # process url_uid
            docs = [int(item) for item in entry_array[5:]]
            for doc in docs:
                if not (doc in url_uid):
                    url_uid[doc] = len(url_uid)
            interaction_info['docs'] = docs
            interaction_info['uids'] = [url_uid[doc] for doc in docs]
            # process clicks
            interaction_info['clicks'] = [0 for _ in entry_array[5:]]
        # If the entry has 4 elements it is a click
        elif len(entry_array) == 4 and entry_array[2] == "C":
            if int(entry_array[0]) == info_per_session['session_id']:
                clicked_uid = url_uid[int(entry_array[3])]
                if clicked_uid in interaction_info['uids']:
                    index = interaction_info['uids'].index(clicked_uid)
                    interaction_info['clicks'][index] = 1
        # Else it is an unknown data format so leave it out
        else:
            continue
    # append the last info_per_session
    interaction_infos.append(interaction_info)
    info_per_session['interactions'] = interaction_infos
    infos_per_session.append(info_per_session)

    # generate infos_per_query
    print('  - {}'.format('generating infos_per_query...'))
    infos_per_query = []
    for info_per_session in infos_per_session:
        interaction_infos = info_per_session['interactions']
        for interaction_info in interaction_infos:
            infos_per_query.append(interaction_info)

    # save and check infos_per_session
    print('  - {}'.format('save and check infos_per_session...'))
    print('    - {}'.format('length of infos_per_session: {}'.format(len(infos_per_session))))
    save_list('data/Yandex/', 'infos_per_session.list', infos_per_session)
    list1 = load_list('data/Yandex/', 'infos_per_session.list')
    assert len(infos_per_session) == len(list1)
    for idx, item in enumerate(infos_per_session):
        assert item == list1[idx]

    # save and check infos_per_query
    print('  - {}'.format('save and check infos_per_query...'))
    print('    - {}'.format('length of infos_per_query: {}'.format(len(infos_per_query))))
    save_list('data/Yandex/', 'infos_per_query.list', infos_per_query)
    list2 = load_list('data/Yandex/', 'infos_per_query.list')
    assert len(infos_per_query) == len(list2)
    for idx, item in enumerate(infos_per_query):
        assert item == list2[idx]
    
    # save and check dictionaries   
    print('  - {}'.format('save and check query_qid, url_uid...'))
    print('    - {}'.format('unique query number: {}'.format(len(query_qid))))
    print('    - {}'.format('unique doc number: {}'.format(len(url_uid))))
    save_dict('data/Yandex/', 'query_qid.dict', query_qid)
    save_dict('data/Yandex/', 'url_uid.dict', url_uid)
    dict1 = load_dict('data/Yandex', 'query_qid.dict')
    dict2 = load_dict('data/Yandex', 'url_uid.dict')
    assert len(query_qid) == len(dict1)
    assert len(url_uid) == len(dict2)
    for key in query_qid:
        assert dict1[key] == query_qid[key]
        assert type(key) == type(1)
    for key in url_uid:
        assert dict2[key] == url_uid[key]
        assert type(key) == type(1)
    
    print('  - {}'.format('Done'))

def generate_data_txt(args):
    # load query_qid & url_uid
    print('  - {}'.format('loading query_qid & url_uid...'))
    query_qid = load_dict('data/Yandex', 'query_qid.dict')
    url_uid = load_dict('data/Yandex', 'url_uid.dict')

    # write train.txt & dev.txt & test.txt per session
    print('  - {}'.format('generating train & dev & test data per session...'))
    infos_per_session = load_list('data/Yandex/', 'infos_per_session.list')
    # Separate all sessions into train : dev : test
    session_num = len(infos_per_session)
    train_dev_split = int(session_num * args.trainset_ratio)
    dev_test_split = int(session_num * (args.devset_ratio + args.trainset_ratio))
    # train_sessions = infos_per_session[:train_dev_split]
    # dev_sessions = infos_per_session[train_dev_split:dev_test_split]
    # test_sessions = infos_per_session[dev_test_split:]
    train_session_num = train_dev_split
    dev_session_num = dev_test_split - train_dev_split
    test_session_num = session_num - dev_test_split
    print('    - {}'.format('train sessions: {}'.format(train_session_num)))
    print('    - {}'.format('dev sessions: {}'.format(dev_session_num)))
    print('    - {}'.format('test sessions: {}'.format(test_session_num)))
    print('    - {}'.format('total sessions: {}'.format(session_num)))
    indices = np.arange(0, session_num)
    # np.random.shuffle(indices)
    print('    - {}'.format('writing into data/Yandex/train_per_session.txt'))
    generate_data_per_session(infos_per_session, indices[:train_dev_split], 'data/Yandex/', 'train_per_session.txt')
    print('    - {}'.format('writing into data/Yandex/dev_per_session.txt'))
    generate_data_per_session(infos_per_session, indices[train_dev_split:dev_test_split], 'data/Yandex/', 'dev_per_session.txt')
    print('    - {}'.format('writing into data/Yandex/test_per_session.txt'))
    generate_data_per_session(infos_per_session, indices[dev_test_split:], 'data/Yandex/', 'test_per_session.txt')

    # write train.txt & dev.txt & test.txt per query
    print('  - {}'.format('generating train & dev & test data per query...'))
    infos_per_query = load_list('data/Yandex/', 'infos_per_query.list')
    # Separate all queries into train : dev : test
    query_num = len(infos_per_query)
    train_dev_split = int(query_num * args.trainset_ratio)
    dev_test_split = int(query_num * (args.devset_ratio + args.trainset_ratio))
    train_query_num = train_dev_split
    dev_query_num = dev_test_split - train_dev_split
    test_query_num = query_num - dev_test_split
    print('    - {}'.format('train queries: {}'.format(train_query_num)))
    print('    - {}'.format('dev queries: {}'.format(dev_query_num)))
    print('    - {}'.format('test queries: {}'.format(test_query_num)))
    print('    - {}'.format('total queries: {}'.format(query_num)))
    indices = np.arange(0, query_num)
    # np.random.shuffle(indices)
    print('    - {}'.format('writing into data/Yandex/train_per_query.txt'))
    generate_data_per_query(infos_per_query, indices[0:train_dev_split], 'data/Yandex/', 'train_per_query.txt')
    print('    - {}'.format('writing into data/Yandex/dev_per_query.txt'))
    generate_data_per_query(infos_per_query, indices[train_dev_split:dev_test_split], 'data/Yandex/', 'dev_per_query.txt')
    print('    - {}'.format('writing into data/Yandex/test_per_query.txt'))
    generate_data_per_query(infos_per_query, indices[dev_test_split:], 'data/Yandex/', 'test_per_query.txt')

    print('  - {}'.format('Done'))

def generate_embedding(args):
    # load query_qid & url_uid
    print('  - {}'.format('loading query_qid & url_uid...'))
    query_qid = load_dict('data/Yandex', 'query_qid.dict')
    url_uid = load_dict('data/Yandex', 'url_uid.dict')
    unique_query_num = len(query_qid)
    unique_doc_num = len(url_uid)

    # load infos_per_query
    print('  - {}'.format('loading infos_per_query...'))
    infos_per_query = load_list('data/Yandex/', 'infos_per_query.list')
    print('    - {}'.format('total query number: {}'.format(len(infos_per_query))))
    print('    - {}'.format('unique query number: {}'.format(unique_query_num)))
    print('    - {}'.format('unique doc number: {}'.format(unique_doc_num)))

    # generate query embeddings under rule QD+Q+D
    print('  - {}'.format('generating embeddings under rule QD+Q+D...'))
    print('    - {}'.format('generate query embeddings'))
    query_embedding_QDQD = torch.zeros(unique_query_num, 1024) # 2^10 = 1024
    for interaction_info in infos_per_query:
        qid = interaction_info['qid']
        clicks = interaction_info['clicks']
        # padding for clicks with less than 10 uids
        for i in range(10 - len(clicks)):
            clicks.append(0)
        assert len(clicks) == 10
        click_offset = 0
        for click in clicks:
            click_offset = click_offset * 2 + click
        query_embedding_QDQD[qid][click_offset] += 1
    assert query_embedding_QDQD.sum() == len(infos_per_query)
    torch.save(query_embedding_QDQD, 'data/Yandex/query_embedding.emb')
    
    # generate doc embeddings under rule QD+Q+D
    print('    - {}'.format('generate doc embeddings'))
    doc_embedding_QDQD = torch.zeros(unique_doc_num, 10240) # 2^10 * 10 = 10240
    for interaction_info in infos_per_query:
        qid = interaction_info['qid']
        uids = interaction_info['uids']
        clicks = interaction_info['clicks']
        # padding for clicks with less than 10 doc
        for i in range(10 - len(clicks)):
            clicks.append(0)
        assert len(clicks) == 10
        rank_offset = 0
        click_offset = 0
        for click in clicks:
            click_offset = click_offset * 2 + click
        for idx, uid in enumerate(uids):
            rank = idx + 1
            rank_offset = rank - 1  # rank start from 1, not 0
            doc_embedding_QDQD[uid][rank_offset * 1024 + click_offset] += 1
    assert doc_embedding_QDQD.sum() == len(infos_per_query * 10)
    torch.save(doc_embedding_QDQD, 'data/Yandex/doc_embedding.emb')

def main():
    parser = argparse.ArgumentParser('YandexRelPredChallenge')
    parser.add_argument('--dataset', default='../dataset/YandexRelPredChallenge/YandexRelPredChallenge',
                        help='dataset path')
    parser.add_argument('--dict_list', action='store_true',
                        help='generate dicts and lists for info_per_session/info_per_query')
    parser.add_argument('--data_txt', action='store_true',
                        help='generate data txt')
    parser.add_argument('--embedding', action='store_true',
                        help='generate NCM embeddings')
    parser.add_argument('--trainset_ratio', default=0.7,
                        help='ratio of the train session/query according to the total number of sessions/queries')
    parser.add_argument('--devset_ratio', default=0.15,
                        help='ratio of the dev session/query according to the total number of sessions/queries')
    args = parser.parse_args()
    if args.dict_list:
        # generate info_per_session & info_per_query
        print('===> {}'.format('generating dicts and lists...'))
        generate_dict_list(args)
    if args.data_txt:
        # load lists saved by generate_dict_list() and generates train.txt & dev.txt & test.txt
        print('===> {}'.format('generating train & dev data txt...'))
        generate_data_txt(args)
    if args.embedding:
        # generate embeddings of query, document and interaction for NCM infos_per_query
        print('===> {}'.format('generating embeddings for infos_per_query...'))
        generate_embedding(args)
    print('===> {}'.format('Done.'))
    
if __name__ == '__main__':
    main()