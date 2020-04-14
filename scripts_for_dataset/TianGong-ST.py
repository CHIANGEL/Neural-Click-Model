# !/usr/bin/python
# coding: utf8

from xml.dom.minidom import parse
import xml.dom.minidom
import time
import pprint
import string
import sys
sys.path.append("..")
import argparse
import re
import os
import numpy as np
import torch
import torch.nn as nn
from utils import *

def xml_clean(args):
    # open xml file reader & writer
    xml_reader = open(os.path.join(args.input, args.dataset), 'r')
    xml_writer = open(os.path.join(args.input, 'clean-' + args.dataset), 'w')
    # print(xml_reader)
    # print(xml_writer)

    # remove useless lines
    read_line_count = 0
    removed_line_count = 0
    interaction_count = 0
    print('  - {}'.format('start reading from xml file...'))
    xml_lines = xml_reader.readlines()
    print('  - {}'.format('read {} lines'.format(len(xml_lines))))
    print('  - {}'.format('start removing useless lines...'))
    for xml_line in xml_lines:
        # print(xml_line, end='')
        read_line_count += 1
        if xml_line.find('<interaction num=') != -1:
            interaction_count += 1
        if xml_line_removable(xml_line):
            # A line that should be removed
            removed_line_count += 1
            if removed_line_count % 1000000 == 0:
                print('  - {}'.format('remove {} lines...'.format(removed_line_count)))
        else:
            xml_writer.write(xml_line)
    
    # It is guaranteed that there are 10 docs for each query
    assert read_line_count == len(xml_lines)
    assert removed_line_count == interaction_count + interaction_count * 10 * (1 + 1 + 2 + 6)
    print('  - {}'.format('read {} lines'.format(read_line_count)))
    print('  - {}'.format('totally {} iteractions'.format(interaction_count)))
    print('  - {}'.format('totally remove {} lines'.format(removed_line_count)))
    args.dataset = 'clean-' + args.dataset

def generate_dict_list(args):
    punc = '\\~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
    session_sid = {}
    query_qid = {}
    url_uid = {}
    uid_description = {}

    print('  - {}'.format('start parsing xml file...'))
    DOMTree = xml.dom.minidom.parse(os.path.join(args.input, args.dataset))
    tiangong2018 = DOMTree.documentElement
    sessions = tiangong2018.getElementsByTagName('session')
        
    # generate infos_per_session
    print('  - {}'.format('generating infos_per_session...'))
    infos_per_session = []
    junk_interation_num = 0
    for session in sessions:
        info_per_session = {}
        # get the session id
        session_number = int(session.getAttribute('num'))
        if not (session_number in session_sid):
            session_sid[session_number] = len(session_sid)
        info_per_session['session_number'] = session_number
        info_per_session['sid'] = session_sid[session_number]
        # print('session: {}'.format(session_number))
        
        # Get information within a query
        interactions = session.getElementsByTagName('interaction')
        interaction_infos = []
        for interaction in interactions:
            interaction_info = {}
            interaction_number = int(interaction.getAttribute('num'))
            query_id = interaction.getElementsByTagName('query_id')[0].childNodes[0].data
            if not (query_id in query_qid):
                query_qid[query_id] = len(query_qid)
            interaction_info['query'] = query_id
            interaction_info['qid'] = query_qid[query_id]
            # print('interaction: {}'.format(interaction_number))
            # print('query_id: {}'.format(query_id))

            # Get document infomation
            docs = interaction.getElementsByTagName('results')[0].getElementsByTagName('result')
            doc_infos = []
            if len(docs) == 0:
                print('  - {}'.format('WARNING: find a query with no docs: {}'.format(query)))
                junk_interation_num += 1
                continue
            elif len(docs) > 10:
                # more than 10 docs is not ok. May cause index out-of-range in embeddings
                print('  - {}'.format('WARNING: find a query with more than 10 docs: {}'.format(query)))
                junk_interation_num += 1
                continue
            elif len(docs) < 10:
                # less than 10 docs is ok. Never cause index out-of-range in embeddings
                print('  - {}'.format('WARNING: find a query with less than 10 docs: {}'.format(query)))
                junk_interation_num += 1
                continue
            for doc in docs:
                # WARNING: there might be junk data in TianGong-ST (e.g. rank > 10),  so we use manual doc_rank here
                doc_rank = int(doc.getAttribute('rank'))
                assert 1 <= doc_rank and doc_rank <= 10
                doc_id = doc.getElementsByTagName('docid')[0].childNodes[0].data
                if not (doc_id in url_uid):
                    url_uid[doc_id] = len(url_uid)
                doc_info = {}
                doc_info['rank'] = doc_rank
                doc_info['url'] = doc_id
                doc_info['uid'] = url_uid[doc_id]
                doc_info['click'] = 0
                doc_infos.append(doc_info)
                # print('      doc ranks at {}: {}'.format(doc_rank, doc_id))

            # Get click information if there are clicked docs
            # Maybe there are no clicks in this query
            clicks = interaction.getElementsByTagName('clicked')
            if len(clicks) > 0:
                clicks = clicks[0].getElementsByTagName('click')
                for click in clicks:
                    clicked_doc_rank = int(click.getElementsByTagName('rank')[0].childNodes[0].data)
                    for item in doc_infos:
                        if item['rank'] == clicked_doc_rank:
                            item['click'] = 1
                            break
                    # print('      click doc ranked at {}'.format(clicked_doc_rank))
            else:
                pass
                # print('      click nothing')
            interaction_info['docs'] = doc_infos
            interaction_info['uids'] = [doc['uid'] for doc in doc_infos]
            interaction_info['clicks'] = [doc['click'] for doc in doc_infos]
            interaction_infos.append(interaction_info)
        info_per_session['interactions'] = interaction_infos
        infos_per_session.append(info_per_session)
    print('  - {}'.format('abandon {} junk interactions'.format(junk_interation_num)))

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
    # pprint.pprint(infos_per_session)
    # print('length of infos_per_session: {}'.format(len(infos_per_session)))
    save_list(args.output, 'infos_per_session.list', infos_per_session)
    list1 = load_list(args.output, 'infos_per_session.list')
    assert len(infos_per_session) == len(list1)
    for idx, item in enumerate(infos_per_session):
        assert item == list1[idx]
    
    # save and check infos_per_query
    print('  - {}'.format('save and check infos_per_query...'))
    print('    - {}'.format('length of infos_per_query: {}'.format(len(infos_per_query))))
    # pprint.pprint(infos_per_query)
    # print('length of infos_per_query: {}'.format(len(infos_per_query)))
    save_list(args.output, 'infos_per_query.list', infos_per_query)
    list2 = load_list(args.output, 'infos_per_query.list')
    assert len(infos_per_query) == len(list2)
    for idx, item in enumerate(infos_per_query):
        assert item == list2[idx]
    
    # save and check dictionaries
    print('  - {}'.format('save and check session_sid, query_qid, url_uid...'))
    print('    - {}'.format('unique session number: {}'.format(len(session_sid))))
    print('    - {}'.format('unique query number: {}'.format(len(query_qid))))
    print('    - {}'.format('unique doc number: {}'.format(len(url_uid))))
    save_dict(args.output, 'session_sid.dict', session_sid)
    save_dict(args.output, 'query_qid.dict', query_qid)
    save_dict(args.output, 'url_uid.dict', url_uid)

    dict1 = load_dict(args.output, 'session_sid.dict')
    dict2 = load_dict(args.output, 'query_qid.dict')
    dict3 = load_dict(args.output, 'url_uid.dict')

    assert len(session_sid) == len(dict1)
    assert len(query_qid) == len(dict2)
    assert len(url_uid) == len(dict3)

    for key in dict1:
        assert dict1[key] == session_sid[key]
        assert key > 0
    for key in dict2:
        assert dict2[key] == query_qid[key]
        assert key[0] == 'q'
        assert key[1:] != ''
    for key in dict3:
        assert dict3[key] == url_uid[key]
        assert key[0] == 'd'
        assert key[1:] != ''

    print('  - {}'.format('Done'))

def generate_data_txt(args):
    # load session_sid & query_qid & url_uid
    print('  - {}'.format('loading session_sid & query_qid & url_uid...'))
    session_sid = load_dict(args.output, 'session_sid.dict')
    query_qid = load_dict(args.output, 'query_qid.dict')
    url_uid = load_dict(args.output, 'url_uid.dict')

    # write train.txt & dev.txt & test.txt per session
    print('  - {}'.format('generating train & dev & test data per session...'))
    infos_per_session = load_list(args.output, 'infos_per_session.list')
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
    print('    - {}'.format('writing into {}train_per_session.txt'.format(args.output)))
    generate_data_per_session(infos_per_session, indices[:train_dev_split], args.output, 'train_per_session.txt')
    print('    - {}'.format('writing into {}dev_per_session.txt'.format(args.output)))
    generate_data_per_session(infos_per_session, indices[train_dev_split:dev_test_split], args.output, 'dev_per_session.txt')
    print('    - {}'.format('writing into {}test_per_session.txt'.format(args.output)))
    generate_data_per_session(infos_per_session, indices[dev_test_split:], args.output, 'test_per_session.txt')

    # write train.txt & dev.txt & test.txt per query
    print('  - {}'.format('generating train & dev & test data per query...'))
    infos_per_query = load_list(args.output, 'infos_per_query.list')
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
    print('    - {}'.format('writing into {}train_per_query.txt'.format(args.output)))
    generate_data_per_query(infos_per_query, indices[0:train_dev_split], args.output, 'train_per_query.txt')
    print('    - {}'.format('writing into {}dev_per_query.txt'.format(args.output)))
    generate_data_per_query(infos_per_query, indices[train_dev_split:dev_test_split], args.output, 'dev_per_query.txt')
    print('    - {}'.format('writing into {}test_per_query.txt'.format(args.output)))
    generate_data_per_query(infos_per_query, indices[dev_test_split:], args.output, 'test_per_query.txt')

    print('  - {}'.format('Done'))

def generate_embedding(args):
    # load query_qid & url_uid
    print('  - {}'.format('loading query_qid & url_uid...'))
    query_qid = load_dict(args.output, 'query_qid.dict')
    url_uid = load_dict(args.output, 'url_uid.dict')
    unique_query_num = len(query_qid)
    unique_doc_num = len(url_uid)

    # load infos_per_query
    print('  - {}'.format('loading infos_per_query...'))
    infos_per_query = load_list(args.output, 'infos_per_query.list')
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
    torch.save(query_embedding_QDQD, '{}query_embedding.emb'.format(args.output))
    
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
    torch.save(doc_embedding_QDQD, '{}doc_embedding.emb'.format(args.output))

def generate_data_for_pyclick(args):
    # load infos_per_session
    pyclick_data_writer = open(os.path.join(args.output, 'TianGong-ST_for_Pyclick'), 'w')
    session_count = 0
    interaction_count = 0
    print('  - {}'.format('generating data for repo Pyclick...'))
    infos_per_session = load_list(args.output, 'infos_per_session.list')
    # pprint.pprint(infos_per_session[:3])
    for info_per_session in infos_per_session:
        session_count += 1
        sid = info_per_session['sid']
        interactions = info_per_session['interactions']
        for interaction in interactions:
            interaction_count += 1
            qid = interaction['qid']
            uids = interaction['uids']
            clicks = interaction['clicks']
            str_per_line = []
            str_per_line.append(str(sid))
            str_per_line.append('0')
            str_per_line.append('Q')
            str_per_line.append(str(qid))
            str_per_line.append('0')
            str_per_line += [str(uid) for uid in uids]
            pyclick_data_writer.write('\t'.join(str_per_line) + '\n')
            for click, uid in zip(clicks, uids):
                if click:
                    str_per_line = []
                    str_per_line.append(str(sid))
                    str_per_line.append('0')
                    str_per_line.append('C')
                    str_per_line.append(str(uid))
                    pyclick_data_writer.write('\t'.join(str_per_line) + '\n')
    print('    - {}'.format('{} sessions in data for Pyclick'.format(session_count)))
    print('    - {}'.format('{} interactions in data for Pyclick'.format(interaction_count)))
    pyclick_data_writer.close()

def main():
    parser = argparse.ArgumentParser('TianGong-ST')
    parser.add_argument('--dataset', default='sogousessiontrack2018.xml',
                        help='dataset name')
    parser.add_argument('--input', default='../../dataset/TianGong-ST/data/',
                        help='input path')
    parser.add_argument('--output', default='../data/TianGong-ST/',
                        help='output path')
    parser.add_argument('--xml_clean', action='store_true',
                        help='remove useless lines in xml files, to reduce the size of xml file')
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
    parser.add_argument('--pyclick', action='store_true',
                        help='generate data files for repo Pyclick')
    args = parser.parse_args()
    if args.xml_clean:
        # remove useless lines in xml files, to reduce the size of xml file
        print('===> {}'.format('cleaning xml file...'))
        xml_clean(args)
    if args.dict_list:
        # generate info_per_session & info_per_query
        print('===> {}'.format('generating dicts and lists...'))
        generate_dict_list(args)
    if args.data_txt:
        # load lists saved by generate_dict_list() and generates train.txt & dev.txt & test.txt
        print('===> {}'.format('generating train & dev & test data txt...'))
        generate_data_txt(args)
    if args.embedding:
        # generate embeddings of query, document and interaction for NCM infos_per_query
        print('===> {}'.format('generating embeddings for infos_per_query...'))
        generate_embedding(args)
    if args.pyclick:
        # generate data files for repo Pyclick
        print('===> {}'.format('generating data files for repo Pyclick...'))
        generate_data_for_pyclick(args)
    print('===> {}'.format('Done.'))
    
if __name__ == '__main__':
    main()