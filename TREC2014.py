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
from utils import *

def preprocess(args):
    punc = '\\~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
    query_qid, qid_query = {}, {}
    url_uid, uid_url = {}, {}
    # click_num = 0

    DOMTree = xml.dom.minidom.parse(args.dataset)
    sessiontrack2014 = DOMTree.documentElement
    sessions = sessiontrack2014.getElementsByTagName('session')
        
    # generate infos_per_session
    print('===> {}'.format('generating infos_per_session...'))
    infos_per_session = []
    for session in sessions:
        info_per_session = {}
        session_number = int(session.getAttribute('num'))
        info_per_session['session_number'] = session_number
        interactions = session.getElementsByTagName('interaction')
        interaction_infos = []
        # print('session: {}'.format(session_number))
        
        # Get information within a query
        for interaction in interactions:
            interaction_info = {}
            interaction_number = int(interaction.getAttribute('num'))
            query = interaction.getElementsByTagName('query')[0].childNodes[0].data
            if not (query in query_qid):
                query_qid[query] = len(query_qid)
                qid_query[query_qid[query]] = query
            interaction_info['query'] = query
            interaction_info['qid'] = query_qid[query]
            docs = interaction.getElementsByTagName('results')[0].getElementsByTagName('result')
            doc_infos = []

            # print('   interaction: {}'.format(interaction_number))
            # print('      query: {}'.format(query))
            # Get document infomation
            doc_rank = 0
            for doc in docs:
                # WARNING: there are junk data in TREC2014 (e.g. rank > 10),  so we use manual doc_rank here
                doc_rank += 1 
                doc_url = doc.getElementsByTagName('url')[0].childNodes[0].data
                if not (doc_url in url_uid):
                    url_uid[doc_url] = len(url_uid)
                    uid_url[url_uid[doc_url]] = doc_url
                try:
                    doc_title = doc.getElementsByTagName('title')[0].childNodes[0].data
                except:
                    doc_title = ''
                try:
                    doc_snippet = doc.getElementsByTagName('snippet')[0].childNodes[0].data
                except:
                    doc_snippet = ''
                # text cleaning
                doc_description = doc_title + doc_snippet
                doc_description = re.sub(r"[%s]+" %punc, "", doc_description) 
                doc_description = ' '.join(map(lambda x: x.strip().lower(), doc_description.split()))
                doc_info = {}
                doc_info['rank'] = doc_rank
                doc_info['original_rank'] = int(doc.getAttribute('rank'))
                doc_info['url'] = doc_url
                doc_info['uid'] = url_uid[doc_url]
                doc_info['description'] = doc_description
                doc_info['click'] = 0
                doc_infos.append(doc_info)
                # print('      doc ranks at {}: {}'.format(doc_rank, doc_url))

            # Get click information if there are clicked docs
            # Maybe there are no clicks in this query
            clicks = interaction.getElementsByTagName('clicked')
            if len(clicks) > 0:
                clicks = clicks[0].getElementsByTagName('click')
                for click in clicks:
                    clicked_doc_rank = int(click.getElementsByTagName('rank')[0].childNodes[0].data)
                    for item in doc_infos:
                        if item['original_rank'] == clicked_doc_rank:
                            item['click'] = 1
                            break
                    # print('      click doc ranked at {}'.format(clicked_doc_rank))
                    # click_num += 1
            else:
                pass
                # print('      click nothing')
            interaction_info['docs'] = doc_infos
            interaction_infos.append(interaction_info)
        info_per_session['interactions'] = interaction_infos
        infos_per_session.append(info_per_session)

    # generate infos_per_query
    print('===> {}'.format('generating infos_per_query...'))
    infos_per_query = []
    for info_per_session in infos_per_session:
        interaction_infos = info_per_session['interactions']
        for interaction_info in interaction_infos:
            infos_per_query.append(interaction_info)

    # save and check infos_per_session
    print('===> {}'.format('save and check infos_per_session...'))
    # pprint.pprint(infos_per_session)
    # print('length of infos_per_session: {}'.format(len(infos_per_session)))
    save_list('data/', 'infos_per_session.list', infos_per_session)
    list1 = load_list('data/', 'infos_per_session.list')
    assert len(infos_per_session) == len(list1)
    for idx, item in enumerate(infos_per_session):
        assert item == list1[idx]

    # save and check infos_per_query
    print('===> {}'.format('save and check infos_per_query...'))
    # pprint.pprint(infos_per_query)
    # print('length of infos_per_query: {}'.format(len(infos_per_query)))
    save_list('data/', 'infos_per_query.list', infos_per_query)
    list2 = load_list('data/', 'infos_per_query.list')
    assert len(infos_per_query) == len(list2)
    for idx, item in enumerate(infos_per_query):
        assert item == list2[idx]

    # save and check dictionaries
    print('===> {}'.format('save and check query_qid, qid_query, url_uid, uid_url...'))
    save_dict('data/dict/', 'query_qid.dict', query_qid)
    save_dict('data/dict/', 'qid_query.dict', qid_query)
    save_dict('data/dict/', 'url_uid.dict', url_uid)
    save_dict('data/dict/', 'uid_url.dict', uid_url)

    dict1 = load_dict('data/dict', 'query_qid.dict')
    dict2 = load_dict('data/dict', 'qid_query.dict')
    dict3 = load_dict('data/dict', 'url_uid.dict')
    dict4 = load_dict('data/dict', 'uid_url.dict')

    assert len(query_qid) == len(dict1)
    assert len(qid_query) == len(dict2)
    assert len(url_uid) == len(dict3)
    assert len(uid_url) == len(dict4)

    for key in dict1:
        assert type(dict1[key]) == type(1)
    for key in dict2:
        assert type(dict2[key]) == type('1')
    for key in dict3:
        assert type(dict3[key]) == type(1)
    for key in dict4:
        assert type(dict4[key]) == type('1')

    print('===> {}'.format('Done'))

def reload(args):
    query_qid = load_dict('data/dict', 'query_qid.dict')
    qid_query = load_dict('data/dict', 'qid_query.dict')
    url_uid = load_dict('data/dict', 'url_uid.dict')
    uid_url = load_dict('data/dict', 'uid_url.dict')

    # write train_session.txt & dev_session.txt
    if args.generate_type == 'per_session':
        infos_per_session = load_list('data/', 'infos_per_session.list')
        # Separate all sessions into train:dev
        session_num = len(infos_per_session)
        train_session_num = int(session_num * args.trainset_ratio)
        dev_session_num = session_num - train_session_num
        # print('{}, {}, {}'.format(session_num, train_session_num, dev_session_num))
        indices = np.arange(0, session_num)
        np.random.shuffle(indices)
        generate_data_per_session(infos_per_session, indices[0: train_session_num], 'data/', 'train_per_session.txt')
        generate_data_per_session(infos_per_session, indices[train_session_num: session_num], 'data/', 'dev_per_session.txt')
    elif args.generate_type == 'per_query':
        infos_per_query = load_list('data/', 'infos_per_query.list')
        # Separate all sessions into train:dev
        query_num = len(infos_per_query)
        train_query_num = int(query_num * args.trainset_ratio)
        dev_query_num = query_num - train_query_num
        # print('{}, {}, {}'.format(query_num, train_query_num, dev_query_num))
        indices = np.arange(0, query_num)
        np.random.shuffle(indices)
        generate_data_per_query(infos_per_query, indices[0: train_query_num], 'data/', 'train_per_query.txt')
        generate_data_per_query(infos_per_query, indices[train_query_num: query_num], 'data/', 'dev_per_query.txt')
    else:
        raise NotImplementedError('Unsupported generate_type: {}'.format(args.generate_type))

def main():
    parser = argparse.ArgumentParser('TREC2014')
    parser.add_argument('--reload', action='store_true',
                        help='reload info_per_session/info_per_query and other dicts from ./data without preprocess')
    parser.add_argument('--dataset', default='../dataset/TREC2014/sessiontrack2014.xml',
                        help='dataset path')
    parser.add_argument('--trainset_ratio', default=0.7,
                        help='ratio of the train session/query according to the total number of sessions/queries')
    parser.add_argument('--generate_type', default='per_session',
                        help='per_session or per_query')
    args = parser.parse_args()
    if args.reload == False:
        # preprocess() generates info_per_session, info_per_query, query_qid, qid_query, url_uid, uid_url
        preprocess(args)
    else:
        # reload() loads 5 dicts saved by preprocess() & generates train_session.txt & dev_session.txt
        reload(args)
    
if __name__ == '__main__':
    main()