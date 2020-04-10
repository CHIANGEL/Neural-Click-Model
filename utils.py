import os
import pprint

def check_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def save_dict(file_path, file_name, dict):
    check_path(file_path)
    file = open(file_path + file_name, 'w')
    file.write(str(dict))
    file.close()

def load_dict(file_path, file_name):
    data_path = os.path.join(file_path, file_name)
    assert os.path.isfile(data_path), '{} file does not exist.'.format(data_path)
    file = open(data_path, 'r')
    return eval(file.read())
    
def save_list(file_path, file_name, list_data):
    check_path(file_path)
    file = open(file_path + file_name, 'w')
    file.write(str(list_data))
    file.close()

def load_list(file_path, file_name):
    data_path = os.path.join(file_path, file_name)
    assert os.path.isfile(data_path), '{} file does not exist.'.format(data_path)
    file = open(data_path, 'r')
    return eval(file.read())

def generate_data_per_session(infos_per_session, indices, file_path, file_name):
    check_path(file_path)
    file = open(file_path + file_name, 'w')
    for key in indices:
        query_sequence_for_print = []
        prev_document_info_for_print = []
        info_per_session = infos_per_session[key]
        interaction_infos = info_per_session['interactions']
        for interaction_info in interaction_infos:
            qid = interaction_info['qid']
            uids = interaction_info['uids']
            clicks = interaction_info['clicks']
            query_sequence_for_print.append(qid)
            for idx, uid in enumerate(uids):
                click = clicks[idx]
                rank = idx + 1
                # No vertical information in demo Yandex dataset
                document_info_for_print = [uid, rank, 1] 
                file.write('{}\t{}\t{}\t{}\n'.format(str(query_sequence_for_print), 
                                                     str(prev_document_info_for_print), 
                                                     str(document_info_for_print),
                                                     click))
                prev_document_info_for_print = [uid, rank, 1, click]
        file.write('\n')
    file.close()

def generate_data_per_query(infos_per_query, indices, file_path, file_name):
    check_path(file_path)
    file = open(file_path + file_name, 'w')
    for key in indices:
        interaction_info = infos_per_query[key]
        qid = interaction_info['qid']
        uids = interaction_info['uids']
        clicks = interaction_info['clicks']
        for idx, uid in enumerate(uids):
            click = clicks[idx]
            rank = idx + 1
            document_info_for_print = [uid, rank, click]
            file.write('{}\t{}\n'.format(str([qid]), str(document_info_for_print)))
        file.write('\n')
    file.close()

def get_unique_queries(sessions):
    """
     Extracts and returns the set of unique queries contained in a given list of search sessions.
    """
    queries = set()
    for session in sessions:
        queries.add(search_session.query)
    return queries

def filter_sessions(sessions, queries):
    """
     Filters the given list of search sessions so that it contains only a given list of queries.
    """
    filtered_sessions = []
    for session in sessions:
        if session[''] in queries:
            filtered_sessions.append(session)
    return filtered_sessions