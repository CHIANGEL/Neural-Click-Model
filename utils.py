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

from pynvml import *
def get_gpu_infos():
    print('-------------------------------------------')
    nvmlInit()     #初始化
    print("  Driver: {}".format(nvmlSystemGetDriverVersion()))  #显示驱动信息
    #>>> Driver: 384.xxx

    #查看设备
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        print("  GPU", i, ":", nvmlDeviceGetName(handle))
    #>>>
    #GPU 0 : b'GeForce GTX 1080 Ti'
    #GPU 1 : b'GeForce GTX 1080 Ti'

    #查看显存、温度、风扇、电源
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print("  Memory Total: ", info.total)
    print("  Memory Free: ", info.free)
    print("  Memory Used: ", info.used)

    print("  Temperature is %d C" % nvmlDeviceGetTemperature(handle,0))
    print("  Fan speed is ", nvmlDeviceGetFanSpeed(handle))
    print("  Power ststus", nvmlDeviceGetPowerState(handle))

    #最后要关闭管理工具
    nvmlShutdown()
    print('-------------------------------------------')

def xml_line_removable(xml_line):
    if xml_line.find('<query>') != -1 and xml_line.find('</query>') != -1:
        return 1
    elif xml_line.find('<url>') != -1 and xml_line.find('</url>') != -1:
        return 1
    elif xml_line.find('<title>') != -1 and xml_line.find('</title>') != -1:
        return 1
    elif xml_line.find('<relevance>') != -1 or xml_line.find('</relevance>') != -1:
        return 1
    elif xml_line.find('<TACM>') != -1 and xml_line.find('</TACM>') != -1:
        return 1
    elif xml_line.find('<PSCM>') != -1 and xml_line.find('</PSCM>') != -1:
        return 1
    elif xml_line.find('<THCM>') != -1 and xml_line.find('</THCM>') != -1:
        return 1
    elif xml_line.find('<UBM>') != -1 and xml_line.find('</UBM>') != -1:
        return 1
    elif xml_line.find('<DBN>') != -1 and xml_line.find('</DBN>') != -1:
        return 1
    elif xml_line.find('<POM>') != -1 and xml_line.find('</POM>') != -1:
        return 1
    return 0