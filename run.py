# coding: utf8

import os
import sys
import time
import pickle
import argparse
import logging

def parse_args():
    parser = argparse.ArgumentParser('NCM')

    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--rank', action='store_true',
                        help='rank on train set')
    parser.add_argument('--gpu', type=str, default='',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adadelta',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.01,
                                help='learning rate')
    train_settings.add_argument('--lr_decay', type=float, default=0.5,
                                help='lr decay')
    train_settings.add_argument('--patience', type=int, default=3,
                                help='lr half when more than the patience times of evaluation\' loss don\'t decrease')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--momentum', type=float, default=0.99,
                                help='momentum')
    train_settings.add_argument('--dropout_rate', type=float, default=0.5,
                                help='dropout rate')
    train_settings.add_argument('--batch_size', type=int, default=1,
                                help='train batch size')
    train_settings.add_argument('--num_steps', type=int, default=200000,
                                help='number of training steps')
    train_settings.add_argument('--num_train_files', type=int, default=40,
                                help='number of training files') # ????????????????????????
    train_settings.add_argument('--num_dev_files', type=int, default=40,
                                help='number of dev files')
    train_settings.add_argument('--num_test_files', type=int, default=40,
                                help='number of test files')
    train_settings.add_argument('--eval_freq', type=int, default=1000,
                                help='the frequency of evaluating on the dev set when training')
    train_settings.add_argument('--check_point', type=int, default=1000,
                                help='the frequency of saving model')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', default='neural_click_model',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=128,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=256,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_doc_num', type=int, default=10,
                                help='max number of docs in a query')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_dirs', nargs='+',
                               default=['../data/20180804'],
                               help='list of dirs that contain the preprocessed train data')
    path_settings.add_argument('--dev_dirs', nargs='+',
                               default=['../data/20180805'],
                               help='list of dirs that contain the preprocessed dev data')
    path_settings.add_argument('--test_dirs', nargs='+',
                               default=['../data/20180805'],
                               help='list of dirs that contain the preprocessed test data')
    path_settings.add_argument('--model_dir', default='./outputs/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='./outputs/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='./outputs/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_dir', default='./outputs/logs/',
                               help='path of the log file')

    path_settings.add_argument('--load_model', type=int, default=-1,
                               help='load model global step')
    path_settings.add_argument('--data_parallel', type=bool, default=False,
                               help='data_parallel')
    path_settings.add_argument('--gpu_num', type=int, default=1,
                               help='gpu_num')

    return parser.parse_args()



def run():
    '''
     The whole system starts from here.
    '''
    # Parse arguments
    args = parse_args()
    assert args.batch_size % args.gpu_num == 0
    assert args.hidden_size % 2 == 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Check the directories
    for dir_path in [args.model_dir, args.result_dir, args.summary_dir, args.log_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # Logger initializations
    logger = logging.getLogger("NCM")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(args.log_dir + time.strftime("%m-%d %H:%M:%S", time.localtime()) + '.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('Running with args: {}'.format(args))

    # Run the NCM system
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)
    if args.rank:
        rank(args)
    logger.info('Run done.')

if __name__ == '__main__':
    run()