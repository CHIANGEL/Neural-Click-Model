# encoding:utf-8
import sys
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging
from dataset import Dataset
from model import Model
from utils import *
from ndcg import RelevanceEstimator
from TianGong_HumanLabel_Parser import TianGong_HumanLabel_Parser
import pprint

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('NCM')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--rank', action='store_true',
                        help='rank on train set')
    parser.add_argument('--generate_click_seq', action='store_true',
                        help='generate click sequence based on model itself')
    parser.add_argument('--generate_click_seq_cheat', action='store_true',
                        help='generate click sequence based on ground truth data')
    parser.add_argument('--gpu', type=str, default='',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adadelta',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.01,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=1e-5,
                                help='weight decay')
    train_settings.add_argument('--momentum', type=float, default=0.99,
                                help='momentum')
    train_settings.add_argument('--dropout_rate', type=float, default=0.5,
                                help='dropout rate')
    train_settings.add_argument('--batch_size', type=int, default=64,
                                help='train batch size')
    train_settings.add_argument('--num_steps', type=int, default=20000,
                                help='number of training steps')
    train_settings.add_argument('--num_train_files', type=int, default=1,
                                help='number of training files')
    train_settings.add_argument('--num_dev_files', type=int, default=1,
                                help='number of dev files')
    train_settings.add_argument('--num_test_files', type=int, default=1,
                                help='number of test files')
    train_settings.add_argument('--minimum_occurrence', type=int, default=1,
                                help='minimum_occurrence for NDCG')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', default='NCM',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=128,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=128,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_d_num', type=int, default=10,
                                help='max number of docs in a session')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_dirs', nargs='+',
                                default=['./data/TianGong-ST/train_per_query.txt'],
                                help='list of dirs that contain the preprocessed train data')
    path_settings.add_argument('--dev_dirs', nargs='+',
                                default=['./data/TianGong-ST/dev_per_query.txt'],
                                help='list of dirs that contain the preprocessed dev data')
    path_settings.add_argument('--test_dirs', nargs='+',
                                default=['./data/TianGong-ST/test_per_query.txt'],
                                help='list of dirs that contain the preprocessed test data')
    path_settings.add_argument('--human_label_dir', default='./data/TianGong-ST/human_label.txt',
                                help='the dir to Human Label txt file')
    path_settings.add_argument('--model_dir', default='./outputs/models/',
                                help='the dir to store models')
    path_settings.add_argument('--result_dir', default='./outputs/results/',
                                help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='./outputs/summary/',
                                help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_dir', default='./outputs/log/',
                                help='path of the log file. If not set, logs are printed to console')

    path_settings.add_argument('--eval_freq', type=int, default=2000,
                                help='the frequency of evaluating on the dev set when training')
    path_settings.add_argument('--check_point', type=int, default=2000,
                                help='the frequency of saving model')
    path_settings.add_argument('--patience', type=int, default=3,
                                help='lr half when more than the patience times of evaluation\' loss don\'t decrease')
    path_settings.add_argument('--lr_decay', type=float, default=0.5,
                                help='lr decay')
    path_settings.add_argument('--load_model', type=int, default=-1,
                                help='load model global step')
    path_settings.add_argument('--data_parallel', type=bool, default=False,
                                help='data_parallel')
    path_settings.add_argument('--gpu_num', type=int, default=1,
                                help='gpu_num')

    return parser.parse_args()

def train(args):
    """
    trains the model
    """
    logger = logging.getLogger("NCM")
    logger.info('Checking the data files...')
    for data_path in args.train_dirs + args.dev_dirs + args.test_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    assert len(args.train_dirs) > 0, 'No train files are provided.'
    dataset = Dataset(args, train_dirs=args.train_dirs, dev_dirs=args.dev_dirs, test_dirs=args.test_dirs)
    logger.info('Initialize the model...')
    model = Model(args, len(dataset.qid_query), len(dataset.uid_url),  len(dataset.vid_vtype))
    logger.info('model.global_step: {}'.format(model.global_step))
    if args.load_model > -1:
        logger.info('Restoring the model...')
        model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Training the model...')
    model.train(dataset)
    logger.info('Done with model training!')

def evaluate(args):
    """
    compute perplexity and log-likelihood for dev file
    """
    logger = logging.getLogger("NCM")
    logger.info('Checking the data files...')
    for data_path in args.train_dirs + args.dev_dirs + args.test_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    assert len(args.dev_dirs) > 0, 'No dev files are provided.'
    dataset = Dataset(args, train_dirs=args.train_dirs, dev_dirs=args.dev_dirs, test_dirs=args.test_dirs)
    logger.info('Initialize the model...')
    model = Model(args, len(dataset.qid_query), len(dataset.uid_url), len(dataset.vid_vtype))
    logger.info('model.global_step: {}'.format(model.global_step))
    assert args.load_model > -1
    logger.info('Restoring the model...')
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Evaluating the model on dev set...')
    dev_batches = dataset.gen_mini_batches('dev', args.batch_size, shuffle=False)
    dev_loss, perplexity, perplexity_at_rank = model.evaluate(dev_batches, dataset, result_dir=args.result_dir,
                                                                result_prefix='dev.predicted.{}.{}.{}'.format(args.algo, args.load_model, time.time()))
    logger.info('Loss on dev set: {}'.format(dev_loss))
    logger.info('Perplexity on dev set: {}'.format(perplexity))
    logger.info('Perplexity at rank: {}'.format(perplexity_at_rank))
    logger.info('Predicted results are saved to {}'.format(os.path.join(args.result_dir)))

def predict(args):
    """
    compute perplexity and log-likelihood for test file
    """
    logger = logging.getLogger("NCM")
    logger.info('Checking the data files...')
    for data_path in args.train_dirs + args.dev_dirs + args.test_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    assert len(args.test_dirs) > 0, 'No test files are provided.'
    dataset = Dataset(args, train_dirs=args.train_dirs, dev_dirs=args.dev_dirs, test_dirs=args.test_dirs)
    logger.info('Initialize the model...')
    model = Model(args, len(dataset.qid_query), len(dataset.uid_url), len(dataset.vid_vtype))
    logger.info('model.global_step: {}'.format(model.global_step))
    assert args.load_model > -1
    logger.info('Restoring the model...')
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Predict on test files...')
    test_batches = dataset.gen_mini_batches('test', args.batch_size, shuffle=False)
    test_loss, perplexity, perplexity_at_rank = model.evaluate(test_batches, dataset, result_dir=args.result_dir,
                                                                result_prefix='test.predicted.{}.{}.{}'.format(args.algo, args.load_model, time.time()))
    logger.info('Loss on test set: {}'.format(test_loss))
    logger.info('perplexity on test set: {}'.format(perplexity))
    logger.info('perplexity at rank: {}'.format(perplexity_at_rank))
    logger.info('Predicted results are saved to {}'.format(os.path.join(args.result_dir)))

def rank(args):
    """
    ranking performance on test files
    """
    logger = logging.getLogger("NCM")
    logger.info('Checking the data files...')
    for data_path in args.train_dirs + args.dev_dirs + args.test_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    dataset = Dataset(args, train_dirs=args.train_dirs, dev_dirs=args.dev_dirs, test_dirs=args.test_dirs)
    logger.info('Initialize the model...')
    model = Model(args, len(dataset.qid_query), len(dataset.uid_url), len(dataset.vid_vtype))
    logger.info('model.global_step: {}'.format(model.global_step))
    assert args.load_model > -1
    logger.info('Restoring the model...')
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Compute NDCG on test files...')
    relevance_queries = TianGong_HumanLabel_Parser().parse(args.human_label_dir)
    relevance_estimator = RelevanceEstimator(args.minimum_occurrence)
    trunc_levels = [1, 3, 5, 10]
    for trunc_level in trunc_levels:
        ndcg_version1, ndcg_version2 = relevance_estimator.evaluate(model, dataset, relevance_queries, trunc_level)
        logger.info("NDCG@{}: {}, {}".format(trunc_level, ndcg_version1, ndcg_version2))
    logger.info('【{}, {}】'.format(args.load_model, args.minimum_occurrence))

def generate_click_seq(args):
    """
    generate the click sequence for test file based on model itself
    """
    logger = logging.getLogger("NCM")
    logger.info('Checking the data files...')
    for data_path in args.train_dirs + args.dev_dirs + args.test_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    assert len(args.test_dirs) > 0, 'No test files are provided.'
    dataset = Dataset(args, train_dirs=args.train_dirs, dev_dirs=args.dev_dirs, test_dirs=args.test_dirs)
    logger.info('Initialize the model...')
    model = Model(args, len(dataset.qid_query), len(dataset.uid_url), len(dataset.vid_vtype))
    logger.info('model.global_step: {}'.format(model.global_step))
    assert args.load_model > -1
    logger.info('Restoring the model...')
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Generating click sequence based on the model itself...')
    test_batches = dataset.gen_mini_batches('test', args.batch_size, shuffle=False)
    file_path = os.path.join(args.model_dir, '..', 'click_seq')
    model.generate_click_seq(test_batches, file_path, '{}.txt'.format(time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time()))))
    logger.info('Done with click sequence generation.')
    
def generate_click_seq_cheat(args):
    """
    generate the click sequence for test file based on ground truth data
    """
    logger = logging.getLogger("NCM")
    logger.info('Checking the data files...')
    for data_path in args.train_dirs + args.dev_dirs + args.test_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    assert len(args.test_dirs) > 0, 'No test files are provided.'
    dataset = Dataset(args, train_dirs=args.train_dirs, dev_dirs=args.dev_dirs, test_dirs=args.test_dirs)
    logger.info('Initialize the model...')
    model = Model(args, len(dataset.qid_query), len(dataset.uid_url), len(dataset.vid_vtype))
    logger.info('model.global_step: {}'.format(model.global_step))
    assert args.load_model > -1
    logger.info('Restoring the model...')
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Generating click sequence based on the model itself...')
    test_batches = dataset.gen_mini_batches('test', args.batch_size, shuffle=False)
    file_path = os.path.join(args.model_dir, '..', 'click_seq')
    model.generate_click_seq_cheat(test_batches, file_path, '{}.txt'.format(time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time()))))
    logger.info('Done with click sequence generation.')

def run():
    """
    Prepares and runs the whole system.
    """
    # get arguments
    args = parse_args()
    assert args.batch_size % args.gpu_num == 0
    assert args.hidden_size % 2 == 0

    # create a logger
    logger = logging.getLogger("NCM")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    check_path(args.model_dir)
    check_path(args.result_dir)
    check_path(args.summary_dir)
    if args.log_dir:
        check_path(args.log_dir)
        file_handler = logging.FileHandler(args.log_dir + time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time())) + '.txt')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    logger.info('Checking the directories...')
    for dir_path in [args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)
    if args.rank:
        rank(args)
    if args.generate_click_seq:
        generate_click_seq(args)
    if args.generate_click_seq_cheat:
        generate_click_seq_cheat(args)
    logger.info('run done.')

if __name__ == '__main__':
    run()
