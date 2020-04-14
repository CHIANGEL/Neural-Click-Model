# coding: utf8

import os
import sys
import time
import pickle
import argparse
import logging
import pprint

from model import Model
from dataset import Dataset

def parse_args():
    parser = argparse.ArgumentParser('NCM')

    parser.add_argument('--dataset', default='Yandex',
                        help='name of the dataset')
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
    train_settings.add_argument('--dropout_rate', type=float, default=0,
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
    model_settings.add_argument('--representation_mode', default='QD',
                                help='representation mode of queries, documents and interactions')
    model_settings.add_argument('--algo', default='NCM',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_type', default='QD+Q+D',
                                help='which type of embeddings to use')
    model_settings.add_argument('--embed_size', type=int, default=128,
                                help='size of the embeddings if embed_type=random')
    model_settings.add_argument('--hidden_size', type=int, default=256,
                                help='size of RNN/LSTM hidden units')
    model_settings.add_argument('--model_type', default='rnn',
                                help='use RNN or LSTM')
    model_settings.add_argument('--max_doc_num', type=int, default=10,
                                help='max number of docs in a query')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_dirs', nargs='+',
                               default=['data/Yandex/train_per_query.txt'],
                               help='list of dirs that contain the preprocessed train data')
    path_settings.add_argument('--dev_dirs', nargs='+',
                               default=['data/Yandex/dev_per_query.txt'],
                               help='list of dirs that contain the preprocessed dev data')
    path_settings.add_argument('--test_dirs', nargs='+',
                               default=['data/Yandex/test_per_query.txt'],
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
    path_settings.add_argument('--data_parallel', action='store_true',
                               help='data_parallel')
    path_settings.add_argument('--gpu_num', type=int, default=1,
                               help='gpu_num')

    return parser.parse_args()

def train(args):
    """
     Training for NCM starts here
    """
    # Get logger
    logger = logging.getLogger("NCM")

    # Check the data files
    logger.info('Checking the data files...')
    for data_path in args.train_dirs + args.dev_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    assert len(args.train_dirs) > 0, 'No train files are provided.'
    
    # Load dataset
    logger.info("Loading the dataset...")
    dataset = Dataset(args, train_dirs=args.train_dirs, dev_dirs=args.dev_dirs)

    # Model initialization
    logger.info('Initializing the model...')
    model = Model(args, len(dataset.query_qid), len(dataset.url_uid))
    if args.load_model > -1:
        logger.info('Restoring the model...')
        model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Start training at model.global_step: {}'.format(model.global_step))

    # Start training
    logger.info('Training the model on training set...')
    model.train(dataset)
    logger.info('Done with model training!')

def evaluate(args):
    """
     Evaluate the pre-trained model on dev dataset
    """
    # Get logger
    logger = logging.getLogger("NCM")

    # Check the data files
    logger.info('Checking the data files...')
    for data_path in args.train_dirs + args.dev_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    assert len(args.dev_dirs) > 0, 'No dev files are provided.'

    # Load dataset
    logger.info("Loading the dataset...")
    dataset = Dataset(args, dev_dirs=args.dev_dirs)

    # Model Construction
    logger.info('Constructing the model...')
    model = Model(args, len(dataset.query_qid), len(dataset.url_uid))

    # Restore the pre-trained model
    logger.info('Restoring the pre-trained model...')
    assert args.load_model > -1, 'args.load_model should be set at evaluation period!' # make sure there is something to store at evaluation
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Start evaluation at model.global_step: {}'.format(model.global_step))

    # Start evaluation
    logger.info('Evaluating the model on dev set...')
    dev_batches = dataset.gen_mini_batches('dev', 1, shuffle=False)
    dev_loss = model.evaluate(dev_batches, dataset, result_dir=args.result_dir,
                              result_prefix='dev.{}.{}.{}'.format(args.algo, args.load_model, time.strftime("%m-%d %H:%M:%S", time.localtime())))
    logger.info('Loss on dev set: {}'.format(dev_loss))
    logger.info('Predicted results are saved to {}'.format(os.path.join(args.result_dir)))
    logger.info('Done with model evaluation!')

def predict(args):
    """
     Predict answers for test files
    """
    # Get logger
    logger = logging.getLogger("NCM")

    # Check the data files
    logger.info('Checking the data files...')
    for data_path in args.test_dirs:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    assert len(args.test_dirs) > 0, 'No test files are provided.'

    # Load dataset
    logger.info("Loading the dataset...")
    dataset = Dataset(args, test_dirs=args.test_dirs)

    # Model Construction
    logger.info('Constructing the model...')
    model = Model(args, len(dataset.query_qid), len(dataset.url_uid))

    # Restore the pre-trained model
    logger.info('Restoring the pre-trained model...')
    assert args.load_model > -1, 'args.load_model should be set at prediction period!' # make sure there is something to store at evaluation
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Start prediction at model.global_step: {}'.format(model.global_step))

    # Compute test loss
    logger.info('Predicting answers on test set...')
    test_batches = dataset.gen_mini_batches('test', 1, shuffle=False)
    test_loss = model.evaluate(test_batches, dataset, result_dir=args.result_dir,
                               result_prefix='test.{}.{}.{}'.format(args.algo, args.load_model, time.strftime("%m-%d %H:%M:%S", time.localtime())))
    logger.info('Predicted results are saved to {}'.format(os.path.join(args.result_dir)))
    logger.info('Loss on test set: {}'.format(test_loss))

    # Compute log likelihood
    logger.info('Computing log likelihood on test set...')
    test_batches = dataset.gen_mini_batches('test', 1, shuffle=False)
    loglikelihood = model.log_likelihood(test_batches, dataset)
    logger.info('Log likelihood on test set: {}'.format(loglikelihood))

   #  Compute perplexity
    logger.info('Computing perplexity on test set...')
    test_batches = dataset.gen_mini_batches('test', 1, shuffle=False)
    perplexity = model.perplexity(test_batches, dataset)
    logger.info('Perplexity on test set: {}'.format(perplexity))


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
    formatter = logging.Formatter('%(asctime)s: %(message)s')
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