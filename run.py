import sys
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging
from dataset import Dataset
from model import Model
import utils
import pprint

def parse_args():
    parser = argparse.ArgumentParser('NCM')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--valid', action='store_true',
                        help='evaluate the model on valid set')
    parser.add_argument('--test', action='store_true',
                        help='evaluate the model on test set')
    parser.add_argument('--rank', action='store_true',
                        help='rank on train set')
    parser.add_argument('--rank_cheat', action='store_true',
                        help='rank on train set in a cheating way')
    parser.add_argument('--generate_click_seq', action='store_true',
                        help='generate click sequence based on model itself')
    parser.add_argument('--generate_click_seq_cheat', action='store_true',
                        help='generate click sequence based on ground truth data')
    parser.add_argument('--generate_synthetic_dataset', action='store_true',
                        help='generate synthetic dataset for reverse ppl')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
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

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', default='NCM',
                                help='the name of the algorithm')
    model_settings.add_argument('--embed_size', type=int, default=128,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=128,
                                help='size of RNN hidden units')
    model_settings.add_argument('--max_d_num', type=int, default=10,
                                help='max number of docs in a session')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--dataset', default='TianGong-ST',
                                help='name of the dataset to be used')
    path_settings.add_argument('--model_dir', default='./outputs/models/',
                                help='the dir to store models')
    path_settings.add_argument('--result_dir', default='./outputs/results/',
                                help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='./outputs/summary/',
                                help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_dir', default='./outputs/log/',
                                help='path of the log file. If not set, logs are printed to console')

    path_settings.add_argument('--eval_freq', type=int, default=100,
                                help='the frequency of evaluating on the valid set when training')
    path_settings.add_argument('--check_point', type=int, default=100,
                                help='the frequency of saving model')
    path_settings.add_argument('--patience', type=int, default=5,
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

def train(args, dataset):
    """
    Train the model
    """
    logger = logging.getLogger("NCM")
    logger.info('Initialize the model...')
    model = Model(args, dataset.query_size, dataset.doc_size, dataset.vtype_size, dataset)
    logger.info('model.global_step: {}'.format(model.global_step))
    if args.load_model > -1:
        logger.info('Reloading the model...')
        model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Training the model...')
    model.train(dataset)
    logger.info('Done with model training!')

def valid(args, dataset):
    """
    Evaluate the model on valid set
    """
    logger = logging.getLogger("NCM")
    logger.info('Initialize the model...')
    model = Model(args, dataset.query_size, dataset.doc_size, dataset.vtype_size, dataset)
    logger.info('model.global_step: {}'.format(model.global_step))
    assert args.load_model > -1, 'args.load_model is required to specify the model file to be loaded!'
    logger.info('Reloading the model...')
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Evaluating the model on valid set...')
    valid_batches = dataset.gen_mini_batches('valid', dataset.validset_size, shuffle=False)
    valid_loss, perplexity = model.evaluate(valid_batches, dataset)
    logger.info('Loss on valid set: {}'.format(float(valid_loss)))
    logger.info('Perplexity on valid set: {}'.format(float(perplexity)))

def test(args, dataset):
    """
    Evaluate the model on test set
    """
    logger = logging.getLogger("NCM")
    logger.info('Initialize the model...')
    model = Model(args, dataset.query_size, dataset.doc_size, dataset.vtype_size, dataset)
    logger.info('model.global_step: {}'.format(model.global_step))
    assert args.load_model > -1, 'args.load_model is required to specify the model file to be loaded!'
    logger.info('Reloading the model...')
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Evaluating the model on test set...')
    test_batches = dataset.gen_mini_batches('test', dataset.testset_size, shuffle=False)
    test_loss, perplexity = model.evaluate(test_batches, dataset)
    logger.info('Loss on test set: {}'.format(float(test_loss)))
    logger.info('perplexity on test set: {}'.format(float(perplexity)))

def rank(args, dataset):
    """
    Rank documents for relevance estimation task
    """
    logger = logging.getLogger("NCM")
    logger.info('Initialize the model...')
    model = Model(args, dataset.query_size, dataset.doc_size, dataset.vtype_size, dataset)
    logger.info('model.global_step: {}'.format(model.global_step))
    assert args.load_model > -1, 'args.load_model is required to specify the model file to be loaded!'
    logger.info('Reloading the model...')
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Computing NDCG@k for relevance estimation...')
    trunc_levels = [1, 3, 5, 10]
    label_batches = dataset.gen_mini_batches('label', dataset.labelset_size, shuffle=False)
    ndcgs = model.ranking(label_batches, dataset)
    for trunc_level in trunc_levels:
        logger.info("NDCG@{}: {}".format(trunc_level, ndcgs[trunc_level]))

def run():
    """
    Prepare and run the whole system.
    """
    # Get arguments
    args = parse_args()
    assert args.batch_size % args.gpu_num == 0
    assert args.hidden_size % 2 == 0

    # Create a logger
    logger = logging.getLogger("NCM")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    utils.check_path(args.model_dir)
    utils.check_path(args.result_dir)
    utils.check_path(args.summary_dir)
    if args.log_dir:
        utils.check_path(args.log_dir)
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

    # Check the directories
    logger.info('Checking the directories...')
    for dir_path in [args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Load dataset
    logger.info('Loading train/valid/test/label data...')
    dataset = Dataset(args)
    
    # Start main process
    if args.train:
        train(args, dataset)
    if args.valid:
        valid(args, dataset)
    if args.test:
        test(args, dataset)
    if args.rank:
        rank(args, dataset)
    logger.info('Run done.')

if __name__ == '__main__':
    run()
