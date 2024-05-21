from __future__ import division
from __future__ import print_function

import os
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import pickle
import pandas as pd


# IMPORT CONSTANTS
from GDMC_Models import Constants
# NEURAL NETWORK MODULES/LAYERS
from GDMC_Models import  GDCM_Max_Min, GDCM_Graph_Comp

from GDMC_Models import Vocab
# DATASET CLASS FOR SICK DATASET
from GDMC_Models import Dataset
# METRICS CLASS FOR EVALUATION
from GDMC_Models import Metrics
# UTILITY FUNCTIONS
from GDMC_Models import utils
# TRAIN AND TEST HELPER FUNCTIONS
from GDMC_Models import Trainer
# CONFIG PARSER
from config import parse_args

import time
import gc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# MAIN BLOCK
def main():
    t_start = time.time()
    global args
    args = parse_args()
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    log_dir = os.path.join(args.save, args.model_name)
    try:
    	os.makedirs(log_dir)
    except Exception as e:
    	print(e)
    	print(' \t logdir ; {} exists'.format(log_dir))

    fh = logging.FileHandler(os.path.join(log_dir, args.expname)+'.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # argument validation
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if args.sparse and args.wd != 0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')

    logger.debug(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)


    train_dir =   os.path.join(args.data, 'Train/')#'data/sick/train/'
    #dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'Test/') #'data/sick/test/' #
    dev_dir =   os.path.join(args.data, 'Dev/')#'data/sick/train/'

    print(train_dir, test_dir,dev_dir )

    args.freeze_embed= True


    if args.model_name == 'GDCM_Graph_Comp':
	    model = GDCM_Graph_Comp.SimilarityTreeLSTM(
        args.input_dim,
        args.mem_dim,
        args.hidden_dim,
        args.sparse,
        args.num_classes,
        args.freeze_embed,
        args.max_num_para,
        args.max_num_sent,
        args.max_num_word,
        args.num_filter,
        args.number_head,
        args.beta,
        args.attn,
        args.radius)

    elif args.model_name == 'GDCM_Max_Min':
	    model = GDCM_Max_Min.SimilarityTreeLSTM(
        args.input_dim,
        args.mem_dim,
        args.hidden_dim,
        args.sparse,
        args.num_classes,
        args.freeze_embed,
        args.max_num_para,
        args.max_num_sent,
        args.max_num_word,
        args.num_filter,
        args.number_head,
        args.beta,
        args.attn,
        args.radius)

    print(' Total number of parameter : {}'.format(count_parameters(model)))
    # print('Number of parameters in DOCLstm:{}'.format(count_parameters(model.doclstm)))
    #print('Number of parameters in Tree-lstm :{}'.format(count_parameters(model.doclstm.childsumtreelstm)))
    # print( 'Size of model : {} byte '.format(sys.getsizeof(model)))

    criterion =  nn.CrossEntropyLoss()



    # emb_file_name = '{}_{}_{}d.pth'.format(args.data_name, args.emb_name, args.input_dim)
    # print(' emb file name:: ', emb_file_name)
    # emb_file = os.path.join(args.data, emb_file_name)
    # print( ' emb file name : ', emb_file)
    # if os.path.isfile(emb_file):
    #     print(' Embed path : {} exists'.format(emb_file))
    #     emb = torch.load(emb_file)
    # else:
    #     # load glove embeddings and vocab
    #     glove_vocab, glove_emb = utils.load_word_vectors(
    #         os.path.join(args.glove, 'glove.6B.200d'))
    #     logger.debug('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
    #     emb = torch.zeros(vocab.size(), glove_emb.size(1), dtype=torch.float, device=device)
    #     emb.normal_(0, 0.05)
    #     # zero out the embeddings for padding and other special words if they are absent in vocab
    #     for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD,
    #                                 Constants.BOS_WORD, Constants.EOS_WORD]):
    #         emb[idx].zero_()
    #     for word in vocab.labelToIdx.keys():
    #         if glove_vocab.getIndex(word):
    #             emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
    #     torch.save(emb, emb_file)
    # print(' emb shape :', emb.shape)
    # model.doclstm.emb.weight.data.copy_(emb)
    # del emb
    # gc.collect()

    # delete emb
    model.to(device), criterion.to(device)
    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,
                                         model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                     model.parameters()), lr=args.lr, weight_decay=args.wd)
    metrics = Metrics(args.num_classes)

    # exteract labels for training data
    trainer = Trainer(args, model, criterion, optimizer, device, args.batchsize, args.num_classes, args.file_len)







    best = float('inf')
    for epoch in range(args.epochs):
        train_loss = trainer.train()

        train_loss, train_pred = trainer.test(0)

        fname = os.path.join(train_dir, 'train_label.pkl')
        fin = open(fname , 'rb')
        ground_truth = pickle.load(fin)
        print("the length of total label:",len(ground_truth))
        fin.close()
        if args.run_type == 'debug':
            train_accuracy = metrics.accuracy(train_pred[:100], ground_truth[:100])
            train_fmeasure = metrics.fmeasure(train_pred[:100], ground_truth[:100])
        else:
            train_accuracy = metrics.accuracy(train_pred, ground_truth)
            train_fmeasure = metrics.fmeasure(train_pred, ground_truth)
        logger.info('==> Epoch: {},\t Train Loss: {} \t Accuracy: {} \t F1-score: {}'.format(
                 epoch, train_loss, train_accuracy , train_fmeasure))
        del ground_truth
        del train_pred
        gc.collect()

        #Load developement data set
        fname = os.path.join(dev_dir,'dev_label.pkl')
        fin = open(fname , 'rb')    # Load from pickle file instead of creating
        dev_data = pickle.load(fin)
        fin.close()
        print(len(dev_data))

        # dev_loss, dev_pred = trainer.test(1)
        # # Exeract label from developement dataset
        #
        #
        # fname = os.path.join(dev_dir, 'dev_label.pkl')
        # fin = open(fname , 'rb')
        # ground_truth = pickle.load(fin)
        # fin.close()
        # # print(' type of dev_pred : ', type(dev_pred))
        # # print(' type of dev_pred : ', type(ground_truth))
        # if args.run_type == 'debug':
        #     dev_accuracy = metrics.accuracy(dev_pred[:100], ground_truth[:100])
        #     dev_fmeasure = metrics.fmeasure(dev_pred[:100], ground_truth[:100])
        # else:
        #     dev_accuracy = metrics.accuracy(dev_pred, ground_truth)
        #     dev_fmeasure = metrics.fmeasure(dev_pred, ground_truth)
        # logger.info('==> Epoch: {},\t Devlopment Loss: {} \t Accuracy: {} \t F1-score: {}'.format(
        #          epoch, dev_loss, dev_accuracy , dev_fmeasure))
        # gc.collect()

        test_loss, test_pred = trainer.test(2)
        fname = os.path.join(test_dir, 'test_label.pkl')
        fin = open(fname , 'rb')
        ground_truth = pickle.load(fin)
        fin.close()
        if args.run_type == 'debug':
            test_accuracy = metrics.accuracy(test_pred[:100], ground_truth[:100])
            test_fmeasure = metrics.fmeasure(test_pred[:100], ground_truth[:100])
        else:
            test_accuracy = metrics.accuracy(test_pred, ground_truth)
            test_fmeasure = metrics.fmeasure(test_pred, ground_truth)
        logger.info('==> Epoch: {},\t Test Loss: {} \t Accuracy: {} \t F1-score: {}'.format(
                 epoch, test_loss, test_accuracy , test_fmeasure))
        del test_pred
        del ground_truth
        gc.collect()



        checkpoint = {
            'model_state_dict': trainer.model.state_dict(),
            'optim_state_dict': trainer.optimizer.state_dict() ,
            'train_loss':train_loss,#'dev_loss':dev_loss,
            #'accuracy': dev_accuracy, 'fmeasure': dev_fmeasure,
            'args': args, 'saved_epoch': epoch , 'total_epoch' : args.epochs,
             'model_name' : args.model_name,
             'dataset_name' : args.data_name,
             'embedding_name' : args.emb_name,


        }
        logger.debug('==> New optimum found, checkpointing everything now...')
        torch.save(checkpoint, '%s_%d.pt' % (os.path.join(log_dir, args.expname), epoch))
        # if best > dev_loss:
        #     best = dev_loss
        #     logger.debug('==> New optimum found, checkpointing everything now...')
        #     torch.save(checkpoint, '%s_best.pt' % os.path.join(log_dir, args.expname))

    t_end = time.time()
    print(' Total time taken : {} , dataset name : {}, model name : {}'.format((t_end-t_start), args.data_name, args.model_name) )

if __name__ == "__main__":
    main()
