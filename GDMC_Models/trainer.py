from __future__ import division
from __future__ import print_function
from tqdm import tqdm
import torch
from . import utils
import os
import pandas as pd


import os
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import _pickle as pickle
# import cPickle as pickle
import scipy

import os
import sys
import pickle
import torch
import time
from config import parse_args




#import torch.autograd.profiler as profiler
import gc


train_data_len = { 'WL' : 56288, 'NELA': 70590, 'FNC_Bin': 44791 ,'contest': 900,'FNC_Mix': 55151, 'ISOT':35311,'clickbait':26961 } #'FNC_Mix': 55482
test_data_len = { 'WL' : 7027, 'NELA': 6222, 'FNC_Bin': 25413, 'contest': 612,'FNC_Mix': 15005,'ISOT': 5311,'clickbait':5778  } #'FNC_Mix': 15077
dev_data_len = { 'WL' : 7032, 'NELA': 6234, 'FNC_Bin': 4980, 'contest': 364,'FNC_Mix': 4808,'ISOT': 3541,'clickbait':5778 } #FNC_Mix': 4825





global args
args = parse_args()
train_dir =  os.path.join(args.data, 'Train/')
test_dir =  os.path.join(args.data, 'Test/')
dev_dir =  os.path.join(args.data, 'Dev/')
print(train_dir)
class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device, batchsize, num_classes, file_len):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer


        self.device = device
        self.epoch = 0
        self.batchsize = batchsize
        self.num_classes = num_classes
        self.file_len = file_len #  Number of news articles in each train/test/validation part file.


    # helper function for training
    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        self.model.zero_grad()
        total_loss=0.0
        count=0
        data_dic=[1000]



        data_size= train_data_len[args.data_name] #35315 #len(data_dic)                #size of data
        batch_size = self.batchsize                    #batch size
        no_batch= int(data_size/batch_size)    # no of batcg
        number_batch_per_file = int(self.file_len/batch_size)

        for batch in tqdm(range(no_batch), desc='\t Training epoch ' + str(self.epoch + 1) + ''):
            if args.run_type == 'debug' and batch > 0:
                continue

            if (batch % number_batch_per_file )==0 :
                del data_dic
                gc.collect()
                filename ='Fold-%d.pkl' % count
                fname_out = os.path.join(train_dir, filename)
                fin = open(fname_out , 'rb')
                data_dic = pickle.load(fin)
                #print(data_dic.keys())
                print("the no of news article pair",len(data_dic))
                count=count+1
            #print('Processing batch : {}'.format(batch))
            batch_loss=torch.zeros(1)


            for idx in tqdm(range(batch*batch_size, (batch+1)*batch_size, 1), desc=' batch # ' + str(batch + 1) + ''):
                if args.run_type == 'debug' and  idx >100:
                    break
                #print(idx)

                body = data_dic[idx]
                label= data_dic[idx]['stance']
                target = utils.map_label_to_target(label, 2)


                output = self.model(body)


                loss = self.criterion(output, target) # / 20
                total_loss += loss.item()
                loss.backward()
                if idx == data_size-1 :
                    del  data_dic
                    gc.collect()
                    break




            self.optimizer.step()
            self.optimizer.zero_grad()
            #gc.collect()



        self.epoch += 1




        return total_loss/data_size  #

    # helper function for testing
    # helper function for testing
    def test(self, a):
        self.model.eval()
        count=0
        test_dic=[5000]
        data_size= test_data_len[args.data_name] #35315 #len(data_dic)                #size of data
        batch_size = self.batchsize                    #batch size
        no_batch= int(data_size/batch_size)    # no of batcg
        number_batch_per_file = int(self.file_len/batch_size)

        with torch.no_grad():
            total_loss = 0.0

            if (a ==0) :
                # for training data
                test_len= train_data_len[args.data_name]
                predictions = torch.zeros(train_data_len[args.data_name], dtype=torch.float, device='cpu')



            elif a==1:
                # for validation data

                predictions = torch.zeros(dev_data_len[args.data_name], dtype=torch.float, device='cpu')
                test_len = dev_data_len[args.data_name]



                # del test_dic
                # gc.collect()
                # fname = os.path.join(dev_dir,'dev_data.pkl')
                # fin = open(fname , 'rb')    # Load from pickle file instead of creating
                # test_dic = pickle.load(fin)
                # fin.close()
                # print(' Test dic len in case of validation data :', len(test_dic))
            elif a ==2:

                #test data sets
                predictions = torch.zeros(test_data_len[args.data_name], dtype=torch.float, device='cpu')
                test_len = test_data_len[args.data_name]




            indices = torch.arange(1, 5, dtype=torch.float, device='cpu')
            for idx in tqdm(range(test_len), desc='Testing epoch  ' + str(self.epoch) + ''):
                if args.run_type == 'debug' and  idx >100:
                    break
                if (idx % self.file_len) == 0 and a==0 :
                    del test_dic
                    gc.collect()
                    filename ='Fold-%d.pkl' % count
                    fname_out = os.path.join(train_dir, filename)
                    fin = open(fname_out , 'rb')
                    test_dic = pickle.load(fin)
                    print("the no of news article pair in training ",len(test_dic))
                    test_dic_key = list(test_dic.keys())
                    test_dic_key.sort()
                    #print(test_dic_key[:200])

                    count=count+1
                elif (idx % self.file_len) ==0 and a==1 :
                    del test_dic
                    gc.collect()
                    filename ='Fold-%d.pkl' % count
                    fname_out = os.path.join(test_dir, filename)
                    fin = open(fname_out , 'rb')
                    test_dic = pickle.load(fin)
                    print("the no of news article pair",len(test_dic))
                    count=count+1

                elif (idx % self.file_len) ==0 and a==2 :
                    del test_dic
                    gc.collect()
                    filename ='Fold-%d.pkl' % count
                    fname_out = os.path.join(test_dir, filename)
                    fin = open(fname_out , 'rb')
                    test_dic = pickle.load(fin)
                    print("the no of news article pair",len(test_dic))
                    count=count+1

                body = test_dic[idx]
                label = test_dic[idx]['stance']   #test_dic[idx]['headline']['label']
                print(" the sample id", idx)
                print(" the sample id label", label)
                if (idx == 50):
                    exit(1)
                target = utils.map_label_to_target(label, self.num_classes)

                output = self.model(body)

                loss = self.criterion(output, target)
                total_loss += loss.item()
                output = output.squeeze().to('cpu')
                value, index= torch.max(output,dim=0)
                predictions[idx] = index
                #print("index",index)
                #print("predictions:", predictions[idx])
                if idx == test_len-1 :
                    del  test_len
                    gc.collect()
                    break

        return total_loss / len(test_dic), predictions
