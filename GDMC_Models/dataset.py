import os
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

from . import Constants
from .tree import Tree


# Dataset class for SICK dataset
class Dataset(data.Dataset):
    def __init__(self, path, vocab, num_classes):
        super(Dataset, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes

        self.lsentences, self.lsentences_word = self.read_sentences(os.path.join(path, 'a.txt')) #lsentences (Sentences of body)
        self.rsentences, self.rsentences_word = self.read_sentences(os.path.join(path, 'b.txt')) # Sentences of body



        "Tree generations code"
        # self.ltrees = self.read_trees(os.path.join(path, 'a.parents'), self.lsentences_word )
        # self.rtrees = self.read_trees(os.path.join(path, 'b.parents'), self.rsentences_word)
        "This tree constructions code is not required"
        self.labels = self.read_labels(os.path.join(path, 'label.txt'))

        self.size = self.labels.size(0)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        #ltree = deepcopy(self.ltrees[index])
        #rtree = deepcopy(self.rtrees[index])
        lsent = deepcopy(self.lsentences[index])
        rsent = deepcopy(self.rsentences[index])
        label = deepcopy(self.labels[index])
        return (lsent, rsent, label)#(ltree, lsent, rtree, rsent, label)

    def get_headline(self, index):
        #rtree = deepcopy(self.rtrees[index]) "tree of sentenc"
        rsent = deepcopy(self.rsentences[index])
        label = deepcopy(self.labels[index])
        return (rsent,label)            #(rtree, rsent, label)

    def get_sentence_body(self, index):
        #ltree = deepcopy(self.ltrees[index]) # tree of sentence
        lsent = deepcopy(self.lsentences[index])
        return lsent        #(ltree, lsent)

    def read_sentences(self, filename):
        with open(filename, 'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        with open(filename, 'r') as f:
            sentences_word = [ line.split() for line in tqdm(f.readlines())]
        return sentences, sentences_word

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.tensor(indices, dtype=torch.long, device='cpu')

    # def read_trees(self, filename, sentences_word):
    #     print(len(sentences_word))
    #     with open(filename, 'r') as f:
    #         trees = [self.read_tree(line.strip(), sentences_word[index], index) for index, line in tqdm(enumerate(f.readlines()))]
    #     return trees

    # def read_tree(self, line, word_list, sent_id):
    #     parents = list(map(int, line.split())) # split the line and convert each item to integer
    #     trees = dict() # A ahshtable to store each node object of tree class
    #     root = None
    #     for i in range(1, len(parents) + 1):
    #         if i - 1 not in trees.keys() and parents[i - 1] != -1:
    #             idx = i
    #             prev = None
    #             while True:
    #                 parent = parents[idx - 1]
    #                 if parent == -1:
    #                     break
    #                 if idx-1 in trees.keys():
    #                     tree = trees[idx -1]
    #                 else:
    #                     try:
    #                         tree = Tree(idx-1, word_list[idx-1])
    #                         trees[idx - 1] = tree
    #                     except Exception as e:
    #                         print(idx-1, word_list, sent_id)
    #
    #
    #                 if prev is not None:
    #                     tree.add_child(prev)
    #
    #                 # tree.idx = idx - 1
    #                 if parent - 1 in trees.keys():
    #                     trees[parent - 1].add_child(tree)
    #                     break
    #                 elif parent == 0:
    #                     root = tree
    #                     root.sent_id = sent_id
    #                     break
    #                 else:
    #                     prev = tree
    #                     idx = parent
    #     return root

    def read_labels(self, filename):
        with open(filename, 'r') as f:
            lis = []
            for label in f.readlines():
               cur_label = label.strip()
               if cur_label == "0":
                   lis.append(0)
               elif cur_label == "1":
                   lis.append(1)
               elif cur_label == "2":
                   lis.append(2)
               elif cur_label == "3":
                   lis.append(3)
               else:
                   print("wrong  output")
            labels = torch.tensor(lis, dtype=torch.float, device='cpu')
        print("labels",labels.shape)
        return labels
