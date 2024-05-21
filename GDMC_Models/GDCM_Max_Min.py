##  model with simle LSTM for sentence encoding (without tree lstm)

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from . import Constants
import pandas as pd
from torch_geometric.nn import GAT
import networkx as nx
import numpy as np

import random

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout_prob=0):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.out_projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, queries, keys, values, indicator, mask=None):
        batch_size = queries.size(0)

        # Project queries, keys, and values
        queries_proj = self.query_projection(queries)
        keys_proj = self.key_projection(keys)
        values_proj = self.value_projection(values)

        # Split queries, keys, and values into num_heads
        queries_split = queries_proj.view(batch_size, -1, self.num_heads, self.embed_dim//self.num_heads).transpose(1, 2)
        keys_split = keys_proj.view(batch_size, -1, self.num_heads, self.embed_dim//self.num_heads).transpose(1, 2)
        values_split = values_proj.view(batch_size, -1, self.num_heads, self.embed_dim//self.num_heads).transpose(1, 2)

        # Compute scores using dot product attention
        scores = torch.matmul(queries_split, keys_split.transpose(-2, -1)) / (self.embed_dim//self.num_heads)**0.5

#         # Apply mask (if provided)
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        # Apply softmax to get attention weights
        # attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = torch.sigmoid(scores)
        if indicator == 0:
            attention_weights = torch.softmax(attention_weights, dim=-1)
            # print("attention weight of positive context set :",attention_weights)

        else:
            attention_weights= torch.sub(1, attention_weights)
            attention_weights = torch.softmax(attention_weights, dim=-1)
            # print("attention weight of negative context set :",attention_weights)





        # print(attention_weights)
        # z=torch.sum(attention_weights,-1)
        # print("summ of all attention weight", z)
        # exit(1)

        # Apply dropout to attention weights
        attention_weights_dropout = self.dropout(attention_weights)

        # Compute attention output
        attention_output = torch.matmul(attention_weights_dropout, values_split)

        # Concatenate attention output from all heads
        attention_output_concat = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # Project attention output
        attention_output_proj = self.out_projection(attention_output_concat)

        return attention_output_proj






class DocLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, sparsity, freeze, max_num_para, max_num_sent, max_num_word, num_atten_head, thresold, no_radius):
        super(DocLSTM, self).__init__()

        self.max_num_para = max_num_para
        self.max_num_sent = max_num_sent
        self.thresold= thresold
        self.max_num_word_body = max_num_word
        self.max_num_word_head = max_num_word
        self.att_head = num_atten_head
        self.mem_dim = mem_dim
        self.in_dim = in_dim
        self.radius = no_radius




        self.conv1 = GAT(self.in_dim, 128,2) # dataset.num_features
        self.conv2 = GAT(128,64,2 )
        self.conv3 = GAT(64,32,2)


        self.head_conv1 = GAT(self.in_dim, 128,2) # dataset.num_features
        self.head_conv2 = GAT(128,32,2 )
        self.multihead_attn = MultiHeadAttention(self.att_head, 96, 0.2)








        torch.manual_seed(0)



    def forward(self, body):

        b_edge_list= body["body"]["adj"]
        b_feature= body["body"]["feature"]
        "Define the the maximum number node in the graph."

        body_matrix= torch.cat(b_feature,0)

        h_edge_list= body["head"]["adj"]

        h_feature= body["head"]["feature"]

        head_matrix= torch.cat(h_feature,0)

        sim= torch.matmul(head_matrix, torch.t(body_matrix))

        out, inds= torch.topk(sim,k=3, dim=1,largest=True)

        inds= torch.flatten(torch.t(inds), start_dim=0).tolist()

        inds=  list(dict.fromkeys(inds))   #inds.unique_everseen() #set(inds)

        if len(inds)>12:
            indexs=inds[0:12]
        else:
            indexs=inds
            rand_num =  random.sample(range(1, len(b_feature)),(len(b_feature)-len(indexs)))

            for i in rand_num:
                if i not in indexs:
                    indexs.append(i)
                    if len(indexs)== 12:
                        break

        "Find Negative Nodes Index"

        out, inds= torch.topk(sim,k=3, dim=1,largest=False)

        inds= torch.flatten(torch.t(inds), start_dim=0).tolist()

        inds=  list(dict.fromkeys(inds))   #inds.unique_everseen() #set(inds)

        if len(inds)>12:
            Neg_indexs=inds[0:12]
        else:
            Neg_indexs=inds
            rand_num =  random.sample(range(1, len(b_feature)),(len(b_feature)-len(Neg_indexs)))

            for i in rand_num:
                if i not in Neg_indexs:
                    Neg_indexs.append(i)
                    if len(Neg_indexs)== 12:
                        break

        "Why transpose is required ??? in edge list"
        b= self.conv1(body_matrix, torch.t(b_edge_list))
        # print("shape of hidden state",b.shape)
        b = b.tanh()
        # print("shape of hidden state",b.shape)
        b= self.conv2(b, torch.t(b_edge_list))
        b = b.tanh()
        # print("secong GAT Layer ::::: shape of hidden state",b.shape)
        b= self.conv3(b, torch.t(b_edge_list))
        b = b.tanh()

        h= self.head_conv1(head_matrix,torch.t(h_edge_list))
        h = h.tanh()
        h= self.head_conv2(h,torch.t(h_edge_list))
        h = h.tanh()

        "Query body nodes and constrcut subgraphs"

        body_Edges= b_edge_list.tolist()
        # print(body_Edges)
        body_graph =nx.Graph(body_Edges)
        # print(len((list(body_graph.nodes))))

        "Subgraph Embedding for Positive Nodes"


        Sub_Graph_node_list=[]
        for i in range(len(indexs)):
            a = nx.ego_graph(body_graph,indexs[i], radius = self.radius)
            Sub_Graph_node_list.append(torch.tensor(list(a.nodes)))

        # print("nodes of subgraph are as follow",Sub_Graph_node_list)
        subgraph_mat=[]
        for i in range(len(Sub_Graph_node_list)):
            subgraph_mat.append(b[Sub_Graph_node_list[i]] )
            # print(subgraph_mat[i].shape)
        # print(len(subgraph_mat))
        subgraph_pooled=[]
        for i in range(len(subgraph_mat)):
            s_max, ind= torch.max(subgraph_mat[i],dim=0)
            s_min, ind= torch.min(subgraph_mat[i],dim=0)
            s_sum= torch.mean(subgraph_mat[i],dim=0)
            sub_graph_features=torch.cat((s_max,s_min,s_sum),0)
            subgraph_pooled.append(sub_graph_features.view(1,96))

        "Subgraph Embedding for Negative Nodes"


        Neg_Sub_Graph_node_list=[]
        for i in range(len(Neg_indexs)):
            # print("Neighbours in body graph for keywords of headline",indexs[i])
            a = nx.ego_graph(body_graph,Neg_indexs[i], radius = self.radius)
            # print(a.nodes)
            Neg_Sub_Graph_node_list.append(torch.tensor(list(a.nodes)))

        # print("nodes of subgraph are as follow",Sub_Graph_node_list)
        neg_subgraph_mat=[]
        for i in range(len(Neg_Sub_Graph_node_list)):
            neg_subgraph_mat.append(b[Neg_Sub_Graph_node_list[i]] )
            # print(subgraph_mat[i].shape)
        # print(len(subgraph_mat))
        Neg_subgraph_pooled=[]
        for i in range(len(neg_subgraph_mat)):
            s_max, ind= torch.max(neg_subgraph_mat[i],dim=0)
            s_min, ind= torch.min(neg_subgraph_mat[i],dim=0)
            s_sum= torch.mean(neg_subgraph_mat[i],dim=0)
            sub_graph_features=torch.cat((s_max,s_min,s_sum),0)
            Neg_subgraph_pooled.append(sub_graph_features.view(1,96))

        s_max, ind= torch.max(h,dim=0)
        s_min, ind= torch.min(h,dim=0)
        s_sum= torch.mean(h,dim=0)

        head_pooled_feature=torch.cat((s_max,s_min,s_sum),0).view(1,96)
        # print(head_pooled_feature.shape)
        "Multi-Head Representation for Positive side"
        subgrap_rep_mat=torch.cat(subgraph_pooled,0)
        num_sugraph= subgrap_rep_mat.shape[0]
        pos_body_multi_head_rep =  self.multihead_attn(head_pooled_feature.view(1,1,96),
         subgrap_rep_mat.view(1,num_sugraph,96), subgrap_rep_mat.view(1,num_sugraph,96), 0).view(1,96)
        # self.multihead_attn(head_pooled_feature, subgrap_rep_mat,subgrap_rep_mat)

        "Multi-Head Representation for Negative side"
        Neg_subgraph_pooled = torch.cat(Neg_subgraph_pooled,0)
        num_sugraph = Neg_subgraph_pooled.shape[0]
        Neg_body_multi_head_rep = self.multihead_attn(head_pooled_feature.view(1,1,96),
         Neg_subgraph_pooled.view(1,num_sugraph,96), Neg_subgraph_pooled.view(1,num_sugraph,96), 1).view(1,96)

        # self.multihead_attn(head_pooled_feature, Neg_subgraph_pooled,Neg_subgraph_pooled)
        # # print("Negative multi_attention output",Neg_body_multi_head_rep.shape)

        "Global Representation from body Graph"
        s_max, ind= torch.max(b,dim=0)
        s_min, ind= torch.min(b,dim=0)
        s_sum= torch.mean(b,dim=0)

        Global_body_graph_rep=torch.cat((s_max,s_min,s_sum),0).view(1,96)

        del Sub_Graph_node_list, subgraph_mat, subgraph_pooled , Neg_Sub_Graph_node_list, neg_subgraph_mat, Neg_subgraph_pooled
        return  head_pooled_feature, pos_body_multi_head_rep, Neg_body_multi_head_rep, Global_body_graph_rep




# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes, max_num_word_head, attn):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_num_word_head= max_num_word_head

        self.method = attn


        if self.method == 'sim':
            # print("i am from sim")
            self.layer_1 = nn.Linear(1152, 512)  # for only deep feature.
            self.layer_2 = nn.Linear(512, 128)
            self.layer_3 = nn.Linear(128, 2)

        elif self.method == 'additive':
            # print("I am from additive")
            self.v = torch.nn.Parameter(
            torch.FloatTensor(96).uniform_(-0.1, 0.1))
            self.W_1 = torch.nn.Linear(96, 96)
            self.W_2 = torch.nn.Linear(96, 96)

            self.layer_1 = nn.Linear(192, 64)  # for only deep feature.
            self.layer_2 = nn.Linear(64, 2)


        elif self.method == 'scaled':
            # print(" I am from scaled")
            self.wh = nn.Linear(192, self.hidden_dim)  # for only deep feature.
            self.wp = nn.Linear(self.hidden_dim, 2)


    def forward(self,  head, pos, neg, body):
        # print("the shape of head", head.shape)
        # print("the shape of multi", pos.shape)
        # print("the shape of multi", neg.shape)
        # print("the shape of multi", body.shape)
        if self.method == 'sim':

            head_pos_angle = torch.mul(head, pos)
            # print(head_multi_angle.shape)
            head_pos_diff = torch.abs(torch.add(head, pos))
            # print(head_multi_diff.shape)

            head_body_angle = torch.mul(head,body)
            # print(head_body_angle.shape)
            head_body_diff = torch.abs(torch.add(head, body))
            # print(head_body_diff.shape)

            head_neg_angle = torch.mul(head, neg)
            # print(head_multi_angle.shape)
            head_neg_diff = torch.abs(torch.add(head, neg))
            # print(head_multi_diff.shape)

            pos_neg_angle = torch.mul(pos, neg)
            # print(head_multi_angle.shape)
            pos_neg_diff = torch.abs(torch.add(pos, neg))
            # print(head_multi_diff.shape)

            feature_vec=torch.cat((head, pos, neg, body,  head_pos_angle, head_pos_diff, head_body_angle,
            head_body_diff, head_neg_angle, head_neg_diff, pos_neg_angle, pos_neg_diff  ), 1)
            # print(feature_vec.shape)

            out = torch.sigmoid(self.layer_1(feature_vec))
            out = torch.sigmoid(self.layer_2(out))
            out = self.layer_3(out)


        elif self.method == 'additive':
            # print("I am from additive")

            weights = self.W_1(head) + self.W_2(pos)  # [seq_length, decoder_dim]
            # print("shape of weights", weights.shape)
            alpha_1 = torch.tanh(weights) @ self.v
            # print(alpha_1)
            weights = self.W_1(head) + self.W_2(neg)  # [seq_length, decoder_dim]
            # print("shape of weights", weights.shape)
            alpha_2 = torch.tanh(weights) @ self.v


            weights = self.W_1(head) + self.W_2(body)  # [seq_length, decoder_dim]
            # print("shape of weights", weights.shape)
            alpha_3 = torch.tanh(weights) @ self.v
            # print(alpha_2)

            attn_weighs = torch.softmax(torch.cat((alpha_1,alpha_2, alpha_3),0),0)
            # print(attn_weighs)
            # print(attn_weighs)

            rep_matrix = torch.cat((pos, neg,body),0)
            # print(rep_matrix.shape)
            final_rep_body = torch.matmul(attn_weighs,rep_matrix).view(1,96)
            # print(final_rep_body.shape)

            feature= torch.cat((head,final_rep_body),1)
            # print(feature.shape)

            out = torch.sigmoid(self.layer_1(feature))
            out = torch.sigmoid(self.layer_2(out))



        elif self.method == 'scaled':

            rep_matrix= torch.cat((head, pos, neg, body),0)
            # print(rep_matrix.shape)
            sim=torch.matmul(head,torch.t(rep_matrix))
            view_score =torch.softmax(sim,1)
            final_rep= torch.matmul(view_score, rep_matrix )
            # print("final_vec representation",final_rep.shape)
            feature_vec=torch.cat((head,final_rep ),1)
            # print(feature_vec.shape)
            # exit(0)

            out = torch.sigmoid(self.wh(feature_vec)) # for model with only deep feature
            out =self.wp(out) # No softmax

        return out




# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, hidden_dim, sparsity, freeze, num_classes, \
        max_num_para, max_num_sent, max_num_word, num_filter, num_head, thresold, attn, radius):
        super(SimilarityTreeLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.doclstm = DocLSTM( in_dim, mem_dim, sparsity, freeze, max_num_para, max_num_sent, max_num_word, num_head, thresold,  radius)
        self.similarity = Similarity(mem_dim, hidden_dim, num_classes, max_num_word, attn)
    def forward(self, body):
        head, pos_multi_head_rep, neg_multi_head_rep, body_rep= self.doclstm(body)
        output = self.similarity(head, pos_multi_head_rep, neg_multi_head_rep, body_rep)
        return output
