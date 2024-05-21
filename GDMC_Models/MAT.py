##  model with simle LSTM for sentence encoding (without tree lstm)

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from . import Constants
import pandas as pd


class SingleHead_Self_Attention(nn.Module):
    def __init__(self, in_dim):
        super(SingleHead_Self_Attention, self).__init__()
        self.input_dim= in_dim
        self.query_projection= nn.Linear(in_dim,in_dim)
        self.key_projection= nn.Linear(in_dim,in_dim)
        self.value_projection= nn.Linear(in_dim,in_dim)


    def forward(self, sent_mat, indicator):
        Query = self.query_projection(sent_mat)
        Key = self.key_projection(sent_mat)
        valaue = self.value_projection(sent_mat)
        if indicator == 1:
            similalrity= torch.matmul(Query, torch.t(Key))
        elif indicator == 0:
            similalrity= torch.matmul(Query, torch.t(Key))
            similalrity= torch.sub(1,similalrity)
        sim_norm= torch.div(similalrity,(torch.sqrt(torch.tensor(self.input_dim))))
        attention_weight= torch.softmax(sim_norm,dim=0)
        attention_output= torch.matmul(attention_weight,valaue)

        return attention_output, attention_weight







class DocLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, sparsity, freeze, max_num_para, max_num_sent, max_num_word, num_atten_head):
        super(DocLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        if freeze:
            self.emb.weight.requires_grad = False
        self.max_num_para = max_num_para
        self.max_num_sent = max_num_sent
        self.max_num_word_body = max_num_word
        self.max_num_word_head = max_num_word
        self.mem_dim = mem_dim
        self.in_dim = in_dim
        self.att_head = num_atten_head
        self.sentence_head_BILSTM = nn.LSTM(in_dim, mem_dim, 1,bidirectional=True)
        torch.manual_seed(0)
        self.sim_head_sent = nn.Linear((2 * self.mem_dim)*(2*self.mem_dim) , 1)


        if self.att_head == 1:
            self.attention_head1= SingleHead_Self_Attention(2*self.mem_dim)
            self.concnate_head_output= nn.Linear(((2*self.mem_dim)*self.att_head), (2*self.mem_dim))
        elif self.att_head == 8 :
            self.attention_head1= SingleHead_Self_Attention(2*self.mem_dim)
            self.attention_head2= SingleHead_Self_Attention(2*self.mem_dim)
            self.attention_head3= SingleHead_Self_Attention(2*self.mem_dim)
            self.attention_head4= SingleHead_Self_Attention(2*self.mem_dim)
            self.attention_head5= SingleHead_Self_Attention(2*self.mem_dim)
            self.attention_head6= SingleHead_Self_Attention(2*self.mem_dim)
            self.attention_head7= SingleHead_Self_Attention(2*self.mem_dim)
            self.attention_head8= SingleHead_Self_Attention(2*self.mem_dim)
            self.concnate_head_output= nn.Linear(((2*self.mem_dim)*self.att_head), (2*self.mem_dim))
        else:
            self.attention_head1= SingleHead_Self_Attention(2*self.mem_dim)
            self.attention_head2= SingleHead_Self_Attention(2*self.mem_dim)
            self.concnate_head_output= nn.Linear(((2*self.mem_dim)*self.att_head), (2*self.mem_dim))

        self.multihead_to_feature_map= nn.Linear((max_num_sent*2*mem_dim),(2*mem_dim))

        self.sent_pad =   torch.randn(1, 2* mem_dim)
        self.para_pad = torch.zeros(1, 2* mem_dim) #torch.Size([1, 1, 150])
        self.word_pad= torch.randn(1, in_dim)


    def forward(self, body):

        rsent = body['headline']['rsent']

        rinputs_list=[]

        for word in rsent:
            # print("input word to lstm",word)
            rinputs_list.append(self.emb(word).view(1, self.in_dim))

        if len(rsent) < self.max_num_word_body:
             rinputs_list += [ self.word_pad] * (self.max_num_word_body - len(rinputs_list))

        seq_head_input=torch.cat(rinputs_list[:self.max_num_word_body],0)
        #print("dimension of headline input ",seq_head_input.shape)
        rinputs=self.emb(rsent)
        head_o, (head_h,head_c) = self.sentence_head_BILSTM(seq_head_input.contiguous().view(self.max_num_word_body, 1, self.in_dim))
        #print("dimension of headline input using BILSTM ",head_h.shape)
        head_hid_2d=head_h.view(2,100)
        h_left=head_hid_2d[0]
        h_right=head_hid_2d[1]
        Bi_Head_h=torch.cat((h_left,h_right),0)
        del rinputs_list
        body=body['body_list']
        count=0
        sent_hidden_list = []
        for p_id in body:
            count=count+1
            if count > self.max_num_para:   # condition for only two paragrphs
               break

            for s_id, sentence in enumerate(body[p_id]):
                lsent = sentence
                linputs = self.emb(lsent)

                linputs_list=[]
                for word in lsent:
                    linputs_list.append(self.emb(word).view(1, self.in_dim))

                if len(lsent) < self.max_num_word_body:
                    linputs_list += [ self.word_pad] * (self.max_num_word_body - len(linputs_list))

                seq_head_input=torch.cat(linputs_list[:self.max_num_word_body],0)

                body_sent_o,(body_sent_h,body_sent_c) = self.sentence_head_BILSTM(seq_head_input.contiguous().view(self.max_num_word_body, 1, self.in_dim))

                body_hid_2d=body_sent_h.view(2,100)
                body_sent_left=body_hid_2d[0]
                body_sent_right=body_hid_2d[1]
                Bi_body_sent_h=torch.cat((body_sent_left,body_sent_right),0)
                del linputs_list
                sent_hidden_list.append(Bi_body_sent_h.view(1,self.mem_dim*2))

        head_sent_similarity=[]
        for vec in sent_hidden_list:
            Bi_Head_h_Transpose= Bi_Head_h.view(2*self.mem_dim,1) #torch.t(Bi_Head_h)
            Head_sent_mul= torch.matmul(Bi_Head_h_Transpose, vec.view(1,(2*self.mem_dim)))
            sim_input= torch.flatten(Head_sent_mul)
            sim_output=torch.sigmoid(self.sim_head_sent(sim_input))

            head_sent_similarity.append(sim_output)

        head_sent_sim_vec=torch.cat(head_sent_similarity,0)


        sent_prob=torch.softmax(head_sent_sim_vec,dim=0)

        sent_hidden_mat=torch.cat((sent_hidden_list[:len(sent_hidden_list)]),0)
        attend_sent_mat=torch.mul((sent_prob.view(1,len(head_sent_similarity))),torch.t(sent_hidden_mat))
        del head_sent_similarity

        attend_sent_mat=torch.t(attend_sent_mat)

        high_sim_sent_hidden=[]
        low_sim_sent_hidden=[]
        for i in range(len(sent_hidden_list)):

            if head_sent_sim_vec[i] >=0.5:

                # print(attend_sent_mat[i].shape)
                high_sim_sent_hidden.append(attend_sent_mat[i].view(1,2*self.mem_dim))
            else:
                #print(attend_sent_mat[i])
                low_sim_sent_hidden.append(attend_sent_mat[i].view(1,2*self.mem_dim))




        "Detils number of sentence in high and low"
        a_high=len(high_sim_sent_hidden)
        b_low=(len(low_sim_sent_hidden))
        if len(high_sim_sent_hidden) == 0:
            value, index = torch.max(head_sent_sim_vec,dim=0)
            high_sim_sent_decoder = attend_sent_mat[index]
            high_sim_sent_decoder=high_sim_sent_decoder.view(1,2*self.mem_dim)

        else:
            high_sim_sent_decoder= torch.cat((high_sim_sent_hidden[:len(high_sim_sent_hidden)]), 0)
        "Low similairty Decoder Side"
        if len(low_sim_sent_hidden) == 0:
            value, index = torch.min(head_sent_sim_vec,dim=0)
            low_sim_sent_decoder = attend_sent_mat[index]
            low_sim_sent_decoder= low_sim_sent_decoder.view(1,2*self.mem_dim)

        else:
            low_sim_sent_decoder= torch.cat((low_sim_sent_hidden[:len(low_sim_sent_hidden)]), 0)


        if self.att_head == 1:
            att1_op, att1_w= self.attention_head1(high_sim_sent_decoder,1)
            head_concat= att1_op
        elif self.att_head == 8:
            att1_op, att1_w= self.attention_head1(high_sim_sent_decoder,1)
            att2_op, att2_w= self.attention_head2(high_sim_sent_decoder,1)
            att3_op, att3_w= self.attention_head3(high_sim_sent_decoder,1)
            att4_op, att4_w= self.attention_head4(high_sim_sent_decoder,1)
            att5_op, att5_w= self.attention_head5(high_sim_sent_decoder,1)
            att6_op, att6_w= self.attention_head6(high_sim_sent_decoder,1)
            att7_op, att7_w= self.attention_head7(high_sim_sent_decoder,1)
            att8_op, att8_w= self.attention_head8(high_sim_sent_decoder,1)
            head_concat= torch.cat((att1_op,att2_op,att3_op,att4_op,att5_op,att6_op,att7_op,att8_op),1)
        else:
            att1_op, att1_w= self.attention_head1(high_sim_sent_decoder,1)
            att2_op, att2_w= self.attention_head2(high_sim_sent_decoder,1)
            head_concat= torch.cat((att1_op,att2_op),1)
        high_multihead_output= self.concnate_head_output(head_concat)


        number_of_sent_vector= high_multihead_output.shape[0]
        # print(number_of_sent_vector)

        if self.max_num_sent> number_of_sent_vector:
            sent_pad= torch.zeros((self.max_num_sent -number_of_sent_vector),2*self.mem_dim)
            final_high_rep_mat=torch.cat((high_multihead_output,sent_pad),dim=0)
        else:
            indices=[range(self.max_num_sent)]
            final_high_rep_mat= high_multihead_output[indices]

        final_high_rep_vec= torch.flatten(final_high_rep_mat)
        high_sent_feature= self.multihead_to_feature_map(final_high_rep_vec)
        if self.att_head == 1:
            att1_op, att1_w= self.attention_head1(low_sim_sent_decoder,0)
            head_concat= att1_op
        elif self.att_head == 8:
            att1_op, att1_w= self.attention_head1(low_sim_sent_decoder,0)
            att2_op, att2_w= self.attention_head2(low_sim_sent_decoder,0)
            att3_op, att3_w= self.attention_head3(low_sim_sent_decoder,0)
            att4_op, att4_w= self.attention_head4(low_sim_sent_decoder,0)
            att5_op, att5_w= self.attention_head5(low_sim_sent_decoder,0)
            att6_op, att6_w= self.attention_head6(low_sim_sent_decoder,0)
            att7_op, att7_w= self.attention_head7(low_sim_sent_decoder,0)
            att8_op, att8_w= self.attention_head8(low_sim_sent_decoder,0)
            head_concat= torch.cat((att1_op,att2_op,att3_op,att4_op,att5_op,att6_op,att7_op,att8_op),1)
        else:
            att1_op, att1_w= self.attention_head1(low_sim_sent_decoder,0)
            att2_op, att2_w= self.attention_head2(low_sim_sent_decoder,0)
            head_concat= torch.cat((att1_op,att2_op),1)
        low_multihead_output= self.concnate_head_output(head_concat)
        # print("final multihead output",low_multihead_output.shape)
        number_of_sent_vector= low_multihead_output.shape[0]
        # print(number_of_sent_vector)

        if self.max_num_sent> number_of_sent_vector:
            sent_pad= torch.zeros((self.max_num_sent -number_of_sent_vector),2*self.mem_dim)
            final_low_rep_mat=torch.cat((low_multihead_output,sent_pad),dim=0)
        else:
            indices=[range(self.max_num_sent)]
            final_low_rep_mat= low_multihead_output[indices]

        final_low_rep_vec= torch.flatten(final_low_rep_mat)
        low_sent_feature= self.multihead_to_feature_map(final_low_rep_vec)
        # print(low_sent_feature.shape)

        return   Bi_Head_h, low_sent_feature,high_sent_feature




# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes, max_num_word_head):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_num_word_head= max_num_word_head


        self.wh = nn.Linear((14 * self.mem_dim) , self.hidden_dim)  # for only deep feature.
        self.wp = nn.Linear(self.hidden_dim, 2)

    def forward(self, head, low, high):

        head= head.view(2*self.mem_dim,1)
        low=low.view(2*self.mem_dim,1)
        high=high.view(2*self.mem_dim,1)


        "Summary related features"
        low_head_diff = torch.abs(torch.add(low, - head))
        high_head_diff = torch.abs(torch.add(high, - head))
        #low_high_diff = torch.abs(torch.add(low, - high))

        low_head_angle = torch.mul(low, head)
        high_head_angle = torch.mul(high, head)
        #low_high_angle = torch.mul(low, high)

        feature_vec=torch.cat((low_head_diff,high_head_diff,low_head_angle,high_head_angle, low,high,head),0)







        out = torch.sigmoid(self.wh(torch.t(feature_vec))) # for model with only deep feature
        out =self.wp(out) # No softmax
        return out




# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim,  sparsity, freeze, num_classes, \
        max_num_para, max_num_sent, max_num_word,  num_filter, num_head):
        super(SimilarityTreeLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.doclstm = DocLSTM(vocab_size, in_dim, mem_dim, sparsity, freeze, max_num_para, max_num_sent, max_num_word, num_head)
        self.similarity = Similarity(mem_dim, hidden_dim, num_classes, max_num_word)
    def forward(self, body):
        head, low, high= self.doclstm(body)
        output = self.similarity(head, low, high)
        return output
