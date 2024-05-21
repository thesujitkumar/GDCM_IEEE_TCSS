import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Multi-head Attention Dual Summarization configrations')
    # data arguments

    parser.add_argument('--run_type', default='final',
                        help='run type : debug/final')
    parser.add_argument('--model_name', default='model_t',
                        help='model_name')

    parser.add_argument('--data', default='data/IOST_data/Parsed_Data',
                        help='path to dataset')

    parser.add_argument('--data_name', default='IOST',
                        help='Name of dataset')
    parser.add_argument('--beta', default= 0.5,  type=float,
                        help='Thresold value for split')
    parser.add_argument('--topk', default= 4,  type=int,
                        help='Number of sentence from top k')
    parser.add_argument('--num_filter', default= 10,  type=int,
                        help='Number of filters for convolution')



    parser.add_argument('--glove', default='data/glove/',
                        help='directory with GLOVE embeddings')
    parser.add_argument('--emb_name',  default='GLOVE',
                        help='Name of embeddings')
    parser.add_argument('--save', default='checkpoints/',
                        help='directory to save checkpoints in') # to save model in model dir ??? edit here
    parser.add_argument('--expname', type=str, default='train',
                        help='Name to identify experiment')
    # model arguments
    parser.add_argument('--input_dim', default=200, type=int,
                        help='Size of input word vector')
    parser.add_argument('--mem_dim', default=100, type=int,
                        help='Size of TreeLSTM cell state')
    parser.add_argument('--hidden_dim', default=100, type=int,
                        help='Size of classifier MLP')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='Number of classes in dataset')
    parser.add_argument('--freeze_embed', action='store_true',
                        help='Freeze word embeddings')
    parser.add_argument('--number_head', default=1, type=int,
                        help='number of total epochs to run')
    # training arguments
    parser.add_argument('--epochs', default=1, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batchsize', default=50, type=int,
                        help='batchsize for optimizer updates')
    parser.add_argument('--lr', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--sparse', action='store_true',
                        help='Enable sparsity for embeddings, \
                              incompatible with weight decay')
    parser.add_argument('--max_num_para', default=5, type = int,
                        help='max number of  paragraph in news article')
    parser.add_argument('--max_num_sent', default=5, type = int,
                            help='max number of  sentence in a paragraph of news article')
    parser.add_argument('--max_num_word', default=12, type = int,
                                help='max number of  word in a sentence of news article')

    parser.add_argument('--file_len', type=int,
                                help=' # of news article each train/val/test file')
    parser.add_argument('--optim', default='adagrad',
                        help='optimizer (default: adagrad)')
    parser.add_argument('--attn', default='sim',
                        help='Method to combine headline and body representations')
    parser.add_argument('--radius', type=int, default= 1,
                        help='Include all neighbors of distance<=radius from a given node')
    # miscellaneous options
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed (default: 123)')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args
