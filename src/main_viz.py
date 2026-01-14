import time
import torch
import argparse
from pre_process import *
from utils import *
from exp import Experiments
from utils import print_local_time, set_seed


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str,
                    default='environment', help='dataset')
parser.add_argument('--pre_train', type=str,
                    default="bert", help='Pre_trained model')
parser.add_argument('--hidden', type=int, default=64,
                    help='dimension of hidden layers in MLP')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--wandb', type=int, default=0,
                    help="Enable wandb logging")
parser.add_argument('--mixture', type=str, default=None,
                    help="Type of weighting in mixture model")
parser.add_argument('--padmaxlen', type=int, default=128,
                    help='max length of padding')
parser.add_argument('--complex', type=bool, default=False,
                    help="Complex Quantum Taxo")
parser.add_argument('--matrixsize', type=int, default=768,
                    help="Size of density matrix")
parser.add_argument('--negsamples', type=int, default=50,
                    help="Number of negative samples per node")

# Training hyper-parameters
parser.add_argument('--expID', type=int, default=0, help='-th of experiments')
parser.add_argument('--epochs', type=int, default=50, help='training epochs')
parser.add_argument('--batch_size', type=int, default=128,
                    help='training batch size')
parser.add_argument('--lr', type=float, default=9e-5,
                    help='learning rate for pre-trained model')
parser.add_argument('--lr_proj', type=float, default=1e-3,
                    help='learning rate for pre-trained model')
parser.add_argument('--eps', type=float, default=1e-8, help='adamw_epsilon')
parser.add_argument('--optim', type=str, default="adamw", help='Optimizer')
parser.add_argument('--embed_size', type=int, default=8, help='Embedding Size')
parser.add_argument('--wtbce', type=float, default=0.45,
                    help='weight for BCE loss')
parser.add_argument('--wtkl', type=float, default=0.45,
                    help='weight for KL loss')
parser.add_argument('--wtreg', type=float, default=0.1,
                    help='weight for regularization loss')
parser.add_argument('--method', type=str, default='normal',
                    help='Ablations on method')

# Others
parser.add_argument('--cuda', type=bool, default=True,
                    help='use cuda for training')
parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID')
parser.add_argument('--seed', type=int, default=20,
                    help="seed for random generators")
parser.add_argument('--model', type=str, default='bert', help='PLM')
parser.add_argument('--path', type=str, default='../path',
                    help='Path to checkpoint')

args = parser.parse_args()

args.cuda = False  # for inference

exp = Experiments(args)
# exp.visualize_ellipses(
#     tag='test', path=f'../final_result/{args.dataset}/KL_volume_containment_{args.expID}_{args.method}_bert_{args.negsamples}.pt', concepts_to_plot=['desert', 'geophysical environment'], visualize=True)
exp.visualize_ellipses(
    tag='test', path=args.path, concepts_to_plot=['desert', 'geophysical environment'], visualize=True)
