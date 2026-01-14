import time
import os.path
import torch
import argparse
from pre_process import *
from utils import *
from exp import Experiments
import torch.multiprocessing as tmp
from utils import print_local_time, set_seed
import wandb

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["WANDB_MODE"] = "online"

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str,
                    default='environment', help='dataset')
parser.add_argument('--pre_train', type=str,
                    default="bert", help='Pre_trained model')
parser.add_argument('--hidden', type=int, default=64,
                    help='dimension of hidden layers in MLP')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout')
parser.add_argument('--wandb', type=int, default=1,
                    help="Enable wandb logging")
parser.add_argument('--mixture', type=str, default=None,
                    help="Type of weighting in mixture model")
parser.add_argument('--padmaxlen', type=int, default=30,
                    help='max length of padding')
parser.add_argument('--complex', type=bool, default=False,
                    help="Complex Quantum Taxo")
parser.add_argument('--matrixsize', type=int, default=768,
                    help="Size of density matrix")
parser.add_argument('--negsamples', type=int, default=50,
                    help="Number of negative samples per node")
parser.add_argument('--model', type=str, default='bert',
                    help='Pretained Language Model')

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
parser.add_argument('--accumulation_steps', type=int, default=1,
                    help='Increase accumulation steps to use Gradient Accumulation')
parser.add_argument('--lam', type=float, default=0.1, help='Weight on KL(P/C)')
parser.add_argument('--C', type=float, default=1.5,
                    help='Scaling for the volume difference ratio')
parser.add_argument('--is_multi_parent', type=float, default=False,
                    help='Does a single child have multiple parents in the dataset?', required=True)

# Others
parser.add_argument('--cuda', type=bool, default=True,
                    help='use cuda for training')
parser.add_argument('--gpu_id', type=int, default=1,
                    help='GPU ID for training.')
parser.add_argument('--seed', type=int, default=20,
                    help="seed for random generators")
parser.add_argument('--method', type=str, default='normal',
                    help='Experiment method conducted')

start_time = time.time()
print("Start time at : ")
print_local_time()

args = parser.parse_args()


def experiment(args):
    torch.set_float32_matmul_precision('high')
    args.cuda = torch.cuda.is_available() and args.cuda == True

    if args.wandb == 1:
        wandb.init(
            project='GaussBox',

            name=f'{args.expID}-{args.dataset}-{args.method}-{args.model}',
            config=args
        )

    # args.expID = wandb.run.id
    # wandb.config.update({'expID': args.expID}, allow_val_change=True)

    if args.cuda:
        torch.cuda.set_device(args.gpu_id)

    print(args.cuda)

    print(args)

    set_seed(args.seed)
    if not os.path.isfile(os.path.join("../data/", args.dataset, "processed", "taxonomy_data_"+str(args.expID)+str(args.negsamples)+"_.pkl")):
        if args.dataset == 'computer_science' or args.dataset == 'psychology' or args.dataset == 'mesh' or args.dataset == 'wordnet_verb' or args.dataset == 'semeval_food':
            create_multiparent_data(args)
        else:
            create_data(args)

    # args.expID = wandb.run

    exp = Experiments(args)
    exp.train()
    """Train the model"""

    # exp.predict(tag="test")
    # exp.save_prediction()

    wandb.finish()

    print("Time used :{:.01f}s".format(time.time()-start_time))
    print("End time at : ")
    print_local_time()
    print("************END***************")


if __name__ == '__main__':
    # Experiments
    torch.set_float32_matmul_precision('high')

    experiment(args)
