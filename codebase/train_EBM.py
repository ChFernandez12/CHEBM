import argparse
import time
import sys
import os
from tqdm import tqdm
import random

import torch
from torch.optim import Adam
from texttable import Texttable
from torch.utils.data import DataLoader, random_split, Subset
from distutils.util import strtobool
import numpy as np
from torch.nn import BCELoss

#from preprocess_data import transform_qm9, transform_zinc250k
from model import *
#from preprocess_data.data_loader import NumpyTupleDataset
#from util import *

from model.modules import *
from utils import arg_parser, logger, data_loader, forward_pass_and_eval
from model import utils, model_loader

import datetime

from torch.utils.tensorboard import SummaryWriter


### Args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed to use')
parser.add_argument('--data_name', type=str, default='qm9', choices=['qm9', 'zinc250k'], help='Dataset name')
parser.add_argument('--hidden', type=int, default=64, help='Hidden dimension')
parser.add_argument('--depth', type=int, default=2, help='Number of graph conv layers')
parser.add_argument('--add_self', type=strtobool, default='false', help='Add shortcut in graphconv')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')
parser.add_argument('--swish', type=strtobool, default='true', help='Use swish as activation function')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size during training')
parser.add_argument('--shuffle', type=strtobool, default='true', help='Shuffle the data batch')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers in the data loader')
parser.add_argument('--c', type=float, default=0.5, help='Dequantization using uniform distribution of [0,c)')
parser.add_argument('--alpha', type=float, default=1.0, help='Weight for energy magnitudes regularizer')
parser.add_argument('--step_size', type=int, default=10, help='Step size in Langevin dynamics')
parser.add_argument('--sample_step', type=int, default=30, help='Number of sample step in Langevin dynamics')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--noise', type=float, default=0.005, help='The standard variance of the added noise during Langevin Dynamics')
parser.add_argument('--clamp', type=strtobool, default='true', help='Clamp the data/gradient during Langevin Dynamics')
parser.add_argument('--wd', type=float, default=0, help='Weight Decay')
parser.add_argument('--max_epochs', type=int, default=100, help='Maximum training epochs')
parser.add_argument('--save_dir', type=str, default='trained_models/qm9', help='Location for saving checkpoints')
parser.add_argument('--save_interval', type=int, default=1, help='Interval (# of epochs) between saved checkpoints')
# parser.add_argument('--num_atoms', type=int, default=5, help='Number of atoms')
# parser.add_argument('--cuda', type=strtobool, default='false', help='Use cuda')
# parser.add_argument('--unobserved', type=int, default=0, help='Unobserved')
# parser.add_argument('--model_unobserved', type=int, default=0, help='Model_unobserved')
parser.add_argument('--suffix', type=str, default='_springs5', help='Dataset name')
# parser.add_argument('--batch_size_multiGPU', type=int, default=128, help='Batch size during training')
# parser.add_argument('--datadir', type=str, default='data', help='Dataset name')
# parser.add_argument('--load_temperatures', type=strtobool, default='false', help='Use cuda')
# parser.add_argument("--training_samples", type=int, default=0, help="If 0 use all data available, otherwise reduce number of samples to given number")

parser.add_argument(
    "--test_samples", type=int, default=0,
    help="If 0 use all data available, otherwise reduce number of samples to given number"
)
parser.add_argument(
    "--GPU_to_use", type=int, default=None, help="GPU to use for training"
)

############## training hyperparameter ##############
parser.add_argument(
    "--epochs", type=int, default=500, help="Number of epochs to train."
)

parser.add_argument(
    "--lr_decay",
    type=int,
    default=200,
    help="After how epochs to decay LR by a factor of gamma.",
)
parser.add_argument("--gamma", type=float, default=0.5, help="LR decay factor.")
parser.add_argument(
    "--training_samples", type=int, default=0,
    help="If 0 use all data available, otherwise reduce number of samples to given number"
)

parser.add_argument(
    "--prediction_steps",
    type=int,
    default=10,
    metavar="N",
    help="Num steps to predict before re-using teacher forcing.",
)

############## architecture ##############
parser.add_argument(
    "--encoder_hidden", type=int, default=256, help="Number of hidden units."
)
parser.add_argument(
    "--decoder_hidden", type=int, default=256, help="Number of hidden units."
)
parser.add_argument(
    "--encoder",
    type=str,
    default="mlp",
    help="Type of path encoder model (mlp or cnn).",
)
parser.add_argument(
    "--decoder",
    type=str,
    default="mlp",
    help="Type of decoder model (mlp, rnn, or sim).",
)
parser.add_argument(
    "--prior",
    type=float,
    default=1,
    help="Weight for sparsity prior (if == 1, uniform prior is applied)",
)
parser.add_argument(
    "--edge_types",
    type=int,
    default=2,
    help="Number of different edge-types to model",
)

########### Different variants for variational distribution q ###############
parser.add_argument(
    "--dont_use_encoder",
    action="store_true",
    default=False,
    help="If true, replace encoder with distribution to be estimated",
)
parser.add_argument(
    "--lr_z",
    type=float,
    default=0.1,
    help="Learning rate for distribution estimation.",
)

### global latent temperature ###
parser.add_argument(
    "--global_temp",
    action="store_true",
    default=False,
    help="Should we model temperature confounding?",
)
parser.add_argument(
    "--load_temperatures",
    help="Should we load temperature data?",
    action="store_true",
)

parser.add_argument(
    "--num_cats",
    type=int,
    default=3,
    help="Number of categories in temperature distribution.",
)

### unobserved time-series ###
parser.add_argument(
    "--unobserved",
    type=int,
    default=0,
    help="Number of time-series to mask from input.",
)
parser.add_argument(
    "--model_unobserved",
    type=int,
    default=0,
    help="If 0, use NRI to infer unobserved particle. "
    "If 1, removes unobserved from data. "
    "If 2, fills empty slot with mean of observed time-series (mean imputation)",
)
parser.add_argument(
    "--dont_shuffle_unobserved",
    action="store_true",
    default=False,
    help="If true, always mask out last particle in trajectory. "
    "If false, mask random particle.",
)
parser.add_argument(
    "--teacher_forcing",
    type=int,
    default=0,
    help="Factor to determine how much true trajectory of "
    "unobserved particle should be used to learn prediction.",
)

############## loading and saving ##############

parser.add_argument(
    "--timesteps", type=int, default=49, help="Number of timesteps in input."
)
parser.add_argument(
    "--num_atoms", type=int, default=5, help="Number of time-series in input."
)
parser.add_argument(
    "--dims", type=int, default=4, help="Dimensionality of input."
)
parser.add_argument(
    "--datadir",
    type=str,
    default="./data",
    help="Name of directory where data is stored.",
)
parser.add_argument(
    "--save_folder",
    type=str,
    default="logs",
    help="Where to save the trained model, leave empty to not save anything.",
)
parser.add_argument(
    "--expername",
    type=str,
    default="",
    help="If given, creates a symlinked directory by this name in logdir"
    "linked to the results file in save_folder"
    "(be careful, this can overwrite previous results)",
)
parser.add_argument(
    "--sym_save_folder",
    type=str,
    default="../logs",
    help="Name of directory where symlinked named experiment is created."
)
parser.add_argument(
    "--load_folder",
    type=str,
    default="",
    help="Where to load pre-trained model if finetuning/evaluating. "
    + "Leave empty to train from scratch",
)

############## fine tuning ##############
parser.add_argument(
    "--test_time_adapt",
    action="store_true",
    default=False,
    help="Test time adapt q(z) on first half of test sequences.",
)
parser.add_argument(
    "--lr_logits",
    type=float,
    default=0.01,
    help="Learning rate for test-time adapting logits.",
)
parser.add_argument(
    "--num_tta_steps",
    type=int,
    default=100,
    help="Number of test-time-adaptation steps per batch.",
)

############## almost never change these ##############
parser.add_argument(
    "--dont_skip_first",
    action="store_true",
    default=False,
    help="If given as argument, do not skip first edge type in decoder, i.e. it represents no-edge.",
)
parser.add_argument(
    "skip_first",
    action="store_true",
    default=False,
    help="If given as argument, do not skip first edge type in decoder, i.e. it represents no-edge.",
)
parser.add_argument(
    "--temp", type=float, default=0.5, help="Temperature for Gumbel softmax."
)
parser.add_argument(
    "--hard",
    action="store_true",
    default=False,
    help="Uses discrete samples in training forward pass.",
)
parser.add_argument(
    "--no_validate", action="store_true", default=False, help="Do not validate results throughout training."
)
parser.add_argument(
    "--no_cuda", action="store_true", default=False, help="Disables CUDA training."
)
parser.add_argument("--var", type=float, default=5e-7, help="Output variance.")
parser.add_argument(
    "--encoder_dropout",
    type=float,
    default=0.0,
    help="Dropout rate (1 - keep probability).",
)
parser.add_argument(
    "--decoder_dropout",
    type=float,
    default=0.0,
    help="Dropout rate (1 - keep probability).",
)
parser.add_argument(
    "--no_factor",
    action="store_true",
    default=False,
    help="Disables factor graph model.",
)


args = parser.parse_args()


args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
args.validate = not args.no_validate
args.shuffle_unobserved = not args.dont_shuffle_unobserved
args.skip_first = not args.dont_skip_first
args.use_encoder = not args.dont_use_encoder
args.time = datetime.datetime.now().isoformat()

if args.device.type != "cpu":
    if args.GPU_to_use is not None:
        torch.cuda.set_device(args.GPU_to_use)
    torch.cuda.manual_seed(args.seed)
    args.num_GPU = 1  # torch.cuda.device_count()
    args.batch_size_multiGPU = args.batch_size * args.num_GPU
else:
    args.num_GPU = None
    args.batch_size_multiGPU = args.batch_size

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.set_precision(10)
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

tab_printer(args)


### Code adapted from https://github.com/rosinality/igebm-pytorch

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

        
def clip_grad(parameters, optimizer):
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]

                if 'step' not in state or state['step'] < 1:
                    continue

                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']

                bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))

                
  
        
def train(model, train_dataloader, n_atom, device):
    parameters = model.parameters()
    optimizer = Adam(parameters, lr=args.lr, betas=(0.0, 0.999), weight_decay=args.wd)
    for epoch in range(args.max_epochs):
        t_start = time.time()
        losses_reg = []
        losses_en = []
        losses = []
        losses_bce = []
        for i, batch in enumerate(tqdm(train_dataloader)):

            data, relations, temperatures = data_loader.unpack_batches(args, batch)
            
            #TODO: Relations to flat_adj
            # flat_adj = np.ones(args.num_atoms*args.num_atoms-args.num_atoms)
            # adj = torch.Tensor(flat_adj)
            # if args.cuda:
            #     adj = adj.cuda()

            pos_x = data #CFL [128, 5, 49, 4]
            pos_adj =  relations.unsqueeze(dim=1) #CFL [128, 20] #CFL Change it to (num_atoms*num_atoms)-num_atoms ?
            # logits = model(data, rel_rec, rel_send, adj)

            # ### Dequantization
            # pos_x = batch[0].to(device) 
            # pos_x += args.c * torch.rand_like(pos_x, device=device)  # (128, 9, 5)
            # pos_adj = batch[1].to(device) 
            # pos_adj += args.c * torch.rand_like(pos_adj, device=device)  # (128, 4, 9, 9)

            
            ### Langevin dynamics
            # neg_x = torch.rand(pos_x.shape[0], n_atom, args.timesteps, 4, device=device) * (1 + args.c) #CFL TODO: Change harcoded 4 (x,y,vx,vy)
            neg_adj = torch.rand(pos_adj.shape[0], 1, n_atom * n_atom - n_atom, device=device)  #CFL from an uniform distribution 
        
            #pos_adj = rescale_adj(pos_adj)
            # neg_x.requires_grad = True
            neg_adj.requires_grad = True
            
            
            
            requires_grad(parameters, False)
            model.eval()
            

            
            # noise_x = torch.randn(neg_x.shape[0], n_atom, args.timesteps, 4, device=device)  # (128, 9, 5)
            noise_adj = torch.randn(neg_adj.shape[0], 1, n_atom * n_atom - n_atom, device=device)  #(128, 4, 9, 9) 
            for k in range(args.sample_step):

                # noise_x.normal_(0, args.noise) #CFL fills noise_x wth normal distribution and std noise 
                noise_adj.normal_(0, args.noise)
                # neg_x.data.add_(noise_x.data)
                neg_adj.data.add_(noise_adj.data)

                neg_out = model(pos_x, rel_rec, rel_send, neg_adj)

                # neg_out = model(neg_adj, neg_x)
                neg_out.sum().backward() #CFL compute derivatives over a scalar (loss)
                if args.clamp:
                    # neg_x.grad.data.clamp_(-0.01, 0.01)
                    neg_adj.grad.data.clamp_(-0.01, 0.01)
        

                # neg_x.data.add_(neg_x.grad.data, alpha=-args.step_size)
                neg_adj.data.add_(neg_adj.grad.data, alpha=-args.step_size)

                # neg_x.grad.detach_()
                # neg_x.grad.zero_()
                neg_adj.grad.detach_()
                neg_adj.grad.zero_()
                
                # neg_x.data.clamp_(0, 1 + args.c)
                neg_adj.data.clamp_(0, 1)

            ### Training by backprop
            # neg_x = neg_x.detach()
            neg_adj = neg_adj.detach()
            requires_grad(parameters, True)
            model.train()
            
            model.zero_grad()
            
            
            pos_out = model(pos_x, rel_rec, rel_send, pos_adj)
            neg_out = model(pos_x, rel_rec, rel_send, neg_adj) #CFL Positive x
            
            loss_reg = (pos_out ** 2 + neg_out ** 2)  # energy magnitudes regularizer
            loss_en = pos_out - neg_out  # loss for shaping energy function

            bce_loss = BCELoss()
            loss_bce = bce_loss(neg_adj.float(), pos_adj.float() )
            # print('BCE loss', bce_loss(neg_adj.float(), pos_adj.float() ))
            losses_bce.append(loss_bce)

            loss = loss_en + args.alpha * loss_reg #+ loss_bce 
            loss = loss.mean()
            loss.backward()
            clip_grad(parameters, optimizer)
            optimizer.step()
            

            losses_reg.append(loss_reg.mean())
            losses_en.append(loss_en.mean())
            losses.append(loss)


            

        
        print(neg_adj)
        print(pos_adj)
            
            
        t_end = time.time()
        
        ### Save checkpoints
        if (epoch+1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'epoch_{}.pt'.format(epoch + 1)))
            print('Saving checkpoint at epoch ', epoch+1)
            print('==========================================')
        print('Epoch: {:03d}, Loss: {:.6f}, Energy Loss: {}, Regularizer Loss: {:.6f}, BCE Loss: {:.2f},  Sec/Epoch: {:.2f}'.format(epoch+1, (sum(losses)/len(losses)).item(), (sum(losses_en)/len(losses_en)).item(), (sum(losses_reg)/len(losses_reg)).item(), (sum(losses_bce)/len(losses_bce)).item(), t_end-t_start))
        print('==========================================')
        #tensorboard
        writer.add_scalar('train/loss', (sum(losses)/len(losses)).item(), epoch * len(train_dataloader) + i)
        writer.add_scalar('train/Energy loss', (sum(losses)/len(losses)).item(), epoch * len(train_dataloader) + i)
        writer.add_scalar('train/Regularizer loss', (sum(losses)/len(losses)).item(), epoch * len(train_dataloader) + i)
        writer.add_scalar('train/BCE loss', (sum(losses)/len(losses)).item(), epoch * len(train_dataloader) + i)





if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Load dataset
    t_start = time.time()

    if args.data_name=="qm9":

    #     data_file = "qm9_relgcn_kekulized_ggnp.npz"
    #     transform_fn = transform_qm9.transform_fn
    #     atomic_num_list = [6, 7, 8, 9, 0]
    #     file_path = '../datasets/valid_idx_qm9.json'
    #     valid_idx = transform_qm9.get_val_ids(file_path)
    #     n_atom_type = 5
    #     n_atom = 9
    #     n_edge_type = 4
    # elif args.data_name=="zinc250k":
    #     data_file = "zinc250k_relgcn_kekulized_ggnp.npz"
    #     transform_fn = transform_zinc250k.transform_fn
    #     atomic_num_list = transform_zinc250k.zinc250_atomic_num_list
    #     file_path = '../datasets/valid_idx_zinc250k.json'
    #     valid_idx = transform_zinc250k.get_val_ids(file_path)
    #     n_atom_type = len(atomic_num_list) #10
        n_atom = 5
    #     n_edge_type = 4
        print("no data")
    else:
        print("This dataset name is not supported!")


    # dataset = NumpyTupleDataset.load(os.path.join(args.data_dir, data_file), transform=transform_fn) # 133885
    # if len(valid_idx) > 0:
    #     train_idx = [t for t in range(len(dataset)) if t not in valid_idx]  # 120803 = 133885-13082
    #     train_set = Subset(dataset, train_idx)  # 120,803
    #     test_set = Subset(dataset, valid_idx)  # 13,082
    # else:
    #     torch.manual_seed(args.seed)
    #     train_set, test_set = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

    # train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    (
        train_loader,
        valid_loader,
        test_loader,
        loc_max,
        loc_min,
        vel_max,
        vel_min,
    ) = data_loader.load_data(args)



    t_end = time.time()

    print('==========================================')
    print('Load data done! Time {:.2f} seconds'.format(t_end - t_start))
    print('==========================================')

    #adj = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms) #CFL Create all connected adjacency matrix to test edge2node_adj TODO: 1. Change it for GT , 2. Random initialization
    flat_adj = np.ones(args.num_atoms*args.num_atoms-args.num_atoms)
    adj = torch.Tensor(flat_adj)
    if args.cuda:
        adj = adj.cuda()

    rel_rec, rel_send = utils.create_rel_rec_send(args, args.num_atoms)
    #CFL Encoder MLP by default
    model, _, optimizer, scheduler, edge_probs = model_loader.load_model(
        args, loc_max, loc_min, vel_max, vel_min
    )

    ### Initialize model
    # model = GraphEBM(n_atom_type, args.hidden, n_edge_type, args.swish, args.depth, add_self=args.add_self, dropout = args.dropout)
    print(model)
    print('==========================================')
    model = model.to(device)
    description = 'only_energy'
    writer = SummaryWriter('runs/{}'.format(description))
    # writer.add_graph(model)
    

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    ### Train
    train(model, train_loader, n_atom, device)
    print('FI')
    writer.close()
    


            

    