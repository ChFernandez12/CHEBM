import argparse
import time
import sys
import os
from tqdm import tqdm
import random

import torch
from torch.optim import Adam
from texttable import Texttable
from torch.optim import lr_scheduler
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
# from tensorboard.plugins.hparams import api as hp #TODO

### Args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed to use')
parser.add_argument('--data_name', type=str, default='qm9', choices=['qm9', 'zinc250k'], help='Dataset name')
parser.add_argument('--hidden', type=int, default=64, help='Hidden dimension')
parser.add_argument('--depth', type=int, default=2, help='Number of graph conv layers')
parser.add_argument('--add_self', type=strtobool, default='false', help='Add shortcut in graphconv')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')
parser.add_argument('--swish', type=strtobool, default='true', help='Use swish as activation function')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size during training')
parser.add_argument('--shuffle', type=strtobool, default='true', help='Shuffle the data batch')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers in the data loader')
parser.add_argument('--c', type=float, default=0.5, help='Dequantization using uniform distribution of [0,c)')
parser.add_argument('--alpha', type=float, default=1.0, help='Weight for energy magnitudes regularizer')
parser.add_argument('--step_size', type=int, default=10, help='Step size in Langevin dynamics')
parser.add_argument('--sample_step', type=int, default=3, help='Number of sample step in Langevin dynamics')
parser.add_argument('--valid_sample_step', type=int, default=3, help='Number of sample step in Langevin dynamics')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate') #TODO descending learning rate
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
    "--GPU_to_use", type=int, default=1, help="GPU to use for training"
)

############## training hyperparameter ##############
parser.add_argument(
    "--epochs", type=int, default=500, help="Number of epochs to train."
)

parser.add_argument(
    "--lr_decay",
    type=int,
    default=25,
    help="After how epochs to decay LR by a factor of gamma.",
)
parser.add_argument("--gamma", type=float, default=0.01, help="LR decay factor.")
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


args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

        
def train(model, train_dataloader, valid_dataloader, n_atom, device):
    parameters = model.parameters()
    optimizer = Adam(parameters, lr=args.lr, betas=(0.0, 0.999), weight_decay=args.wd)

    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay, gamma=args.gamma
    )
    #TODO: call train for each epoch
    for epoch in range(args.max_epochs):
        t_start = time.time()
        losses_reg = []
        losses_bce = []
        losses_en = []
        losses = []
        losses_mse = []
        accuracies = []
        for i, batch in enumerate(tqdm(train_dataloader)):

            data, relations, temperatures = data_loader.unpack_batches(args, batch)
            
            #TODO: Relations to flat_adj
            # flat_adj = np.ones(args.num_atoms*args.num_atoms-args.num_atoms)
            # adj = torch.Tensor(flat_adj)
            # if args.cuda:
            #     adj = adj.cuda()

            pos_x = data #CFL [128, 5, 49, 4]
            pos_adj =  relations.unsqueeze(dim=1).float() #CFL [128, 20] #CFL Change it to (num_atoms*num_atoms)-num_atoms ?
            # logits = model(data, rel_rec, rel_send, adj)

            # ### Dequantization
            # pos_x = batch[0].to(device) 
            # pos_x += args.c * torch.rand_like(pos_x, device=device)  # (128, 9, 5)
            # pos_adj = batch[1].to(device) 
            # pos_adj += args.c * torch.rand_like(pos_adj, device=device)  # (128, 4, 9, 9)

            
            ### Langevin dynamics
            # neg_x = torch.rand(pos_x.shape[0], n_atom, args.timesteps, 4, device=device) * (1 + args.c) #CFL TODO: Change harcoded 4 (x,y,vx,vy)
            neg_adj = torch.rand(pos_adj.shape[0], 1, n_atom * n_atom - n_atom, device=device)  #CFL from an uniform distribution 
            # neg_adj = torch.bernoulli(neg_adj)
            #pos_adj = rescale_adj(pos_adj)
            # neg_x.requires_grad = True
            neg_adj.requires_grad = True
            
            
            requires_grad(parameters, False)
            model.eval()
            

            
            # noise_x = torch.randn(neg_x.shape[0], n_atom, args.timesteps, 4, device=device)  # (128, 9, 5)
            #uniforme between 0 and 1
            #TODO binaritzar adj edges
            noise_adj = torch.randn(neg_adj.shape[0], 1, n_atom * n_atom - n_atom, device=device)  #(128, 4, 9, 9) 
            neg_adjs = []
            for k in range(args.sample_step):

                neg_out = model(pos_x, rel_rec, rel_send, neg_adj)

                #neg_out.sum().backward() #CFL compute derivatives over a scalar (loss)
                #Compute gradient neg_adj
                adj_grad, = torch.autograd.grad([neg_out.sum()],[neg_adj], create_graph=True)

                #Step
                neg_adj = neg_adj - args.step_size * adj_grad
                
                #TODO: Change temperature
                # neg_adj = torch.clamp(neg_adj, 0, 1)
                neg_adj= torch.sigmoid( neg_adj * 10)

                #Return only last step to backpropagate
                if k >= args.sample_step - 1:
                    neg_adjs.append(neg_adj)
                
                #Delete gradients
                neg_adj = neg_adj.detach()
                neg_adj.requires_grad = True
            

            model.train()
            model.zero_grad()
            #Energy of positive and negative adjacency
            #TODO: Try with binary negative adjacency instead of uniform noise
            pos_out = model(pos_x, rel_rec, rel_send, pos_adj)
            neg_out = model(pos_x, rel_rec, rel_send, neg_adj) #CFL Positive x

            #Losses
            loss_reg = (pos_out ** 2 + neg_out ** 2).mean()  # energy magnitudes regularizer
            loss_en = pos_out.mean() - neg_out.mean()  # loss for shaping energy function

            bce_loss = BCELoss()
            loss_bce = bce_loss(neg_adjs[-1].float(), pos_adj.float() ).mean()
            loss_mse = torch.pow(neg_adjs[-1]- pos_adj, 2).mean()


            loss = loss_bce + loss_en + args.alpha * loss_reg #
            # loss = loss.mean()
            loss.backward()

            
            clip_grad(parameters, optimizer)
            optimizer.step()
        
            losses_bce.append(loss_bce.detach())
            losses_mse.append(loss_mse.detach())
            losses_reg.append(loss_reg.detach())
            losses_en.append(loss_en.detach())
            losses.append(loss)

            #Metrics
            train_acc = torch.sum(((neg_adj > 0.5)*torch.ones(neg_adj.shape).to(device) == pos_adj))/ (neg_adj.shape[0]*neg_adj.shape[1]*neg_adj.shape[2])
            accuracies.append(train_acc)

        scheduler.step()
    
        ######## VALIDATION ########

        t_start = time.time()

        losses_val_reg = []
        losses_val_bce = []
        losses_val_en = []
        losses_val = []
        losses_val_mse = []
        accuracies_val = []

        requires_grad(parameters, False)
        model.eval()
        
        for i, batch in enumerate(tqdm(valid_dataloader)):

            data, relations, temperatures = data_loader.unpack_batches(args, batch)

            pos_x = data #CFL [128, 5, 49, 4]
            pos_adj =  relations.unsqueeze(dim=1).float() #CFL [128, 20] #CFL Change it to (num_atoms*num_atoms)-num_atoms ?
            neg_adj = torch.rand(pos_adj.shape[0], 1, n_atom * n_atom - n_atom, device=device)  #CFL from an uniform distribution 

            neg_adj.requires_grad = True

            #uniforme between 0 and 1
            #TODO binaritzar adj edges
            noise_adj = torch.randn(neg_adj.shape[0], 1, n_atom * n_atom - n_atom, device=device)  #(128, 4, 9, 9) 
            neg_adjs = []
            for k in range(args.valid_sample_step):

                neg_out = model(pos_x, rel_rec, rel_send, neg_adj)

                #Compute gradient neg_adj
                adj_grad, = torch.autograd.grad([neg_out.sum()],[neg_adj], create_graph=True)

                #Step
                neg_adj = neg_adj - args.step_size * adj_grad
                
                #TODO: Change temperature
                # neg_adj = torch.clamp(neg_adj, 0, 1)
                neg_adj= torch.sigmoid( neg_adj * 10)
                
                #Delete gradients
                neg_adj = neg_adj.detach()
                neg_adj.requires_grad = True

            neg_adj = neg_adj.detach()
            
            #Energy of positive and negative adjacency
            #TODO: Try with binary negative adjacency instead of uniform noise
            pos_out = model(pos_x, rel_rec, rel_send, pos_adj)
            neg_out = model(pos_x, rel_rec, rel_send, neg_adj) #CFL Positive x

            #Losses
            loss_reg = (pos_out ** 2 + neg_out ** 2).mean()  # energy magnitudes regularizer
            loss_en = pos_out.mean() - neg_out.mean()  # loss for shaping energy function

            bce_loss = BCELoss()
            loss_bce = bce_loss(neg_adj.float(), pos_adj.float() ).mean()
            loss_mse = torch.pow(neg_adj- pos_adj, 2).mean()


            loss = loss_bce  + loss_en + args.alpha * loss_reg 
                    
            losses_val_bce.append(loss_bce.detach())
            losses_val_mse.append(loss_mse.detach())
            losses_val_reg.append(loss_reg.detach())
            losses_val_en.append(loss_en.detach())
            losses_val.append(loss)

            #Metrics
            train_acc = torch.sum(((neg_adj > 0.5)*torch.ones(neg_adj.shape).to(device) == pos_adj))/ (neg_adj.shape[0]*neg_adj.shape[1]*neg_adj.shape[2])
            accuracies_val.append(train_acc)

        print('--------------')
        print(pos_out)
        print(neg_out)
        print('--------------')
        print(neg_adj)
        print(pos_adj)
        
        t_end = time.time()
        
        ### Save checkpoints
        if (epoch+1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'epoch_{}.pt'.format(epoch + 1)))
            print('Saving checkpoint at epoch ', epoch+1)
            print('==========================================')
            
        print('Epoch: {:03d}, Loss: {:.6f}, Energy Loss: {}, Regularizer Loss: {:.6f}, MSE Loss: {:.2f},  Sec/Epoch: {:.2f}'.format(epoch+1, (sum(losses)/len(losses)).item(), (sum(losses_en)/len(losses_en)).item(), (sum(losses_reg)/len(losses_reg)).item(), (sum(losses_mse)/len(losses_mse)).item(), t_end-t_start))
        print('==========================================')
        #tensorboard
        writer.add_scalar('train/loss', (sum(losses)/len(losses)).item(), epoch * len(train_dataloader) + i)
        writer.add_scalar('train/Energy loss', (sum(losses_en)/len(losses_en)).item(), epoch * len(train_dataloader) + i)
        writer.add_scalar('train/Regularizer loss', (sum(losses_reg)/len(losses_reg)).item(), epoch * len(train_dataloader) + i)
        writer.add_scalar('train/MSE loss', (sum(losses_mse)/len(losses_mse)).item(), epoch * len(train_dataloader) + i)
        writer.add_scalar('train/accuracy', (sum(accuracies)/len(accuracies)).item(), epoch * len(train_dataloader) + i)
        writer.add_scalar('train/BCE loss', (sum(losses_bce)/len(losses_bce)).item(), epoch * len(train_dataloader) + i)
        
        writer.add_scalar('val/loss', (sum(losses_val)/len(losses_val)).item(), epoch * len(valid_dataloader) + i)
        writer.add_scalar('val/Energy loss', (sum(losses_val_en)/len(losses_val_en)).item(), epoch * len(valid_dataloader) + i)
        writer.add_scalar('val/Regularizer loss', (sum(losses_val_reg)/len(losses_val_reg)).item(), epoch * len(valid_dataloader) + i)
        writer.add_scalar('val/MSE loss', (sum(losses_val_mse)/len(losses_val_mse)).item(), epoch * len(valid_dataloader) + i)
        writer.add_scalar('val/accuracy', (sum(accuracies_val)/len(accuracies_val)).item(), epoch * len(valid_dataloader) + i)
        writer.add_scalar('val/BCE loss', (sum(losses_val_bce)/len(losses_val_bce)).item(), epoch * len(valid_dataloader) + i)
        





if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Load dataset
    t_start = time.time()

    if args.data_name=="qm9":

        n_atom = 5
        print("no data")
    else:
        print("This dataset name is not supported!")


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
    # model, _, _, _, _ = model_loader.load_model(
    #     args, loc_max, loc_min, vel_max, vel_min
    # )
    model = model_loader.load_encoder(args)



    ### Initialize model
    # model = GraphEBM(n_atom_type, args.hidden, n_edge_type, args.swish, args.depth, add_self=args.add_self, dropout = args.dropout)
    print(model)
    print('==========================================')
    model = model.to(device)
    description = 'energy_bce_100'
    dt_string = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    writer = SummaryWriter('runs/{}/{}'.format(description,dt_string))
    # writer.add_graph(model)
    

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    ### Train
    train(model, train_loader, valid_loader, n_atom, device)
    print('FI')
    writer.close()
    


            

    