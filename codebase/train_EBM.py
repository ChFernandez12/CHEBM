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
# import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
# from tensorboard.plugins.hparams import api as hp #TODO
from matplotlib.patches import ConnectionPatch
### Args

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.set_precision(10)
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())



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


def get_trajectory_figure(data):
    fig = plt.figure()
    axes = plt.gca()
    lims = None
    if lims is not None:
        axes.set_xlim([lims[0], lims[1]])
        axes.set_ylim([lims[0], lims[1]])

    # state = state[b_idx].permute(1, 2, 0).cpu().detach().numpy()
    data = data.cpu().detach().numpy()
    print(data.shape)
    loc, vel = data[:, :,:2], data[:, :,2:] #(5,49,2)
    vel_norm = np.sqrt((vel ** 2).sum(axis=1))
    
    for i in range(loc.shape[0]):
        plt.plot(loc[i, :, 0], loc[i, :, 1], label = str(i))
        plt.plot(loc[i, -1, 0], loc[i, -1, 1], 'd')
    plt.legend()
    return plt, fig

def get_graph_figure(data, relations, obj):
    data = data.cpu().detach().numpy()
    loc, vel = data[:, :,:2], data[:, :,2:]

    fig = plt.figure()
    axes = plt.gca()
    # G = nx.DiGraph()
    relations = relations.cpu().detach().numpy()
    idx1 = 0
    idx2 = 0

    for i in range(obj): 
        for j in range( obj): 
            if idx1 % (obj + 1) == 0:
                idx1 += 1
                continue
            if relations[idx2] > 0.5: 
                # plt.arrow(loc[i, -1, 0], loc[i, -1, 1], loc[j, -1, 0] - loc[i, -1, 0], loc[j, -1, 1] - loc[i, -1, 1], width = 0.00001, head_width = 0.005, linestyle = ':')
                xyA = (loc[i, -1, 0], loc[i, -1, 1])
                xyB = (loc[j, -1, 0], loc[j, -1, 1])
                coordsA = "data"
                coordsB = "data"
                con = ConnectionPatch(xyA, xyB, coordsA, coordsB,
                                    arrowstyle="-|>", shrinkA=5, shrinkB=5,
                                    mutation_scale=20, fc="w")
                axes.add_artist(con)
                # G.add_edge(i,j)
            idx2 += 1
            idx1 += 1
    # axes.set_ylim([min(loc[:, -1, 1]), max(loc[:, -1, 1])])
    # axes.set_xlim([min(loc[:, -1, 1]), max(loc[:, -1, 1]) ])



    for i in range(loc.shape[0]):
        plt.plot(loc[i, -1, 0], loc[i, -1, 1], label = str(i))
    # nx.draw_spring(G, with_labels = True)


   

    return plt, fig



def train(model, train_dataloader, valid_dataloader, n_atom, device):
    parameters = model.parameters()
    optimizer = Adam(parameters, lr=args.lr, betas=(0.0, 0.999), weight_decay=args.wd)

    # if args.load_folder:
    #     epoch = 2999
    #     model.load_state_dict(torch.load( os.path.join(args.save_dir + '/bce_requirepar_negF' , 'epoch_{}.pt'.format(epoch))))

    # scheduler = lr_scheduler.StepLR(
    #     optimizer, step_size=args.lr_decay, gamma=args.gamma
    # )
    #TODO: call train for each epoch
    for epoch in range(args.max_epochs):
        t_start = time.time()
        losses_reg = []
        losses_bce = []
        losses_en = []
        losses = []
        losses_mse = []
        accuracies = []
        auroc = []

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
            # neg_adj.requires_grad = True
            
            
            requires_grad(parameters, False) # only interested in the gradients of the input.
            model.eval()
            

            
            # noise_x = torch.randn(neg_x.shape[0], n_atom, args.timesteps, 4, device=device)  # (128, 9, 5)
            #uniforme between 0 and 1
            #TODO binaritzar adj edges
            noise_adj = torch.randn(neg_adj.shape[0], 1, n_atom * n_atom - n_atom, device=device)  #(128, 4, 9, 9) 
            neg_adjs = []
            for k in range(args.sample_step):
                neg_adj.requires_grad = True
                neg_out = model(pos_x, rel_rec, rel_send, neg_adj)

                #neg_out.sum().backward() #CFL compute derivatives over a scalar (loss)
                #Compute gradient neg_adj
                adj_grad, = torch.autograd.grad([neg_out.sum()],[neg_adj], create_graph=True) # Computes and returns the sum of gradients of outputs with respect to the inputs.
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
            

            model.train()
            model.zero_grad()
            requires_grad(parameters, True)
            #Energy of positive and negative adjacency
            #TODO: Try with binary negative adjacency instead of uniform noise
            pos_out = model(pos_x, rel_rec, rel_send, pos_adj)
            neg_out = model(pos_x, rel_rec, rel_send, neg_adj) #CFL Positive x

            #Losses
            loss_reg = (pos_out ** 2 + neg_out ** 2).mean()  # energy magnitudes regularizer
            loss_en = pos_out.mean() - neg_out.mean()  # loss for shaping energy function

            bce_loss = BCELoss()
            loss_bce = bce_loss(neg_adjs[-1].float(), pos_adj.float() ).mean()
            loss_mse = torch.pow(neg_adjs[-1] - pos_adj, 2).mean()


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
            auroc.append(utils.calc_auroc(neg_adj,pos_adj ))


        # scheduler.step()  
                  
        ######## VALIDATION ########

        t_start = time.time()

        losses_val_reg = []
        losses_val_bce = []
        losses_val_en = []
        losses_val = []
        losses_val_mse = []
        accuracies_val = []
        auroc_val = []


        #requires_grad(parameters, False)
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
            model.eval()
            pos_out = model(pos_x, rel_rec, rel_send, pos_adj).detach()
            neg_out = model(pos_x, rel_rec, rel_send, neg_adj).detach() #CFL Positive x

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
            auroc_val.append(utils.calc_auroc(neg_adj,pos_adj ))

        print('--------------')
        print(pos_out)
        print(neg_out)
        print('--------------')
        print(neg_adj)
        print(pos_adj)
        
        t_end = time.time()
        
        ### Save checkpoints
        if (epoch+1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir + '/' + args.expername , 'epoch_{}.pt'.format(epoch + 1)))
            print('Saving checkpoint at epoch ', epoch+1)
            print('==========================================')
            
        print('Epoch: {:03d}, Loss: {}, Energy Loss: {}, Regularizer Loss: {:.6f}, MSE Loss: {:.2f},  Sec/Epoch: {:.2f}'.format(epoch+1, (sum(losses)/len(losses)).item(), (sum(losses_en)/len(losses_en)).item(), (sum(losses_reg)/len(losses_reg)).item(), (sum(losses_mse)/len(losses_mse)).item(), t_end-t_start))
        print('==========================================')
        #tensorboard
        writer.add_scalar('train/loss', (sum(losses)/len(losses)).item(), epoch * len(train_dataloader))
        writer.add_scalar('train/Energy loss', (sum(losses_en)/len(losses_en)).item(), epoch * len(train_dataloader))
        writer.add_scalar('train/Regularizer loss', (sum(losses_reg)/len(losses_reg)).item(), epoch * len(train_dataloader))
        writer.add_scalar('train/MSE loss', (sum(losses_mse)/len(losses_mse)).item(), epoch * len(train_dataloader))
        writer.add_scalar('train/accuracy', (sum(accuracies)/len(accuracies)).item(), epoch * len(train_dataloader))
        writer.add_scalar('train/BCE loss', (sum(losses_bce)/len(losses_bce)).item(), epoch * len(train_dataloader))
        writer.add_scalar('train/AUROC', (sum(auroc)/len(auroc)).item(), epoch * len(train_dataloader))
        
        writer.add_scalar('val/loss', (sum(losses_val)/len(losses_val)).item(), epoch * len(valid_dataloader))
        writer.add_scalar('val/Energy loss', (sum(losses_val_en)/len(losses_val_en)).item(), epoch * len(valid_dataloader))
        writer.add_scalar('val/Regularizer loss', (sum(losses_val_reg)/len(losses_val_reg)).item(), epoch * len(valid_dataloader))
        writer.add_scalar('val/MSE loss', (sum(losses_val_mse)/len(losses_val_mse)).item(), epoch * len(valid_dataloader) )
        writer.add_scalar('val/accuracy', (sum(accuracies_val)/len(accuracies_val)).item(), epoch * len(valid_dataloader) )
        writer.add_scalar('val/BCE loss', (sum(losses_val_bce)/len(losses_val_bce)).item(), epoch * len(valid_dataloader) )
        writer.add_scalar('val/AUROC', (sum(auroc_val)/len(auroc_val)).item(), epoch * len(valid_dataloader) )


        # FIGURES 
        rnd_ex = random.randint(0,200)
        writer.add_figure('GT', get_trajectory_figure(data[rnd_ex])[1], epoch * len(valid_dataloader))
        writer.add_figure('GRAPH_gt', get_graph_figure(data[rnd_ex], relations[rnd_ex], data.shape[1] )[1], epoch * len(valid_dataloader))
        writer.add_figure('GRAPH_pred', get_graph_figure(data[rnd_ex], neg_adj[rnd_ex,0,:], data.shape[1] )[1], epoch * len(valid_dataloader))




if __name__ == '__main__':
    args = arg_parser.parse_args()
    tab_printer(args)
    logs = logger.Logger(args)

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
    
    description = 'energy_bce_1e3'
    dt_string = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    writer = SummaryWriter('runs/{}/{}'.format(args.expername,dt_string))
    # writer.add_graph(model)
    

    if not os.path.exists(args.save_dir + '/' + args.expername):
        os.makedirs(args.save_dir+ '/' + args.expername)
    
    ### Train
    train(model, train_loader, valid_loader, n_atom, device)
    print('FI')
    writer.close()
    


            

    