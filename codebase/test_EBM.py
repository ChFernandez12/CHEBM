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

def test(model, test_dataloader, n_atom, device):
       ######## TEST ########

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
        
        for i, batch in enumerate(tqdm(test_dataloader)):

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


            loss = loss_mse  + loss_en + args.alpha * loss_reg 
                    
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
        print('Epoch: {:03d}, Loss: {}, Energy Loss: {}, Regularizer Loss: {:.6f}, MSE Loss: {},  Sec/Epoch: {:.2f}'.format(epoch+1, (sum(losses_val)/len(losses_val)).item(), (sum(losses_val_en)/len(losses_val_en)).item(), (sum(losses_val_reg)/len(losses_val_reg)).item(), (sum(losses_val_mse)/len(losses_val_mse)).item(), t_end-t_start))
        print('Epoch: {:03d}, ACC: {}, AUROC: {}'.format(epoch+1, (sum(accuracies_val)/len(accuracies_val)).item(), (sum(auroc_val)/len(auroc_val)).item()))
        





if __name__ == '__main__':
    epoch = 2344
    args = arg_parser.parse_args()
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
    # flat_adj = np.ones(args.num_atoms*args.num_atoms-args.num_atoms)
    # adj = torch.Tensor(flat_adj)
    # if args.cuda:
    #     adj = adj.cuda()

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
    print('Load checkpoint: ' + os.path.join(args.save_dir + '/' + args.expername , 'epoch_{}.pt'.format(epoch)))

    model.load_state_dict(torch.load( os.path.join(args.save_dir + '/' + args.expername , 'epoch_{}.pt'.format(epoch))))
    
    # description = 'energy_bce_1e3'
    # dt_string = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    # writer = SummaryWriter('runs/{}/{}'.format(args.expername,dt_string))
    # writer.add_graph(model)
    

    # if not os.path.exists(args.save_dir + '/' + args.expername):
    #     os.makedirs(args.save_dir+ '/' + args.expername)
    
    ### Train
    test(model, test_loader, n_atom, device)
    print('FI')
    # writer.close()
    