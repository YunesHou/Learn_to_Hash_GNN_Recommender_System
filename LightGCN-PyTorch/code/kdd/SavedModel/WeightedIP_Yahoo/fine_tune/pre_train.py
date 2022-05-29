# -*- coding: utf-8 -*-
"""
Created on Wed May 19 12:31:47 2021

@author: Zongyue
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from config import CFG
from tqdm import tqdm
import os, shutil
from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler

from utils import init_logger, VectorDataset, dot, write_down
from utils import BinaryRegularization
from loss import MSELoss
from hash import MLPHash, MLPFunc
from decoders import WeightedInnerProductDecoder, LHTIPSDecoder

os.environ["CUDA_VISIBLE_DEVICES"]="3"

if not os.path.exists(CFG.save_dir):
    os.makedirs(CFG.save_dir)
shutil.copy('config.py', CFG.save_dir)
shutil.copy('pre_train.py', CFG.save_dir)
LOGGER = init_logger(CFG.save_dir+'pre_train.log')


def train(data_hash, query_func, decoder,
           dataset, queryset, dataset_val, queryset_val, device = 'cpu'):
    data_hash = data_hash.to(device)
    query_func = query_func.to(device)
    decoder = decoder.to(device)

    dataloader = DataLoader(dataset, batch_size = CFG.batch_size, shuffle=True)
    queryloader = DataLoader(queryset, batch_size = CFG.batch_size, shuffle=True)

    data_val = DataLoader(dataset_val, batch_size = CFG.batch_size, shuffle=False)
    query_val = DataLoader(queryset_val, batch_size = CFG.batch_size, shuffle=False)
       
    criterion = nn.MSELoss()
    optimizer1 = torch.optim.Adam(data_hash.parameters(), lr = CFG.lr)
    optimizer2 = torch.optim.Adam(query_func.parameters(), lr = CFG.lr)
    optimizer3 = torch.optim.Adam(decoder.parameters(), lr = CFG.lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1, verbose=True) 
    best_valid_loss = 1000
    stagnant_epoch = 0

    for epoch in range(CFG.epochs):
        mse_loss_list = []
        reg_loss_list = []
        norm_loss_list = []
        loss_list = []
        data_hash.train()
        query_func.train()
        decoder.train()

        for i, O in enumerate(tqdm(dataloader)):
            Q = next(iter(queryloader))
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            if i == CFG.max_iter_per_epoch:
                break

            O = O.to(device)
            Q = Q.to(device)

            label = dot(O,Q) - CFG.pred_bias
            o_code = data_hash(O)
            q_code = query_func(Q)
            o_bcode = data_hash.binarize(o_code)
            if CFG.binarize_query == True:
                q_bcode = query_func.binarize(q_code)
            else:
                q_bcode = q_code

            true_o_pred = decoder(o_bcode, q_code)
            true_q_pred = decoder(o_code, q_bcode)
            true_pred = decoder(o_bcode, q_bcode)

            
            if i % 3 == 0: # optimze data_hash 
                mse_loss = criterion(true_q_pred, label)
                reg_loss = BinaryRegularization(o_code)
                loss = mse_loss + CFG.reg_w * reg_loss
                loss.backward()
                optimizer1.step()
                data_hash.force_limit()
            elif i % 3 == 1: # optimize query_func
                mse_loss = criterion(true_o_pred, label)
                if CFG.binarize_query == True:
                    reg_loss = BinaryRegularization(q_code)
                else:
                    reg_loss = 0 * mse_loss
                loss = mse_loss + CFG.reg_w * reg_loss
                loss.backward()
                optimizer2.step()
                query_func.force_limit()
            else: # optimize decoder
                loss = criterion(true_pred, label)
                loss.backward()
                optimizer3.step()
                decoder.force_limit()

            loss_list.append(loss.item()) 
            true_mse_loss = criterion(true_pred, label)
            mse_loss_list.append(true_mse_loss.item())
            reg_loss_list.append(reg_loss.item())


           
        if epoch % CFG.epoch_to_valid != 0:
            LOGGER.info('Epoch %d, MSE Loss: %.3f, Reg Loss: %.3f'%(epoch, 
                    np.mean(mse_loss_list), np.mean(reg_loss_list)))
            continue

        data_hash.eval()
        query_func.eval()
        decoder.eval()
        mse = []
        
        with torch.no_grad():
            for O in tqdm(data_val):
                O = O.to(device)
                o_code = data_hash.binarize(data_hash(O))
                for Q in query_val:
                    Q = Q.to(device)
                    q_code = query_func(Q)
                    if CFG.binarize_query == True:
                        q_code = query_func.binarize(q_code)
                    label = dot(O,Q)
                    label = label - CFG.pred_bias
                    pred = decoder(o_code, q_code)
            
                    mse_loss = criterion(pred, label)
                    mse.append(mse_loss.item())
            print(label)
            print(pred)
        valid_loss = np.mean(mse)**0.5
        LOGGER.info('Epoch %d, Training Loss: %.3f, MSE Loss: %.3f, Reg Loss: %.3f,\
                     Valid RMSE: %.3f'%(epoch, 
                    np.mean(loss_list), np.mean(mse_loss_list), np.mean(reg_loss_list), 
                    valid_loss))
        if valid_loss < best_valid_loss and epoch >= CFG.min_epoch:
            best_valid_loss = valid_loss
            LOGGER.info('Save Model')
            state = {'d_h': data_hash.state_dict(), 
                     'optimizer1': optimizer1.state_dict(),
                     'q_f': query_func.state_dict(),
                     'optimizer2': optimizer2.state_dict(),
                     'decoder': decoder.state_dict(),
                     'optimizer3': optimizer3.state_dict()}
            torch.save(state, CFG.save_dir + 'functions_'+CFG.model_name)
            stagnant_epoch = 0
        else:
            stagnant_epoch += 1
            if stagnant_epoch > CFG.max_stagnant_epoch:
                break

    data_hash.load_state_dict(state['d_h'])
    query_func.load_state_dict(state['q_f'])
    decoder.load_state_dict(state['decoder'])
    print('Finish Training')
    #return data_hash, query_func, decoder 

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if CFG.loader == 'torch':
        vectors = torch.load(CFG.data_file)
    elif CFG.loader == 'pickle':
        vectors = pickle.load(open(CFG.data_file, 'rb'))
    else:
        raise RuntimeError('Unrecognized Loader: '+CFG.loader)

    #assert len(CFG.hidden_dims) == 0
    # Q is query vectors, D is data vectors
    if CFG.use_item == False:
        D = vectors['W']
        Q = vectors['H']
    else:
        D = vectors['H']
        Q = vectors['W']
    #W = W[:10000]
    #H = H[:10000]
    print('D shape')
    print(D.shape)
    print('Q shape')
    print(Q.shape)

    #np.random.shuffle(W)
    #np.random.shuffle(H)

    D_test = D[:CFG.test_num]
    Q_test = Q[:CFG.test_num]

    train_num_D = D.shape[0]-CFG.test_num-CFG.valid_num #2
    train_num_Q = Q.shape[0]-CFG.test_num-CFG.valid_num #2

    D_train = D[CFG.test_num:CFG.test_num+train_num_D] 
    Q_train = Q[CFG.test_num:CFG.test_num+train_num_Q] 
    if D_train.shape[0] > 1000000:
        D_train = D_train[:1000000]
    if Q_train.shape[0] > 1000000:
	    Q_train = Q_train[:1000000]

    D_dataset_train = VectorDataset(D_train, lambda x: torch.FloatTensor(x))
    Q_dataset_train = VectorDataset(Q_train, lambda x: torch.FloatTensor(x))

    print(len(D_dataset_train))
    print(len(Q_dataset_train))

    D_valid = D[CFG.test_num+train_num_D:] 
    Q_valid = Q[CFG.test_num+train_num_Q:] 
    D_dataset_valid = VectorDataset(D_valid, lambda x: torch.FloatTensor(x))
    Q_dataset_valid = VectorDataset(Q_valid, lambda x: torch.FloatTensor(x))


    data_hash = MLPHash(CFG.emb_len, CFG.hidden_dims, CFG.code_len, use_bn=CFG.use_bn)
    if CFG.binarize_query:
        query_func = MLPHash(CFG.emb_len, CFG.hidden_dims, CFG.code_len, use_bn=CFG.use_bn)
    else:
        query_func = MLPFunc(CFG.emb_len, CFG.hidden_dims, CFG.code_len, use_bn=CFG.use_bn)
    if CFG.decoder == 'WeightedIP':
        decoder = WeightedInnerProductDecoder(CFG.code_len)
    elif CFG.decoder == 'LH-TIPS':
        decoder = LHTIPSDecoder(CFG.code_len)
    else:
        raise NotImplementedError
       

    start_time = time.time()
    train(data_hash, query_func, decoder, D_dataset_train, Q_dataset_train, 
          D_dataset_valid, Q_dataset_valid, device = device)
    time_cost = time.time() - start_time
    LOGGER.info('training time cost: %f s'%(time_cost))

    target_dir = os.path.join(CFG.save_dir, 'data_hash')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    write_down(target_dir, data_hash.state_dict())    

    target_dir = os.path.join(CFG.save_dir, 'query_func')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    write_down(target_dir, query_func.state_dict())    

    target_dir = os.path.join(CFG.save_dir, 'decoder')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    write_down(target_dir, decoder.state_dict())    



if __name__ == '__main__':
    main()
