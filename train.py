import os
import sys
import argparse
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from helper import read_data, read_yaml, dump_yaml, dump_json
from models import build_model

from tokenizer import get_build_tokenizer
from dataset import SentimentDataset, collate_fn
from Trainer import Trainer
from utils.evaluation import predictions, accuracy, report, conf_matrix

def main():   
    
    # load config file
    config_file_path = "/home/shashank/repos/varformers/config.yml"
    config = read_yaml(config_file_path) 
    device = config['device']
    filepath = config['filepath']
    exp_name = config['exp_name']
    exp_run = config['exp_run']

    # create experiment directory
    filepath = f'{filepath}/{exp_name}_{exp_run}'
    os.makedirs(f'{filepath}/best_model', exist_ok=True)

    # Data
    data_config = config['data']
    target = data_config['target']
    train_datapath = data_config['train_datapath']
    val_datapath = data_config['val_datapath']
    test_datapath = data_config['test_datapath']

    # Training
    train_config = config['training']
    optimizer = train_config['opt']
    lossfunc = train_config['loss']
    batch_size = train_config['batch_size']
    lr = train_config['lr']
    l2 = float(train_config['l2'])
    epochs = train_config['epochs']
    callbacks = train_config['callbacks']

    # Model params
    resume_from_checkpoint = config['model']['resume_from_checkpoint']
    if resume_from_checkpoint:
        model_checkpoint = config['model']['start_checkpoint']
        model_ = read_yaml(f'{model_checkpoint}/config.yml')
        config['model']['type'] = model_['model']['type']
        config['model']['hyperparameters'] = model_['model']['hyperparameters']
    else:
        model_checkpoint = None
        config['model']['start_checkpoint'] = None
    model_type = config['model']['type']
    model_hp = config['model']['hyperparameters']

    config['evaluation']['model_checkpoint'] = f'{filepath}/best_model'
    dump_yaml(config, f'{filepath}/config.yml')

    # logging
    sys.stdout = open(f'{filepath}/train.log','w')
    
    # loading data
    train_data = read_data(train_datapath)
    val_data = read_data(val_datapath)
    test_data = read_data(test_datapath)

    dim = model_hp['dim']
    nlayers = model_hp['nlayers']
    nheads = model_hp['nheads']
    dropout = model_hp['dropout']
    n_cls = model_hp['n_cls']
    d_ff = model_hp['d_ff']
    decoder_layers = [dim, dim, n_cls]
    
    tokenizer = get_build_tokenizer(f'{filepath}/best_model/vocab.json', train_data, "text")   
    
    train_ds = SentimentDataset(train_data, tokenizer)
    val_ds = SentimentDataset(val_data, tokenizer)
    test_ds = SentimentDataset(test_data, tokenizer)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, 
                            shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=batch_size, 
                        shuffle=True, collate_fn=collate_fn) 
    test_dl = DataLoader(test_ds, batch_size=batch_size, 
                        shuffle=True, collate_fn=collate_fn) 
    
    dump_yaml(config['model'], f'{filepath}/best_model/model_config.yml')   
    ntokens = tokenizer.get_vocab_size()
        
    model = build_model(model_type, ntokens, dim, nheads, d_ff,
                            nlayers, dropout, decoder_layers, device) 
    
    # Training
    if optimizer == 'adam':
        opt = torch.optim.Adam
    elif optimizer == 'sgd':
        opt = torch.optim.SGD
    else:
        raise NotImplementedError(
            'Only adam and sgd available as options!'
        )

    if lossfunc == 'log':
        loss_fn = nn.BCELoss()
    elif lossfunc == 'weighted_log':
        id2label = train_data[target].astype("category").cat.categories.tolist()
        weights = 1 / torch.as_tensor(
            train_data[target].value_counts()[id2label],
            dtype=torch.float32)
        total_sum = weights.sum()
        weights /= total_sum

        loss_fn = nn.BCELoss(weight=weights.to(device))
    else:
        raise NotImplementedError(
            'Only log and weighted_log available as options!'
        )
    
    trainer = Trainer(model, opt, lr, l2, loss_fn, callbacks, 
                      device, filepath, model_checkpoint)
    
    trainer.train(epochs, train_dl, val_dl)
    
    if config["testing"]["mode"]:
        
        model.load_state_dict(torch.load(f'{filepath}/best_model/model.pt')['model_state_dict'])
        test_labels, pred_labels = predictions(model, test_dl, device)    
        
        acc = accuracy(test_labels, pred_labels)
        classification_report = report(test_labels, pred_labels, filepath, mapping=None)
        confusion_matrix = conf_matrix(test_labels, pred_labels)
        
        print(f"Accuracy Score: {np.round(acc,2)}")
        print("\nClassification Report:")
        print(classification_report)
        print("\nConfusion Matrix:")
        print(confusion_matrix)
if __name__ == '__main__':
    main()
