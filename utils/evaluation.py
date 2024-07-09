import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def predictions(model, test_dl, device='cpu'):
    """
    Function to get classifcaiton predictions from a model and test_dataloader

    Args:
        model: model to get predictions from
        test_dl: test dataloader containing data as well as true labels
        device: device to run model on ('cpu'/'cuda')

    Return:
        true_labels from test set, predicted labels via inference on test data
    """
    model.eval()
    test_labels, pred_labels = [], []

    for batch in test_dl:
        with torch.no_grad():
            x, y, mask = [ele.to(device) for ele in batch]
        
        out = model(x, mask)
        test_labels += y.tolist()
        pred_labels += (out >= 0.5).float().tolist()
        # print(f'Labels: {test_labels}', flush=True)
        # print(f'Labels: {pred_labels}', flush=True)
        # print(f'Prediction: {out}', flush=True)
    return test_labels, pred_labels

def accuracy(test_labels, pred_labels):
    """
    Function to return accuracy score of predicted labels as compared to true labels
    """
    return accuracy_score(test_labels, pred_labels)

def report(test_labels, pred_labels, filepath, mapping=None):
    """
    Function to generate a classifcaiton report from a from the predicted data
    at filepath as classification_report.csv

    Args:
        test_labels: true labels from test set
        pred_labels: predicted labels from trained model
        filepath: path to store classification_report.csv
        mapping[optional]: mapping of label_id to true label_names (id2label)
    """
    
    if mapping is not None:
        test_labels = list(map(lambda x: mapping[x], test_labels))
        pred_labels = list(map(lambda x: mapping[x], pred_labels))
        
    report = pd.DataFrame(classification_report(test_labels, pred_labels, output_dict=True)).transpose()
    report.to_csv(f'{filepath}/best_model/classification_report.csv')
    return report

def conf_matrix(test_labels, pred_labels):
    """
    Function to return confusion matrix of predicted labels as compared to true labels
    """    
    return confusion_matrix(test_labels, pred_labels)













