import numpy as np
import os
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from models.gen import GEN
from models.gcn import GCNNet
from utils import *

# Function to make predictions
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor().to(device)
    total_labels = torch.Tensor().to(device)
    print('Making predictions for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1)), 0)
    return total_labels.cpu().numpy().flatten(), total_preds.cpu().numpy().flatten()

datasets = ['davis', 'kiba', 'allergy']
modelings = [GEN, GCNNet]
cuda_name = "cuda:0"
print('cuda_name:', cuda_name)

TEST_BATCH_SIZE = 1
result = []

device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

for dataset in datasets:
    processed_data_file_test = f'data/processed/{dataset}_test.pt'
    if not os.path.isfile(processed_data_file_test):
        print(f'Please run prepare_data.py to prepare data for the {dataset} dataset in PyTorch format!')
        continue
    
    try:
        test_data = TestbedDataset(root='data', dataset=f'{dataset}_test')
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f'Error loading test data for {dataset}: {e}')
        continue
    
    for modeling in modelings:
        model_st = modeling.__name__
        print(f'\nPredicting for {dataset} using {model_st}')
        
        model = modeling().to(device)
        model_file_name = f'model_{model_st}_{dataset}.model'
        
        if not os.path.isfile(model_file_name):
            print(f'Model file {model_file_name} is not available!')
            continue
        
        try:
            model.load_state_dict(torch.load(model_file_name, map_location=device))
        except Exception as e:
            print(f'Error loading model {model_st} for {dataset}: {e}')
            continue

        G, P = predicting(model, device, test_loader)
        ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
        ret = [dataset, model_st] + [round(e, 3) for e in ret]
        result.append(ret)
        
        print('dataset,model,rmse,mse,pearson,spearman,ci')
        print(ret)

with open('result.csv', 'w') as f:
    f.write('dataset,model,rmse,mse,pearson,spearman,ci\n')
    for ret in result:
        f.write(','.join(map(str, ret)) + '\n')

print('Results saved to result.csv')
