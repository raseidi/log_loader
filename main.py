import os
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from torch.utils.data import DataLoader

from log_loader.loader import LogForML

def train_val(name):
    return name.startswith('train_val_')

data_path = 'data/split_datasets'
for path, subdirs, files in os.walk(data_path):
    for name in files:
        file = os.path.join(path, name)
        if train_val(os.path.basename(file)):
            print(file)
            # log = xes_importer.apply(file)
            # log_dataset = LogForML(log=log, prefix_len=5)
            # data_loader = DataLoader(log_dataset, batch_size=2)
            # print(os.path.basename(file))
            # print(type(log_dataset))
            # print(type(data_loader))
            # print('-'*20)
            # print('\n')
