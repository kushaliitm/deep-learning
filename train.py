import argparse
import helper as hp
import torch
import os
import json

parser = argparse.ArgumentParser(description = 'train.py')

parser.add_argument('--data-dir', nargs = '*', action = "store", default = "./flowers/", help = "folder path for data")
parser.add_argument('--save-dir', action = "store", required=True, help = "filepath for saving checkpoint")
parser.add_argument('--learning-rate', action = "store", default = 0.001, help = "learning rate for the optimizer")
parser.add_argument('--epoch-num', action = "store", type = int, default = 3, help = "epoch value")
parser.add_argument('--architecture', action = "store", default = "vgg16", type = str, help = "specify the neural network structure: vgg16 or densenet121")
parser.add_argument('--hidden-size', type = int, action = "store", default = 1000, help = "state the units for fc2")
parser.add_argument('--optimizer', action='store', default='adam', help='Optimizer to optimize')

pa = parser.parse_args()
pa = vars(pa)
print(pa)
data_path = pa['data_dir']
save_dir = pa["save_dir"]
learning_rate = pa['learning_rate']
architecture = pa['architecture']
hidden_size = pa['hidden_size']
epoch_number = pa['epoch_num']

if (not os.path.exists(f'experiments/{save_dir}')):
    os.makedirs(f'experiments/{save_dir}')
    
file_path = f'experiments/{save_dir}/checkpoint.pt'

# saving parameters
with open(f'experiments/{save_dir}/parameters.json', 'w') as f:
    json.dump(pa, f)
    
# load the data - data_load() from help.py
print('Loading data')
train_loader, validation_loader, test_loader = hp.load_data(data_path)
criterion = torch.nn.NLLLoss()

# build model
print(f'Loading weights from {architecture}')
model, optimizer = hp.get_model_and_optimizer(pa)

# train model
print('Training model')
hp.train_model(model, optimizer, learning_rate,train_loader,validation_loader,criterion,epoch_number, file_path)

# checkpoint the model

print("model has been successfully trained")
