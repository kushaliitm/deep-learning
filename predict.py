import json
import argparse
import numpy as np
import helper as hp

parser = argparse.ArgumentParser(description = 'predict.py')

parser.add_argument('--input_img', default = 'flowers/test/5/image_05159.jpg', action = "store", type = str, help = "image path")
parser.add_argument('--checkpoint', default = 'temp', action = "store", type = str, help = "path from where the checkpoint is loaded")
parser.add_argument('--top_k', default = 5, dest = "top_k", action = "store", type = int)
parser.add_argument('--category_names', dest = "category_names", action = "store", default = 'cat_to_name.json')

pa = parser.parse_args()
pa = vars(pa)
print(pa)

image_path = pa['input_img']
topk = pa['top_k']
input_img = pa['input_img']
checkpoint_path = pa['checkpoint']


with open(f'experiments/{checkpoint_path}/parameters.json', 'r') as f:
    parameters = json.load(f)
    
training_loader, testing_loader, validation_loader = hp.load_data()

# load previously saved checkpoint

model, optimizer = hp.get_model_and_optimizer(parameters)
model, _, _ = hp.load_saved_model(model, optimizer, checkpoint_path)

# load label conversion
with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)

ps, index, labels = hp.predict(image_path, model, cat_to_name, topk)
for i in range(len(ps)):
    print("The probability of the flower to be {} is {:.2f} %.".format(labels[i], ps[i] * 100))