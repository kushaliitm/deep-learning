import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import json
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

def load_data(data_path = "./flowers"):
    data_dir = data_path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    return trainloader, testloader, validloader

def load_pretrained_model(model_name, hidden_units):
    if model_name == 'densenet':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(p = 0.4)),
                          ('fc2', nn.Linear(hidden_units, 256)),
                          ('relu2', nn.ReLU()),
                          ('drop2', nn.Dropout(p = 0.3)),
                          ('fc3', nn.Linear(256, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    else:
        model = models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(p = 0.4)),
                          ('fc2', nn.Linear(hidden_units, 256)),
                          ('relu2', nn.ReLU()),
                          ('drop2', nn.Dropout(p = 0.3)),
                          ('fc3', nn.Linear(256, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))   
    model.classifier = classifier
    return model

def get_model_and_optimizer(args):
    model = load_pretrained_model(args['architecture'], args['hidden_size'])
    if (args['optimizer'] == 'adam'):
        optimizer = optim.Adam(model.classifier.parameters(), lr=args['learning_rate'])
        
    return model, optimizer

def train_model(model, optimizer, learning_rate,train_loader,valid_loader,criterion,epoch_number, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device);
    
    epochs = epoch_number
    steps = 0
    running_loss = 0
    print_every = 50
    for epoch in range(epochs):
        save_checkpoint({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, checkpoint_path)
        
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            outputs = model.forward(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model.forward(inputs)
                        batch_loss = criterion(outputs, labels)
                    
                        test_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Training Loss: {running_loss/print_every:.3f}.. "
                  f"validation Loss: {test_loss/len(valid_loader):.3f}.. "
                  f"Test Accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()
                
def validation(model,test_loader,criterion):
    test_loss = 0
    accuracy = 0
    total = 0      
    correct = 0
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels)
            total += labels.size(0) 
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            correct += torch.sum(equals.type(torch.FloatTensor)).item()
    accuracy = correct*100/total
    print('Accuracy of the network on the test images:{:.1f} %'.format(accuracy))
    
def save_checkpoint(state, file_path):
    torch.save(state, file_path)

def load_json(filename):
    with open(filename, 'r') as f:
        cat_to_name = json.load(f, object_pairs_hook=OrderedDict)
    class_labels = train_data.classes
    return cat_to_name,class_labels

def load_saved_model(model, optimizer, experiment):
    print("=> loading checkpoint '{}'".format(experiment))
    
    checkpoint = torch.load(f'experiments/{experiment}/checkpoint.pt')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    if title is not None:
        ax.set_title(title)
    ax.imshow(image)
    
    return ax

def predict(image_path, model, cat_to_name, topk=5):
    # Predict the class (or classes) of an image using a trained deep learning model.
    model.eval()
    image = process_image(image_path)
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    image_tensor.resize_([1, 3, 224, 224])
    model.to('cpu')
    result = torch.exp(model(image_tensor))
    ps, index = result.topk(topk)
    ps, index = ps.detach(), index.detach()
    ps.resize_([topk])
    index.resize_([topk])
    
    ps, index = ps.tolist(), index.tolist()
    labels = []
    for i in index:
        labels.append(cat_to_name[str(i)])
    return ps, index, labels

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_loader = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
            
    return np_image