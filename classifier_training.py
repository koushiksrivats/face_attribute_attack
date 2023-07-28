#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
from torchvision import models
from torchvision import transforms
import torchvision.models as models
from tqdm import tqdm
import argparse
import os
import wandb

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('--train_data', type=str, help='path to the train data')
parser.add_argument('--test_data', type=str, help='path to the test data')
parser.add_argument('--batch_size', type=int, help='batch size for training')
parser.add_argument('--epochs', type=int, help='# epochs to train the training')
parser.add_argument('--classifier_name', type=str, help='Choose from the list to train (resnet18, resnet50, densenet121, vgg19, efficientnet, xception)')
parser.add_argument('--output_path', type=str, help='Path to save the output model')
parser.add_argument('--wandb_project_name', type=str, help='Name of wandb project to save the logs')
parser.add_argument('--experiment_name', type=str, help='Name of wandb experiment to save the logs')
parser.add_argument('--resume_training', default=False, type=bool, help='Flag to resume traininig from chekpoint')




def get_model(name):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Changing number of model's output classes to 1
    #for resnet18
    if name == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, 1)

        train_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])
        test_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])

    #for resnet50
    elif name == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 1)

        train_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])
        test_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])

    # for densenet 121
    elif name == 'densenet121':
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(1024, 1)

        train_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])
        test_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])

    #for vgg19_bn
    elif name == 'vgg19':
        model = models.vgg19_bn(pretrained=False)
        model.classifier[6] = nn.Linear(4096,1)

        train_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])
        test_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])

    #for efficientnet
    elif name=='efficientnet':
        from efficientnet_pytorch import EfficientNet
        
        model = EfficientNet.from_name('efficientnet-b3')
        model._fc = nn.Linear(1536, 1)

        train_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])
        test_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])


    #for xception
    elif name=='xception':
        from models.forensic_classifiers.xception import Xception
        model = Xception(num_classes=1)

        train_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])
        test_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])

    # Transfer execution to GPU
    model = model.to('cuda')
    
    return model , train_transform, test_transform



# Load the weigths for the model
def load_weights(model, path):
    model.load_state_dict(torch.load(path))
    return model



### Training
def train(model, train_loader, test_loader, start_epoch, end_epoch, op_folder, bs, wandb, resume_training=False):
    optimizer = optim.Adam(model.parameters(),lr=0.0002)
    loss_func = nn.BCEWithLogitsLoss()#nn.CrossEntropyLoss()
    best_test_acc = 0

    # Load model and optimizer state dicts if resume flag is true
    if resume_training:
        ckpt = torch.load(os.path.join(op_folder, 'best_epoch.pt'))
        # Load model
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = int(ckpt['epoch']) + 1
        best_test_acc = ckpt['test_acc']
        print(f'Checkpoint file loaded. Resuming from epoch :{start_epoch}, where the best accuracy was : {best_test_acc}')

    for epoch in tqdm(range(start_epoch, end_epoch)):    
        model.train()        
        for ii, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()              
            optimizer.zero_grad()
            output = model(data)                    
            loss = loss_func(output.squeeze(1), target.float())
            print(f'epoch : {epoch}, step: {ii}, loss : {loss}')
            loss.backward()
            optimizer.step() 
            
            stats = dict(epoch=epoch, step=ii, train_classifier_loss=loss)
            wandb.log(stats)

        # Test every epoch 
        test_acc = test(model, test_loader, bs)
        test_stats = dict(epoch=epoch, test_accuracy=test_acc)
        wandb.log(test_stats)

        # Save the epoch with best score
        if test_acc >= best_test_acc:
            best_test_acc = test_acc

            train_state = dict(model=model.state_dict(),
                                optimizer=optimizer.state_dict(),
                                epoch = epoch,
                                test_acc = best_test_acc
                                )

            print(f'Saving best epoch at epoch {epoch}')
            save_filename = str(op_folder)+'best_epoch.pt'
            torch.save(train_state, save_filename)



### Testing
def test(model, test_loader, bs):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for ii, (data, target) in enumerate(test_loader): 

            data, target = data.cuda(), target.cuda() 
            output = model(data)
            predicted = torch.round(torch.sigmoid(output))
            total += target.size(0)
            correct += (predicted == target).sum().item()/bs


    #loss_log.append(loss.item())       
    acc = 100 * correct // total
    print(f'Accuracy of the network on the 2000 test images: {acc} %')
    return acc



if __name__ == "__main__":
    args = parser.parse_args()

    # Train
    model, train_transform, test_transform = get_model(args.classifier_name)
    
    train_dataset = torchvision.datasets.ImageFolder(root=args.train_data, transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=args.test_data, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size ,shuffle=False, drop_last=False, num_workers=8)

    start_epoch = 0
    end_epoch = args.epochs

    # Initialise wandb logging
    config_values = vars(args)
    config_values['train_dataset_size'] = int(train_dataset.__len__())
    config_values['test_dataset_size'] = int(test_dataset.__len__())
    wandb.init(
        # Set the project where this run will be logged
        project=args.wandb_project_name, 
        name=args.experiment_name,
        # Track hyperparameters and run metadata
        config=config_values,
        id=args.experiment_name, 
        resume=True
        )

    os.makedirs(args.output_path, exist_ok=True)
    train(model, train_loader, test_loader, start_epoch, end_epoch, args.output_path, args.batch_size, wandb, args.resume_training)
