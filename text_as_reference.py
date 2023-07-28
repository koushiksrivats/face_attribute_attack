import numpy as np
import os
import math
import itertools
import clip
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import models

from tqdm import tqdm
from utils import ensure_checkpoint_exists
from tqdm import tqdm
from natsort import natsorted
from random import shuffle
from models.stylegan2.model import Generator
from torch.autograd import Variable
# from models.facial_recognition.model_irse import Backbone

import argparse
import configparser

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, help='Path to the configuration file')



# To reproduce results
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)



def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


class CLIPLoss(torch.nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.model.eval();
        self.face_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device="cuda").view(1,3,1,1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device="cuda").view(1,3,1,1)

    def forward(self, image, text):
        image = image.sub(self.mean).div(self.std)
        image = self.face_pool(image)
        similarity = 1 - self.model(image, text)[0]/ 100
        return similarity
    


class Normalize(nn.Module):
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input_):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input_ - mean) / std



def get_classifier(name):
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ### dense121
    if name == "densenet121":
        model_dense = models.densenet121(pretrained=True)
        model_dense.classifier = nn.Linear(1024, 1)
        model_dense.load_state_dict(torch.load(os.path.join(config.get('DATA_PATHS', 'forensic_classifier_weights'), name, 'best_epoch.pt'))['model'] )
        model = nn.Sequential(norm_layer,model_dense)
        model = model.to('cuda')  
        model.eval();
        

    ###loading berkley paper classifier
    elif name == "baseline":
        model_res = models.resnet50(num_classes=1)
        model_res.fc = nn.Linear(2048, 2)
        model_res.load_state_dict(torch.load(os.path.join(config.get('DATA_PATHS', 'forensic_classifier_weights'), name, 'best_epoch.pt'))['model'])
        model = nn.Sequential(norm_layer,model_res)
        model = model.to('cuda')  
        model.eval();

    ###res18
    elif name == "resnet18":
        model_res18 = models.resnet18(pretrained=True)
        model_res18.fc = nn.Linear(512, 1)
        model_res18.load_state_dict(torch.load(os.path.join(config.get('DATA_PATHS', 'forensic_classifier_weights'), name, 'best_epoch.pt'))['model'])
        model_res18 = nn.Sequential(norm_layer,model_res18)
        model = model_res18.to('cuda')  
        model.eval();

    ###res50
    elif name == "resnet50":
        model_res50 = models.resnet50(pretrained=True)
        model_res50.fc = nn.Linear(2048, 1)
        model_res50.load_state_dict(torch.load(os.path.join(config.get('DATA_PATHS', 'forensic_classifier_weights'), name, 'best_epoch.pt'))['model'])
        model_res50 = nn.Sequential(norm_layer,model_res50)
        model = model_res50.to('cuda')  
        model.eval();

    ###vgg19
    elif name == "vgg19":
        model_vgg = models.vgg19_bn(pretrained=True)
        model_vgg.classifier[6] = nn.Linear(4096,1)
        model_vgg.load_state_dict(torch.load(os.path.join(config.get('DATA_PATHS', 'forensic_classifier_weights'), name, 'best_epoch.pt'))['model'])
        model_vgg = nn.Sequential(norm_layer,model_vgg)
        model = model_vgg.to('cuda')  
        model.eval();

    # efficientnet
    elif name=='efficientnet':
        from efficientnet_pytorch import EfficientNet
        
        model = EfficientNet.from_name('efficientnet-b3')
        model._fc = nn.Linear(1536, 1)
        model.load_state_dict(torch.load(os.path.join(config.get('DATA_PATHS', 'forensic_classifier_weights'), name, 'best_epoch.pt'))['model'])
        model = nn.Sequential(norm_layer,model)
        model = model.to('cuda')  
        model.eval()

    # xception
    elif name=='xception':
        from models.forensic_classifiers.xception import Xception
        model = Xception(num_classes=1)
        model.load_state_dict(torch.load(os.path.join(config.get('DATA_PATHS', 'forensic_classifier_weights'), name, 'best_epoch.pt'))['model'])
        model = nn.Sequential(norm_layer,model)
        model = model.to('cuda')  
        model.eval()
        
    return model



#loading stylegan
def get_stylegan_generator(ckpt_path='pretrained_models/stylegan2-ffhq-config-f.pt'):

    ensure_checkpoint_exists(ckpt_path)

    g_ema = Generator(1024, 512, 8)
    g_ema.load_state_dict(torch.load(ckpt_path)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    mean_latent = g_ema.mean_latent(4096)

    return g_ema, mean_latent


def get_latent_and_prompt(g_ema, mean_latent, ip_prompt):
    text_inputs = torch.cat([clip.tokenize(ip_prompt)]).cuda()
    # Generate an image (select a z) to optimize by selecting the best of n Zs
    latent_code_init_not_trunc = torch.randn(10, 512).cuda()
    with torch.no_grad():
        _, latent_code_init, _ = g_ema([latent_code_init_not_trunc], return_latents=True,
                                    truncation=0.5, truncation_latent=mean_latent)
        im,_ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)

    min_idx = clip_loss(im, text_inputs)

    # Choose the latent with the min
    latent = latent_code_init[min_idx.argmin()].unsqueeze(0).detach().clone()
    latent.requires_grad = True
    
    image = im[min_idx.argmin()].unsqueeze(0).detach().clone()
    
    return latent, text_inputs, ip_prompt, image



# Define all the training methods
def text_as_reference(ip_prompt, wb_model_name, ckpt_path, log_path):
    # Init stylegan generator
    g_ema, mean_latent = get_stylegan_generator(ckpt_path)

    # Get the inital latent code
    latent, text_inputs, prmpt_str, image = get_latent_and_prompt(g_ema, mean_latent, ip_prompt)

    # Load the target white-box model
    target_model = get_classifier(wb_model_name)

    # Define adversarial target ouput
    y_target = Variable( torch.LongTensor([1]), requires_grad=False).cuda()

    # Define the loss
    loss_ce = nn.BCEWithLogitsLoss()

    # Create a directory to store the final generated adversarial image
    os.makedirs(os.path.join(log_path, wb_model_name), exist_ok=True)
    # For each image, create sub-directory to track the changes over the optimization steps
    os.makedirs(os.path.join(log_path, wb_model_name, 'step_wise_updates'), exist_ok=True)

    # OPTIMIZE
    latent_dummy = latent.detach().clone()

    # Generate the noises latents to be used
    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(1, 1, 1, 1).normal_())
    for noise in noises:
        noise.requires_grad = True

    # Optimize for the given latent and noise
    optimizer = torch.optim.Adam([latent]+noises, lr=learning_rate)

    # Optimze for the given iterations
    for i in range(optimize_steps):
        img_gen, _ = g_ema([latent], input_is_latent=True, noise = noises)
        img_gen = ((img_gen+1)/2).clamp(0,1)

        c_loss = clip_loss(img_gen, text_inputs)
        l2_loss = ((latent_dummy - latent) ** 2).sum()

        #forensic classifier
        output = target_model(torch.nn.functional.interpolate(img_gen, size=224))
        loss_class = loss_ce(output.squeeze(1), y_target.float())

        # For every n iteration save the generated image
        if i % 5 == 0:
            torchvision.utils.save_image(img_gen,  os.path.join(log_path, wb_model_name, 'step_wise_updates', f"{str(i).zfill(5)}.jpg"), normalize=True, range=(0, 1))

        # Compute final loss and optimze
        loss = (clip_loss_weightage*c_loss)  + (l2_loss_weightage*l2_loss) + (forensic_classifier_loss_weightage*loss_class)

        print(f'Iteration: {i}, Total Loss: {loss.item()}, CLIP Loss: {c_loss.item()}, L2 Loss: {l2_loss.item()}, Forensic Classifier Loss: {loss_class.item()}, wb model prediction: {torch.sigmoid(output).item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save the final optimzed image of the current sample
    torchvision.utils.save_image(img_gen, os.path.join(log_path, wb_model_name, f'final_{prmpt_str}.png'), normalize=True, range=(0,1))





if __name__ == "__main__":
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_file)

    # Load all the hyper parameters
    optimize_steps = int(config.get('OPTIMIZE_PARAMETERS', 'optimize_steps'))
    learning_rate = float(config.get('OPTIMIZE_PARAMETERS', 'learning_rate'))
    clip_loss_weightage = float(config.get('OPTIMIZE_PARAMETERS', 'clip_loss_weightage'))
    l2_loss_weightage = float(config.get('OPTIMIZE_PARAMETERS', 'l2_loss_weightage'))
    forensic_classifier_loss_weightage = float(config.get('OPTIMIZE_PARAMETERS', 'forensic_classifier_loss_weightage'))
    wb_model_name = str(config.get('OPTIMIZE_PARAMETERS', 'white_box_model_name'))
    ip_prompt = str(config.get('ATTRIBUTES', 'prompt'))

    log_path = config.get('DATA_PATHS', 'train_log_path')

    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, 'optimize_configurations.txt'), 'w') as f:
        f.write(f'Optimize steps : {optimize_steps} \n')
        f.write(f'Learning Rate : {learning_rate} \n')
        f.write(f'Clip Loss Weightage : {clip_loss_weightage} \n')
        f.write(f'L2 Loss Weightage : {l2_loss_weightage} \n')
        f.write(f'Forensic Classifier Loss Weightage : {forensic_classifier_loss_weightage} \n')
        f.write(f'Text Prompt : {ip_prompt} \n')

    # Init clip loss
    clip_loss = CLIPLoss().cuda()

    # Text-as-refence
    ckpt_path = config.get('DATA_PATHS', 'stylegan_weights')
    text_as_reference(ip_prompt, wb_model_name, ckpt_path, log_path)