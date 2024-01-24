import numpy as np
import os
import math
import itertools
import clip
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import lpips

from torchvision import models
from tqdm import tqdm
from utils import ensure_checkpoint_exists
from tqdm import tqdm
from natsort import natsorted
from random import shuffle
from models.stylegan2.model import Generator
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms

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


def ImageTransform(resize=256):
    transform = transforms.Compose(
    [
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    return transform



#loading stylegan
def get_stylegan_generator(ckpt_path='pretrained_models/stylegan2-ffhq-config-f.pt'):

    ensure_checkpoint_exists(ckpt_path)

    g_ema = Generator(1024, 512, 8)
    g_ema.load_state_dict(torch.load(ckpt_path)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    mean_latent = g_ema.mean_latent(4096)

    return g_ema, mean_latent


# Define the optimization method
def image_as_reference(ref_image_path, wb_model_name, ckpt_path, log_path):
    # Create a directory to store the final generated adversarial image
    os.makedirs(os.path.join(log_path, wb_model_name), exist_ok=True)
    # For each image, create sub-directory to track the changes over the optimization steps
    os.makedirs(os.path.join(log_path, wb_model_name, 'step_wise_updates'), exist_ok=True)

    # Init stylegan generator
    g_ema, mean_latent = get_stylegan_generator(ckpt_path)

    # Load the target white-box model
    target_model = get_classifier(wb_model_name)
    # Define adversarial target ouput
    y_target = Variable( torch.LongTensor([1]), requires_grad=False).cuda()

    # Define the losses
    loss_ce = nn.BCEWithLogitsLoss()
    percept = lpips.LPIPS(net='vgg').cuda()

    # Load reference image
    transform = ImageTransform()
    ref_img = transform(Image.open(ref_image_path).convert('RGB')).unsqueeze(0).cuda()
    
    # SOURCE IMAGE
    # Generate latent vector for source image
    n_mean_latent = 2
    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device='cuda')
        latent_out = g_ema.style(noise_sample)
        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
    latent_n = latent_mean.detach().clone().unsqueeze(0).repeat(1, 1).unsqueeze(1).repeat(1, 18, 1)

    # Generate corresponding noises
    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(1, 1, 1, 1).normal_())

    # Define the latent vector to be optimized
    latent_n.requires_grad = True
    for noise in noises:
        noise.requires_grad = False

    # Define the optimizer
    optimizer = torch.optim.Adam([latent_n], lr=learning_rate)

    # Save source image
    with torch.no_grad():
        source_img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)
        source_img_gen = ((source_img_gen+1)/2).clamp(0,1)
        torchvision.utils.save_image(source_img_gen,  os.path.join(log_path, wb_model_name, "source_image.jpg"), normalize=True, range=(0, 1))


    for i in range(optimize_steps):
        img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)
        img_gen = ((img_gen+1)/2).clamp(0,1)

        batch, channel, height, width = img_gen.shape
        if height > 256:
            factor = height // 256
            img_gen = img_gen.reshape(batch, channel, height // factor, factor, width // factor, factor)
            img_gen = img_gen.mean([3, 5])

        # Perceptual loss
        p_loss = percept(img_gen, ref_img).sum()
        # L2-loss
        l2_loss = ((latent_mean.detach().clone().unsqueeze(0).repeat(1, 1).unsqueeze(1).repeat(1, 18, 1)  - latent_n) ** 2).sum()
        #forensic classifier loss
        output = target_model(torch.nn.functional.interpolate(img_gen, size=224))
        loss_class = loss_ce(output.squeeze(1), y_target.float())

        # Total loss
        loss =  (perceptual_loss_weightage*p_loss) + (l2_loss_weightage*l2_loss) + (forensic_classifier_loss_weightage*loss_class)

        print(f'[{i+1}/{optimize_steps}] Loss: {loss.item()} Perceptual Loss: {p_loss.item()} L2 Loss: {l2_loss.item()} Forensic Classifier Loss: {loss_class.item()}, WB Prediction: {torch.sigmoid(output).item()}')

        if i % 10 == 0:
            torchvision.utils.save_image(img_gen,  os.path.join(log_path, wb_model_name, 'step_wise_updates', f"{str(i).zfill(5)}.jpg"), normalize=True, range=(0, 1))
            

        optimizer.zero_grad()
        loss.backward()
        if mod_level == 'coarse':
            latent_n.grad[0][4:18] = torch.zeros(14,512)
        elif mod_level == 'medium':
            latent_n.grad[0][0:4] = torch.zeros(4,512)
            latent_n.grad[0][8:18] = torch.zeros(10,512)
        elif mod_level == 'fine':
            # latent_n.grad[0][0:8] = torch.zeros(8,512)
            latent_n.grad[0][0:14] = torch.zeros(14,512)
        optimizer.step()
        # noise_normalize(noises)


    # Save the final optimzed image of the current sample
    torchvision.utils.save_image(img_gen, os.path.join(log_path, wb_model_name, f'final_{mod_level}_adv_image.png'), normalize=True, range=(0,1))




if __name__ == "__main__":
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_file)

    # Load all the hyper parameters
    optimize_steps = int(config.get('OPTIMIZE_PARAMETERS', 'optimize_steps'))
    learning_rate = float(config.get('OPTIMIZE_PARAMETERS', 'learning_rate'))
    perceptual_loss_weightage = float(config.get('OPTIMIZE_PARAMETERS', 'perceptual_loss_weightage'))
    l2_loss_weightage = float(config.get('OPTIMIZE_PARAMETERS', 'l2_loss_weightage'))
    forensic_classifier_loss_weightage = float(config.get('OPTIMIZE_PARAMETERS', 'forensic_classifier_loss_weightage'))
    wb_model_name = str(config.get('OPTIMIZE_PARAMETERS', 'white_box_model_name'))

    log_path = config.get('DATA_PATHS', 'train_log_path')
    ckpt_path = config.get('DATA_PATHS', 'stylegan_weights')

    ref_image_path = str(config.get('ATTRIBUTES', 'reference_image_path'))
    mod_level = str(config.get('ATTRIBUTES', 'level'))

    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, 'optimize_configurations.txt'), 'w') as f:
        f.write(f'Optimize steps : {optimize_steps} \n')
        f.write(f'Learning Rate : {learning_rate} \n')
        f.write(f'Perceptual Loss Weightage : {perceptual_loss_weightage} \n')
        f.write(f'L2 Loss Weightage : {l2_loss_weightage} \n')
        f.write(f'Forensic Classifier Loss Weightage : {forensic_classifier_loss_weightage} \n')
        f.write(f'White Box Model Name : {wb_model_name} \n')
        f.write('\n')
        f.write(f'Reference Image Path : {ref_image_path} \n')
        f.write(f'Modification Level : {mod_level} \n')

    # Image-as-refence
    image_as_reference(ref_image_path, wb_model_name, ckpt_path, log_path)