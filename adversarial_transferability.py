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
        #image = image.add(1).div(2)
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


def compute_accuracy(pred):
    pred = np.asarray(pred)
    label = np.ones(np.shape(pred))
    correct = (pred == label).sum()
    
    accuracy = 100 * (correct/ float(len(pred)))
    # print(f'Accuracy : {accuracy}')
    return accuracy


#loading stylegan
def get_stylegan_generator(ckpt_path='pretrained_models/stylegan2-ffhq-config-f.pt'):

    ensure_checkpoint_exists(ckpt_path)

    g_ema = Generator(1024, 512, 8)
    g_ema.load_state_dict(torch.load(ckpt_path)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    mean_latent = g_ema.mean_latent(4096)

    return g_ema, mean_latent


def get_latent_and_prompt(g_ema, mean_latent, prompt_strings):
    ip_prompt = np.random.choice(prompt_strings)
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



# Generate and save test images and starting Zs
def generate_and_save_initial_latents(num_of_test_images, save_path='results/'):
    images = []
    latents = []
    text_prompts = []
    chosen_prompt_strings = []
    for i in tqdm(range(num_of_test_images)):
        latent , prompt, prompt_string, image = get_latent_and_prompt(g_ema, mean_latent, prompt_strings)
        latents.append(latent)
        text_prompts.append(prompt)
        chosen_prompt_strings.append(prompt_string)
        images.append(image)

    # return images, latents, text_prompts, chosen_prompt_strings

    os.makedirs(save_path, exist_ok=True)

    # Save the initial latents with its corresponding prompts
    save_dict = dict(latents=latents, text_prompts=text_prompts, chosen_prompt_strings=chosen_prompt_strings)
    torch.save(save_dict, os.path.join(save_path ,'initial_latents_and_prompts.pt'))

    # Save images
    for i, (img, prmpt) in enumerate(zip(images, chosen_prompt_strings)):
        torchvision.utils.save_image((img+1)/2, save_path+f'z_{i}_{str(prmpt)}.png')



# Load the latents
def load_start_latents(start_latents_path):
    data = torch.load(start_latents_path)

    latents = data['latents']
    text_prompts = data['text_prompts']
    chosen_prompt_strings = data['chosen_prompt_strings']

    return latents, text_prompts, chosen_prompt_strings


# Define all the training methods
def one_vs_many_classifier(start_latents_path, all_wb_model_names, log_path):
    # Load the stored latents
    stored_latents, stored_text_prompts, stored_chosen_prompt_strings = load_start_latents(start_latents_path)

    # pre-load all the models from the list
    all_models = {}
    for classifier_name in all_wb_model_names:
        all_models[classifier_name] = get_classifier(classifier_name)

    # Define the loss
    loss_ce= nn.BCEWithLogitsLoss()

    # Create a dataframe to store all the results
    output_df = [['Setting:', 'One vs Many']]

    # Loop over all the models and optimize each sample
    for whitebox_classifier_name in (pbar := tqdm(all_wb_model_names)):
        output_df.append(['',''])
        output_df.append(['White-Box:', whitebox_classifier_name])
        pbar.set_description(f'Processing {whitebox_classifier_name} as white-box classifier')

        # Create a dict to track the predictions
        classifier_predictions = {}
        for c_name in all_wb_model_names:
            classifier_predictions[c_name] = []

        # Create a directory to store the final generated adversarial image
        os.makedirs(os.path.join(log_path, 'one_vs_many', whitebox_classifier_name), exist_ok=True)

        # Define target ouput and model
        y_target = Variable( torch.LongTensor([1]), requires_grad=False).cuda()
        target_model = all_models[whitebox_classifier_name]

        # Loop over all the samples and optimize
        for ind, (latent, text_inputs, prmpt_str) in tqdm(enumerate(zip(stored_latents, stored_text_prompts, stored_chosen_prompt_strings))):
            latent_dummy = latent.detach().clone()

            # For each image, create sub-directory to track the changes over the optimization step
            os.makedirs(os.path.join(log_path, 'one_vs_many', 'step_wise_updates', str(ind)), exist_ok=True)

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
                    torchvision.utils.save_image(img_gen,  os.path.join(log_path, 'one_vs_many', 'step_wise_updates', str(ind), f"{str(i).zfill(5)}.jpg"), normalize=True, range=(0, 1))

                # Compute final loss and optimze
                loss = (clip_loss_weightage*c_loss)  + (l2_loss_weightage*l2_loss) + (forensic_classifier_loss_weightage*loss_class)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Save the final optimzed image of the current sample
            torchvision.utils.save_image(img_gen, os.path.join(log_path, 'one_vs_many', whitebox_classifier_name, f'final_{ind}_{prmpt_str}.png'), normalize=True, range=(0,1))

            # After the latents have been optimized evaluate the image on other models in the black box setting
            for key in classifier_predictions.keys():
                eval_model = all_models[key]
                eval_model_prediction = eval_model(torch.nn.functional.interpolate(img_gen, size=224))
                classifier_predictions[key].append(torch.round(torch.sigmoid(eval_model_prediction)).detach().cpu().item())


        # Compute the final accuracy for all the samples for the current setting
        stats_df = [['Model', 'Accuracy']]

        for model_name in classifier_predictions.keys():
            preds = classifier_predictions[model_name]
            stats_df.append([model_name, compute_accuracy(preds)])

        output_df.extend(stats_df)
        output_df.append(['',''])

        print(whitebox_classifier_name)
        print(stats_df)

    # Save the final scores
    output_df = pd.DataFrame(output_df)
    output_df.to_csv(os.path.join(log_path, 'one_vs_many', 'accuracy.csv'))



def ensemble_classifier(start_latents_path, wb_model_combination, all_wb_model_names, log_path):
    # Combine the names of the current combination of the white box classifiers to form the folder and experiment name
    folder_name = '_'.join(wb_model_combination)

    # Load the stored latents
    stored_latents, stored_text_prompts, stored_chosen_prompt_strings = load_start_latents(start_latents_path)

    # pre-load all the models from the list
    all_models = {}
    for classifier_name in all_wb_model_names:
        all_models[classifier_name] = get_classifier(classifier_name)

    # Define the loss
    loss_ce= nn.BCEWithLogitsLoss()

    # Create a dataframe to store all the results
    output_df = [['White-Box:', folder_name]]

    # Create a dict to track the predictions
    classifier_predictions = {}
    for c_name in all_wb_model_names:
        classifier_predictions[c_name] = []

    # Create a directory to store the final generated adversarial image
    os.makedirs(os.path.join(log_path, 'ensemble', folder_name), exist_ok=True)

    # Define target output
    y_target = Variable( torch.LongTensor([1]), requires_grad=False).cuda()

    # optimize each sample using the average loss of n-1 white box classifiers
    for ind, (latent, text_inputs, prmpt_str) in tqdm(enumerate(zip(stored_latents, stored_text_prompts, stored_chosen_prompt_strings))):
        latent_dummy = latent.detach().clone()

        # For each image, create sub-directory to track the changes over the optimization step
        os.makedirs(os.path.join(log_path, 'ensemble', 'step_wise_updates', str(ind)), exist_ok=True)

        # Generate the noises latents to be used
        noises_single = g_ema.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(1, 1, 1, 1).normal_())
        for noise in noises:
            noise.requires_grad = True

        # Optimize for the given latent and noise
        optimizer = torch.optim.Adam([latent]+noises, lr=learning_rate)

        # Optimize for the given iterations
        for i in range(optimize_steps):
            img_gen, _ = g_ema([latent], input_is_latent=True, noise = noises)
            img_gen = ((img_gen+1)/2).clamp(0,1)

            c_loss = clip_loss(img_gen, text_inputs)
            l2_loss = ((latent_dummy - latent) ** 2).sum()

            # iterate over the combinations of forensic classifier and get the average loss
            forensic_classifier_losses = []
            for forensic_classifier in wb_model_combination:
                target_model = all_models[forensic_classifier]
                output = target_model(torch.nn.functional.interpolate(img_gen, size=224))
                target_classification_loss = loss_ce(output.squeeze(1), y_target.float())
                forensic_classifier_losses.append(target_classification_loss)

            # Get the final average classification loss
            loss_class = torch.mean(torch.stack(forensic_classifier_losses))

            # For every n iteration save the generated image
            if i % 5 == 0:
                torchvision.utils.save_image(img_gen,  os.path.join(log_path, 'ensemble', 'step_wise_updates', str(ind), f"{str(i).zfill(5)}.jpg"), normalize=True, range=(0, 1))

            # Compute final loss and optimize
            loss = (clip_loss_weightage*c_loss)  + (l2_loss_weightage*l2_loss) + (forensic_classifier_loss_weightage*loss_class)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save the final optimzed image of the current sample
        torchvision.utils.save_image(img_gen, os.path.join(log_path, 'ensemble', folder_name, f'final_{ind}_{prmpt_str}.png'), normalize=True, range=(0,1))

        # After the latents have been optimized evaluate the image on other models in the black box setting
        for key in classifier_predictions.keys():
            eval_model = all_models[key]
            eval_model_prediction = eval_model(torch.nn.functional.interpolate(img_gen, size=224))
            classifier_predictions[key].append(torch.round(torch.sigmoid(eval_model_prediction)).detach().cpu().item())


    # Compute the final accuracy for all the samples for the current setting
    stats_df = [['Model', 'Accuracy']]

    for model_name in classifier_predictions.keys():
        preds = classifier_predictions[model_name]
        stats_df.append([model_name, compute_accuracy(preds)])

    output_df.extend(stats_df)
    output_df.append(['',''])

    print(folder_name)
    print(stats_df)

    return output_df



def meta_classifier(start_latents_path, wb_model_combination, all_wb_model_names, log_path):
    # Combine the names of the current combination of the white box classifiers to form the folder and experiment name
    folder_name = '_'.join(wb_model_combination)

    # Load the stored latents
    stored_latents, stored_text_prompts, stored_chosen_prompt_strings = load_start_latents(start_latents_path)

    # pre-load all the models from the list
    all_models = {}
    for classifier_name in all_wb_model_names:
        all_models[classifier_name] = get_classifier(classifier_name)

    # Define the loss
    loss_ce= nn.BCEWithLogitsLoss()

    # Create a dataframe to store all the results
    output_df = [['White-Box:', folder_name]]

    # Create a dict to track the predictions
    classifier_predictions = {}
    for c_name in all_wb_model_names:
        classifier_predictions[c_name] = []

    # Create a directory to store the final generated adversarial image
    os.makedirs(os.path.join(log_path, 'meta', folder_name), exist_ok=True)

    # Define target output
    y_target = Variable( torch.LongTensor([1]), requires_grad=False).cuda()

    meta_train_models = list(wb_model_combination)

    # optimize each sample using the average loss of n-1 white box classifiers
    for ind, (latent, text_inputs, prmpt_str) in tqdm(enumerate(zip(stored_latents, stored_text_prompts, stored_chosen_prompt_strings))):
        latent_dummy = latent.detach().clone()

        # For each image, create sub-directory to track the changes over the optimization step
        os.makedirs(os.path.join(log_path, 'meta', 'step_wise_updates', str(ind)), exist_ok=True)

        # Generate the noises latents to be used
        noises_single = g_ema.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(1, 1, 1, 1).normal_())
        for noise in noises:
            noise.requires_grad = True

        # Optimize for the given latent and noise
        optimizer = torch.optim.Adam([latent]+noises, lr=learning_rate)

        # Optimize for the given iterations
        for i in range(optimize_steps):
            shuffle(meta_train_models)
            
            overall_target_loss = []
            img_gen, _ = g_ema([latent], input_is_latent=True, noise = noises)
            img_gen = ((img_gen+1)/2).clamp(0,1)
            
            # Save image every 5 steps
            if i % 5 == 0:
                torchvision.utils.save_image(img_gen,  os.path.join(log_path, 'meta', 'step_wise_updates', str(ind), f"{str(i).zfill(5)}.jpg"), normalize=True, range=(0, 1))

            c_loss = clip_loss(img_gen, text_inputs)
            l2_loss = ((latent_dummy - latent) ** 2).sum()
            
            # Choose all models expect one for meta-train
            for fr_cls_model_name in meta_train_models[:-1]:
                # Meta train 
                #forensic classifier
                fr_cls_model = all_models[fr_cls_model_name]
                fr_cls_prediction = fr_cls_model(torch.nn.functional.interpolate(img_gen, size=224))
                fr_cls_loss = loss_ce(fr_cls_prediction.squeeze(1), y_target.float())    
                overall_target_loss.append(fr_cls_loss)

                # Compute gradients wrt to the meta train classifier loss and the start latents and start noises
                grad_latent_1 = torch.autograd.grad(fr_cls_loss, [latent], retain_graph=True)
                grad_noises_1 = torch.autograd.grad(fr_cls_loss, noises, retain_graph=True)

                fast_weights_latents = list(map(lambda p: p[1] - update_learning_rate * p[0], zip(grad_latent_1, [latent])))
                fast_weights_noises = list(map(lambda p: p[1] - update_learning_rate * p[0], zip(grad_noises_1, noises)))

                assert (latent.shape == fast_weights_latents[0].shape)
                assert (noises[0].shape == fast_weights_noises[0].shape)

                # Meta-test: Generate img with the latents updated from meta-train
                img_gen_test, _ = g_ema(fast_weights_latents, input_is_latent=True, noise = fast_weights_noises)
                img_gen_test = ((img_gen_test+1)/2).clamp(0,1)
                
                # meta-test classifier
                meta_te_loss = []
                for test_model_name in meta_train_models[-1:]:
                    meta_test_model = all_models[test_model_name]
                    meta_test_output = meta_test_model(torch.nn.functional.interpolate(img_gen_test, size=224))
                    meta_test_loss_class = loss_ce(meta_test_output.squeeze(1), y_target.float())
                    meta_te_loss.append(meta_test_loss_class)


                overall_target_loss.append(torch.mean(torch.stack(meta_te_loss)))
            
            loss_class = torch.mean(torch.stack(overall_target_loss))

            # compute the final loss and optimize
            loss = (clip_loss_weightage*c_loss)  + (l2_loss_weightage*l2_loss) + (forensic_classifier_loss_weightage*loss_class)
            # print(f'step : {i}, clip loss : {c_loss.item()}, l2 loss : {l2_loss.item()}, overall classifier loss : {loss_class.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save the final optimzed image of the current sample
        with torch.no_grad():
            final_img_gen, _ = g_ema([latent], input_is_latent=True, noise = noises)
            final_img_gen = ((final_img_gen+1)/2).clamp(0,1)
            torchvision.utils.save_image(final_img_gen, os.path.join(log_path, 'meta', folder_name, f'final_{ind}_{prmpt_str}.png'), normalize=True, range=(0,1))

        # After the latents have been optimized evaluate the image on other models in the black box setting
        for key in classifier_predictions.keys():
            eval_model = all_models[key]
            eval_model_prediction = eval_model(torch.nn.functional.interpolate(final_img_gen, size=224))
            classifier_predictions[key].append(torch.round(torch.sigmoid(eval_model_prediction)).detach().cpu().item())


    # Compute the final accuracy for all the samples for the current setting
    stats_df = [['Model', 'Accuracy']]

    for model_name in classifier_predictions.keys():
        preds = classifier_predictions[model_name]
        stats_df.append([model_name, compute_accuracy(preds)])

    output_df.extend(stats_df)
    output_df.append(['',''])

    print(folder_name)
    print(stats_df)

    return output_df




if __name__ == "__main__":
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_file)

    # Load all the hyper parameters
    optimize_steps = int(config.get('OPTIMIZE_PARAMETERS', 'optimize_steps'))
    learning_rate = float(config.get('OPTIMIZE_PARAMETERS', 'learning_rate'))
    update_learning_rate = float(config.get('OPTIMIZE_PARAMETERS', 'update_learning_rate'))
    clip_loss_weightage = float(config.get('OPTIMIZE_PARAMETERS', 'clip_loss_weightage'))
    l2_loss_weightage = float(config.get('OPTIMIZE_PARAMETERS', 'l2_loss_weightage'))
    forensic_classifier_loss_weightage = float(config.get('OPTIMIZE_PARAMETERS', 'forensic_classifier_loss_weightage'))

    # Execute options
    generate_latents = config.getboolean('EXECUTE_OPTIONS', 'generate_latents')
    train_option = config.get('EXECUTE_OPTIONS', 'option')

    os.makedirs(config.get('DATA_PATHS', 'train_log_path'), exist_ok=True)
    with open(os.path.join(config.get('DATA_PATHS', 'train_log_path'), 'optimize_configurations.txt'), 'w') as f:
        f.write(f'Optimize steps : {optimize_steps} \n')
        f.write(f'Learning Rate : {learning_rate} \n')
        f.write(f'Update Learning Rate : {update_learning_rate} \n')
        f.write(f'Clip Loss Weightage : {clip_loss_weightage} \n')
        f.write(f'L2 Loss Weightage : {l2_loss_weightage} \n')
        f.write(f'Forensic Classifier Loss Weightage : {forensic_classifier_loss_weightage} \n')

    # Init clip loss
    clip_loss = CLIPLoss().cuda()

    # Init stylegan generator
    g_ema, mean_latent = get_stylegan_generator(ckpt_path=config.get('DATA_PATHS', 'stylegan_weights') )


    # Generate base latents
    if generate_latents :
        # Define the set of prompts to be randomly used for generating the initial latents
        prompt_strings = [ "Curly long hair",
                        "This man has big nose",
                        "Professor with white hair",
                        "A smiling Chinese girl",
                        "He has mohawk hairstyle",
                        "Red lipstick"]

        # Generate and save latents and the corresponding images
        num_test_samples = 2
        generate_and_save_initial_latents(num_test_samples, save_path=str(config.get('DATA_PATHS', 'base_latent_path')))


    all_wb_model_names = ['densenet121', 'resnet50', 'resnet18', 'vgg19', 'efficientnet', 'xception']
    
    # Optimize samples based on the chosen method
    if train_option == "one_vs_many":
        one_vs_many_classifier(os.path.join(config.get('DATA_PATHS', 'base_latent_path'), 'initial_latents_and_prompts.pt'), all_wb_model_names, config.get('DATA_PATHS', 'train_log_path'))


    elif train_option == "ensemble":
        # Leave one of the classifier as the black-box and choose the rest of the classifiers as white-box
        all_possible_combinations = list(itertools.combinations(all_wb_model_names, len(all_wb_model_names)-1 ))
        output_df = [['Setting:', 'Ensemble'], ['','']]
        for wb_model_combination in (pbar := tqdm(all_possible_combinations)):
            pbar.set_description(f'Processing the {wb_model_combination} combination')
            accuracy_metrics = ensemble_classifier(os.path.join(config.get('DATA_PATHS', 'base_latent_path'), 'initial_latents_and_prompts.pt'), wb_model_combination, all_wb_model_names, config.get('DATA_PATHS', 'train_log_path'))
            output_df.extend(accuracy_metrics)
        # Save the final scores
        output_df = pd.DataFrame(output_df)
        output_df.to_csv(os.path.join(config.get('DATA_PATHS', 'train_log_path'), 'ensemble', 'ensemble_accuracy.csv'))


    elif train_option == "meta":
        # Leave one of the classifier as the black-box and choose the rest of the classifiers as white-box
        all_possible_combinations = list(itertools.combinations(all_wb_model_names, len(all_wb_model_names)-1 ))
        output_df = [['Setting:', 'Meta'], ['','']]
        for wb_model_combination in (pbar := tqdm(all_possible_combinations)):
            pbar.set_description(f'Processing the {wb_model_combination} combination')
            accuracy_metrics = meta_classifier(os.path.join(config.get('DATA_PATHS', 'base_latent_path'), 'initial_latents_and_prompts.pt'), wb_model_combination, all_wb_model_names, config.get('DATA_PATHS', 'train_log_path'))
            output_df.extend(accuracy_metrics)
        # Save the final scores
        output_df = pd.DataFrame(output_df)
        output_df.to_csv(os.path.join(config.get('DATA_PATHS', 'train_log_path'), 'meta', 'meta_accuracy.csv'))


    elif train_option == "all":
        # one vs many
        one_vs_many_classifier(os.path.join(config.get('DATA_PATHS', 'base_latent_path'), 'initial_latents_and_prompts.pt'), all_wb_model_names, config.get('DATA_PATHS', 'train_log_path'))

        # ensemble
        # Leave one of the classifier as the black-box and choose the rest of the classifiers as white-box
        all_possible_combinations = list(itertools.combinations(all_wb_model_names, len(all_wb_model_names)-1 ))
        output_df = [['Setting:', 'Ensemble'], ['','']]
        for wb_model_combination in (pbar := tqdm(all_possible_combinations)):
            pbar.set_description(f'ENSEMBLE: Processing the {wb_model_combination} combination')
            accuracy_metrics = ensemble_classifier(os.path.join(config.get('DATA_PATHS', 'base_latent_path'), 'initial_latents_and_prompts.pt'), wb_model_combination, all_wb_model_names, config.get('DATA_PATHS', 'train_log_path'))
            output_df.extend(accuracy_metrics)
        # Save the final scores
        output_df = pd.DataFrame(output_df)
        output_df.to_csv(os.path.join(config.get('DATA_PATHS', 'train_log_path'), 'ensemble', 'ensemble_accuracy.csv'))

        # meta learning classifier
        # Leave one of the classifier as the black-box and choose the rest of the classifiers as white-box
        all_possible_combinations = list(itertools.combinations(all_wb_model_names, len(all_wb_model_names)-1 ))
        output_df = [['Setting:', 'Meta'], ['','']]
        for wb_model_combination in (pbar := tqdm(all_possible_combinations)):
            pbar.set_description(f'META: Processing the {wb_model_combination} combination')
            accuracy_metrics = meta_classifier(os.path.join(config.get('DATA_PATHS', 'base_latent_path'), 'initial_latents_and_prompts.pt'), wb_model_combination, all_wb_model_names, config.get('DATA_PATHS', 'train_log_path'))
            output_df.extend(accuracy_metrics)
        # Save the final scores
        output_df = pd.DataFrame(output_df)
        output_df.to_csv(os.path.join(config.get('DATA_PATHS', 'train_log_path'), 'meta', 'meta_accuracy.csv'))

    