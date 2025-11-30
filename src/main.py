import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os

from utils.image import *
from utils.model import *
from utils.losses import *

def train_adam(model, image, style_layers, target_representations, num_iterations, learning_rate):
    '''
        for iteration in range(num_iterations):
            compute loss
            compute gradients
            take optimizer step
            optionally save image
    '''
    optimizer = optim.Adam((image, ), lr=learning_rate)
    loss = nn.MSELoss(reduction='mean')
    
    for iteration in range(num_iterations):
        # reset gradients
        optimizer.zero_grad() 

        # current init_image feature maps
        current_feature_maps = model(image)
        
        # compute losses
        total_loss, content_loss, style_loss, total_variation_loss = compute_loss(image, current_feature_maps, style_layers, target_representations)
        
        # compute gradients
        total_loss.backward()
        
        with torch.no_grad():
            print(f'Adam - {iteration}/{num_iterations}')
            print(f'Total Loss = {total_loss.item():.4f}')
            print(f'Content Loss = {content_loss.item():.4f}\tStyle Loss = {style_loss.item():.4f}\tTotal Variation Loss = {total_variation_loss.item():.4f}')
        
        # update parameters
        optimizer.step()
        
def train_lbfgs(model, image, style_layers, target_representations, num_iterations):
    '''
        def closure():
            clear gradients
            compute loss
            compute gradients
    
        optimizer = LBFGS()
        optimizer.step(closure)
    '''
    
    optimizer = optim.LBFGS((image, ), max_iter=num_iterations, line_search_fn='strong_wolfe')
    iteration = 1

    def closure():
        nonlocal iteration
        optimizer.zero_grad()
        
        current_feature_maps = model(image)
        
        total_loss, content_loss, style_loss, total_variation_loss = compute_loss(image, current_feature_maps, style_layers, target_representations)   
        
        total_loss.backward() 

        with torch.no_grad():
            print(f'Adam - {iteration}/{num_iterations}')
            print(f'Total Loss = {total_loss.item():.4f}')
            print(f'Content Loss = {content_loss.item():.4f}\tStyle Loss = {style_loss.item():.4f}\tTotal Variation Loss = {total_variation_loss.item():.4f}')
            
        iteration += 1
        return total_loss            
    
    
    optimizer.step(closure)

def neural_style_transfer(optimizer_name = 'lbfgs', model_name = 'vgg19', init_method='random'):
    # DEFINE PATHS
    content_image_path = './data/content-images/golden_gate.jpg'
    style_image_path = './data/style-images/vg_la_cafe.jpg'
    output_path = './data/outputs'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # LOAD IMAGES
    content_image = prepare_image(content_image_path, device)
    style_image = prepare_image(style_image_path, device)
    
    # INITIALIZE INPUT IMAGE
    init_image = get_init_image(init_method)
    
    # IMAGE TO BE OPTIMIZED 
    image = init_image.clone().detach().requires_grad_(True)

    # PREPARE MODEL
    model, content_feature_maps_index, style_feature_maps_indices, content_layer, style_layers = prepare_model(model_name, device)
    
    # GET TARGET REPRESENTATIONS
    content_feature_maps = model(content_image)
    style_feature_maps = model(style_image)
    
    # MODIFY TARGET REPRESENTATIONS
    target_content_representation = content_feature_maps[content_feature_maps_index[0]].squeeze(0)
    
    target_style_representations = [
        gram_matrix(style_feature_maps[layer])
        for layer in style_layers     
    ]
    
    target_representations = [
        target_content_representation,
        target_style_representations
    ]
    
    # INITIALIZATIONS
    num_epochs = {
        'lbfgs': 1000,
        'adam': 3000
    }
    
    learning_rate = 1e-1
    
    # OPTIMIZATION
    if optimizer_name == 'adam':
        train_adam(model, image, style_layers, target_representations, num_epochs['adam'], learning_rate)
    else:
        train_lbfgs(model, image, style_layers, target_representations, num_epochs['lbfgs'])     
    
    save_image(image, output_path)           