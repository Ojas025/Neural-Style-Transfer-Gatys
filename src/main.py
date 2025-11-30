import torch
import torch.optim as optim
import torch.nn as nn
import warnings
import argparse

from utils.image import *
from utils.model import *
from utils.losses import *

warnings.filterwarnings('ignore')

def train_adam(model, image, config, style_layers, target_representations, num_iterations, learning_rate):
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
        total_loss, content_loss, style_loss, total_variation_loss = compute_loss(image, current_feature_maps, style_layers, target_representations, weights=(config['content_weight'], config['style_weight'], config['total_variation_weight']))
        
        # compute gradients
        total_loss.backward()
        
        if iteration % 100 == 0:
            with torch.no_grad():
                print(f'Adam - {iteration}/{num_iterations}')
                print(f'Total Loss = {total_loss.item():.4f}')
                print(f"Content Loss = {config['content_weight'] * content_loss.item():.4f}\tStyle Loss = {config['style_weight'] * style_loss.item():.4f}\tTotal Variation Loss = {config['total_variation_weight'] * total_variation_loss.item():.4f}\n")
        
        # update parameters
        optimizer.step()
        
def train_lbfgs(model, image, config, style_layers, target_representations, num_iterations):
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
        
        total_loss, content_loss, style_loss, total_variation_loss = compute_loss(image, current_feature_maps, style_layers, target_representations, weights=(config['content_weight'], config['style_weight'], config['total_variation_weight']))   
        
        total_loss.backward() 

        if iteration % 50 == 0:
            with torch.no_grad():
                print(f'LBFGS - {iteration}/{num_iterations}')
                print(f'Total Loss = {total_loss.item():.4f}')
                print(f"Content Loss = {config['content_weight'] * content_loss.item():.4f}\tStyle Loss = {config['style_weight'] * style_loss.item():.4f}\tTotal Variation Loss = {config['total_variation_weight'] * total_variation_loss.item():.4f}\n")
            
        iteration += 1
        return total_loss            
    
    
    optimizer.step(closure)

def neural_style_transfer(config):
    # DEFINE PATHS
    content_image_path = f'./data/content-images/{config["content_image"]}'
    style_image_path = f'./data/style-images/{config["style_image"]}'
    output_path = f'{config["content_image"].split('.')[0]}_{config["style_image"].split('.')[0]}.jpg'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # LOAD IMAGES
    content_image = prepare_image(content_image_path, device, (416,512))
    style_image = prepare_image(style_image_path, device, (416,512))
    
    # print(f"Content Image Size: ({h},{w})")
    # print(f"Style Image Size: ({h},{w})")

    # INITIALIZE INPUT IMAGE
    init_image = get_init_image(config['init_method'], content_image, style_image, (416,512), device)
    # print(f"Init Image Size: ({h},{w})")/
    
    # IMAGE TO BE OPTIMIZED 
    image = init_image.clone().detach().requires_grad_(True)

    # PREPARE MODEL
    model, content_feature_maps_index, style_feature_maps_indices, content_layer, style_layers = prepare_model(config['model'], device)
    
    # GET TARGET REPRESENTATIONS
    content_feature_maps = model(content_image)
    style_feature_maps = model(style_image)
    
    # MODIFY TARGET REPRESENTATIONS
    target_content_representation = content_feature_maps[content_layer].squeeze(0)
    
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
    if config['optimizer'] == 'adam':
        train_adam(model, image, config, style_layers, target_representations, num_epochs['adam'], learning_rate)
    else:
        train_lbfgs(model, image, config, style_layers, target_representations, num_epochs['lbfgs'])     
    
    save_image(image, output_path)           
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_image", type=str, help="Content Image Name", default="golden_gate.jpg")
    parser.add_argument("--style_image", type=str, help="Style Image Name", default="vg_la_cafe.jpg")
    
    parser.add_argument("--content_weight", type=float, help="Weight factor for content loss", default=1)
    parser.add_argument("--style_weight", type=float, help="Weight factor for style loss", default=1e6)
    parser.add_argument("--total_variation_weight", type=float, help="Weight factor for total variation loss", default=1e-6)
    
    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19'], default='vgg19')
    parser.add_argument('--init_method', type=str, choices=['content', 'style', 'random'], default='random')
    
    args = parser.parse_args()
    
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    
    neural_style_transfer(config)
    
main()        