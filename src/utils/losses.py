import torch.nn as nn

from utils.model import *

CONTENT_WEIGHT = 1
STYLE_WEIGHT = 1e6
TOTAL_VARIATION_WEIGHT = 1e-4

def compute_content_loss(target, current):
    content_loss = nn.MSELoss(reduction='mean')(target, current)
    
    return content_loss

def compute_style_loss(target_grams, current_grams):
    style_loss = 0.0
    for gram_target, gram_current in zip(target_grams, current_grams):
        # 0th index for batch access
        layer_loss = nn.MSELoss(reduction="sum")(gram_target[0], gram_current[0])
        style_loss += layer_loss
    style_loss = style_loss / len(target_grams)
    
    return style_loss

def compute_total_variation(image, should_normalize=False):
    (b, c, h, w) = image.size()
    
    row_wise_tv = torch.pow(image[:,:,1:,:] - image[:,:,:-1,:], 2).sum()
    col_wise_tv = torch.pow(image[:,:,:,1:] - image[:,:,:,:-1], 2).sum()

    total_variation_loss = row_wise_tv + col_wise_tv
    
    if should_normalize:
        total_variation_loss /= (b*c*h*w)
    
    return total_variation_loss

def compute_loss(image, current_feature_maps, style_layers, target_representations):
    current_content = current_feature_maps['relu4_2'].squeeze(0)
    current_style_maps = [
        gram_matrix(current_feature_maps[layer])
        for layer in style_layers
    ]
    
    # compute individual losses
    content_loss = compute_content_loss(target_representations[0], current_content)
    style_loss = compute_style_loss(target_representations[1], current_style_maps)
    total_variation_loss = compute_total_variation(image)
    
    # total weighted loss
    total_loss = (CONTENT_WEIGHT * content_loss) + (STYLE_WEIGHT * style_loss) + (TOTAL_VARIATION_WEIGHT * total_variation_loss)
    
    return total_loss, content_loss, style_loss, total_variation_loss