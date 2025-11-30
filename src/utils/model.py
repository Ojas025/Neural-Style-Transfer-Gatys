from models.vgg_nets import *

def prepare_model(model_name, device):
    if model_name == 'vgg16':
        model = VGG16()
    elif model_name == 'vgg19':
        model = VGG19(use_relu=True)
    else:
        raise Exception(f'{model_name} not supported')
    
    style_layers_dict = model.style_layers
    content_layers_dict = model.content_layer
    
    content_layer = list(content_layers_dict.keys())[0]
    style_layers = list(style_layers_dict.keys())

    content_index = list(content_layers_dict.values())[0]
    style_indices = list(style_layers_dict.values())
    
    return model.to(device).eval(), content_index, style_indices, content_layer, style_layers
        
    
def gram_matrix(feature_map, should_normalize=True):
    (b, c, h, w) = feature_map.size()
    flattened = feature_map.view(b, c, w*h)
    
    # batch matrix multiplication
    gram = torch.bmm(flattened, flattened.transpose(1,2))
    
    if should_normalize:
        gram /= (c * h * w)
    
    return gram        