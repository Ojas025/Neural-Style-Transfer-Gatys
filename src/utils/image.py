from torchvision import transforms
import torch
from PIL import Image
import os

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def load_image(path):
    if not os.path.exists(path):
        raise Exception(f'Path does not exist: {path}')
    
    image = Image.open(path)
    image = image.convert('RGB')
    
    return image

def resize_image(image, size):
    # current dimensions
    w, h = image.size
    
    # resize while keeping the aspect ratio
    scale = size / max(h,w)
    
    nw, nh = int(w * scale), int(h * scale)
    
    image = image.resize((nw, nh), Image.LANCZOS)
    
    return image, nh, nw

def prepare_image(path, device, size):
    image = load_image(path)
    
    # image, h, w = resize_image(image, size)
    
    image = image.resize((size[1],size[0]), Image.LANCZOS)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
    ])
    
    image = transform(image)
    image = image.to(device)
    image = image.unsqueeze(0)
    
    return image

def save_image(image, path):
    # remove batch dimension
    image = image.squeeze(0)
    
    image = image.detach().cpu().float()
    
    std = torch.tensor(IMAGENET_STD).view(3,1,1)
    mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
    
    image = image * std + mean
    
    image = torch.clamp(image, 0, 1)
    
    # [C,H,W] -> [H,W,C]
    # print("image size: ", image.size())
    
    # image = image.permute(1, 2, 0).contiguous()
    
    # print("image size post permute: ", image.size())
    # print(image.size())
    
    # image = (image * 255.0).byte()

    image = transforms.ToPILImage()(image)
    
    os.makedirs('./data/output', exist_ok=True)
    image.save(os.path.join('./data/output', path))

def get_init_image(init_method, content_image, style_image, size, device):
    if init_method == 'random':
        image = torch.randn([1, 3, size[0], size[1]], device=device, requires_grad=True, dtype=torch.float32) * 0.5
    elif init_method == 'content':
        image = content_image.clone().detach().requires_grad_(True)
    else:
        image = style_image.clone().detach().requires_grad_(True)
    
    return image