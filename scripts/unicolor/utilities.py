import hashlib
import os
import sys

import cv2
import numpy as np
import requests
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
import yaml
from PIL import Image
from skimage import color
from torch.autograd import Variable
from tqdm import tqdm

l_norm, ab_norm = 1.0, 1.0
l_mean, ab_mean = 50.0, 0

# denormalization for l
def uncenter_l(l):
    return l * l_norm + l_mean

def gray2rgb_batch(l):
    # gray image tensor to rgb image tensor
    l_uncenter = uncenter_l(l)
    l_uncenter = l_uncenter / (2 * l_mean)
    return torch.cat((l_uncenter, l_uncenter, l_uncenter), dim=1)

###### vgg preprocessing ######
def vgg_preprocess(tensor):
    # input is RGB tensor which ranges in [0,1]
    # output is BGR tensor which ranges in [0,255]
    tensor_bgr = torch.cat((tensor[:, 2:3, :, :], tensor[:, 1:2, :, :], tensor[:, 0:1, :, :]), dim=1)
    tensor_bgr_ml = tensor_bgr - torch.Tensor([0.40760392, 0.45795686, 0.48501961]).type_as(tensor_bgr).view(1, 3, 1, 1)
    return tensor_bgr_ml * 255

def batch_lab2rgb_transpose_mc(img_l_mc, img_ab_mc, nrow=8):
    if isinstance(img_l_mc, Variable):
        img_l_mc = img_l_mc.data.cpu()
    if isinstance(img_ab_mc, Variable):
        img_ab_mc = img_ab_mc.data.cpu()

    if img_l_mc.is_cuda:
        img_l_mc = img_l_mc.cpu()
    if img_ab_mc.is_cuda:
        img_ab_mc = img_ab_mc.cpu()

    assert img_l_mc.dim() == 4 and img_ab_mc.dim() == 4, "only for batch input"

    img_l = img_l_mc * l_norm + l_mean
    img_ab = img_ab_mc * ab_norm + ab_mean
    pred_lab = torch.cat((img_l, img_ab), dim=1)
    grid_lab = vutils.make_grid(pred_lab, nrow=nrow).numpy().astype("float64")
    return (np.clip(color.lab2rgb(grid_lab.transpose((1, 2, 0))), 0, 1) * 255).astype("uint8")

def output_to_pil(x):
    x = x.detach()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.

    # Changed so this works on M1 Macs
    x = x.cpu()

    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    if x.shape[2] == 1:
        x = x[:, :, 0]
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def preprocess(img, size):
    if size == None:
        size = [(img.size[1] // 16) * 16, (img.size[0] // 16) * 16]
    transform = T.Compose([T.Resize(size), T.ToTensor()])
    x = transform(img)
    x = x * 2 - 1
    x = x.unsqueeze(0)
    return x

def replace(words, ori, rep):
    for i in range(len(words)):
        if words[i] == ori:
            words[i] = rep
    return words

def find(path, name):
    for root, dirs, files in os.walk(path):
        for f in files:
            if name in f:
                return os.path.join(root, f)

def load_model(model, dir, step):
    print(dir)
    # Load config
    config_path = os.path.join(dir, 'config.yaml')
    with open(config_path, 'rb') as fin:
        config = yaml.safe_load(fin)
    model_config = config['model']
    model_config['learning_rate'] = 0.0
    # Load model
    loaded = model.load_from_checkpoint(
        find(dir, step+'.ckpt'),
        **model_config,
        load_vqgan_from_separate_file=False,
        strict=True
    )
    return loaded

def rgb_to_gray(x):
    return 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()

def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path

def draw_color(l, color, rect):
    y0, y1, x0, x1 = rect
    l = np.array(l.convert('RGB'))
    lab = cv2.cvtColor(l, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0:1]
    ab = lab[:, :, 1:3]
    draw = np.array(color).astype(np.uint8)
    if len(draw.shape) == 1:
        draw = np.expand_dims(draw, axis=[0, 1])
    draw = cv2.cvtColor(draw, cv2.COLOR_RGB2LAB)
    ab[y0:y1, x0:x1, :] = draw[:, :, 1:3]
    lab = np.concatenate([l, ab], axis=2)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img)

def color_resize(l, color):
    color = color.resize(l.size)
    resized = draw_color(l.convert('RGB'), color, [None, None, None, None])
    return resized

def get_sample_range(length, sample_size):
    num = int( np.ceil( float(length) / float(sample_size) ) )
    start = sample_size // 2
    stop = start + (num - 1) * sample_size
    steps = np.linspace(start=start, stop=stop, num=num).astype(int)
    return list(steps)

def get_input_range(rows, cols, r, c, sample_shape):
    # Index range for input
    c0 = c - sample_shape[1] // 2
    c1 = c0 + sample_shape[1]
    if c0 < 0:
        c0 = 0
        c1 = c0 + sample_shape[1]
    if c1 > cols:
        c1 = cols
        c0 = c1 - sample_shape[1]
    r0 = r - sample_shape[0] // 2
    r1 = r0 + sample_shape[0]
    if r0 < 0:
        r0 = 0
        r1 = r0 + sample_shape[0]
    if r1 > rows:
        r1 = rows
        r0 = r1 - sample_shape[0]
    
    return int(r0), int(c0), int(r1), int(c1)

def get_mask_range(rows, cols, r, c, mask_size, input_range):
    # Index range for mask
    r0 = r - mask_size[0] // 2;  r1 = r0 + mask_size[0]
    c0 = c - mask_size[1] // 2;  c1 = c0 + mask_size[1]
    r0 = max(0, r0);  r1 = min(rows, r1)
    c0 = max(0, c0);  c1 = min(cols, c1)
    r_start = input_range[0];  c_start = input_range[1]
    return int(r0 - r_start), int(c0 - c_start), int(r1 - r_start), int(c1 - c_start)

def get_predict_range(rows, cols, r, c, patch_size, input_range):
    # Index range for predict patch
    r0 = r - patch_size[0] // 2;  r1 = r0 + patch_size[0]
    c0 = c - patch_size[1] // 2;  c1 = c0 + patch_size[1]
    r0 = max(0, r0);  r1 = min(rows, r1)
    c0 = max(0, c0);  c1 = min(cols, c1)

    pos = []
    for row in range(r0, r1):
        for col in range(c0, c1):
            nrow = row - input_range[0]
            ncol = col - input_range[1]
            width = input_range[3] - input_range[1]
            pos.append( int(nrow * width + ncol) )
    
    return int(r0), int(c0), int(r1), int(c1), pos

def draw_full_color(l, color, rect):
    y0, y1, x0, x1 = rect
    img = np.array(l.convert('RGB'))
    draw = np.array(color).astype(np.uint8)
    if len(draw.shape) == 1:
        draw = np.expand_dims(draw, axis=[0, 1])
    img[y0:y1, x0:x1, :] = draw[:, :, :]
    return Image.fromarray(img)


def draw_strokes(image, img_size, strokes):
    org_size = image.size[::-1]
    # Draw strokes
    draw_img = image.copy().convert('RGB')
    for stk in strokes:
        ind = stk['index']
        ind = [int(ind[0] / img_size[0] * org_size[0]), int(ind[1] / img_size[1] * org_size[1])]
        patch_size = [int(org_size[0]/img_size[0]*10), int(org_size[1]/img_size[1]*10)]
        border_size = [int(org_size[0]/img_size[0]*3), int(org_size[1]/img_size[1]*3)]
        color = np.zeros(patch_size+[3])
        color[:, :] = stk['color']
        color = color.astype(np.uint8)
        color = cv2.copyMakeBorder(color, border_size[0], border_size[0], border_size[1], border_size[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        draw_img = draw_full_color(draw_img, color, [ind[0], ind[0]+color.shape[0], ind[1], ind[1]+color.shape[1]])
    return draw_img.resize(org_size[::-1])

##### color space
xyz_from_rgb = np.array(
    [[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]]
)
rgb_from_xyz = np.array(
    [[3.24048134, -0.96925495, 0.05564664], [-1.53715152, 1.87599, -0.20404134], [-0.49853633, 0.04155593, 1.05731107]]
)

def tensor_lab2rgb(input):
    """
    n * 3* h *w
    """
    input_trans = input.transpose(1, 2).transpose(2, 3)  # n * h * w * 3
    L, a, b = input_trans[:, :, :, 0:1], input_trans[:, :, :, 1:2], input_trans[:, :, :, 2:]
    y = (L + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)

    neg_mask = z.data < 0
    z[neg_mask] = 0
    xyz = torch.cat((x, y, z), dim=3)

    mask = xyz.data > 0.2068966
    mask_xyz = xyz.clone()
    mask_xyz[mask] = torch.pow(xyz[mask], 3.0)
    mask_xyz[~mask] = (xyz[~mask] - 16.0 / 116.0) / 7.787
    mask_xyz[:, :, :, 0] = mask_xyz[:, :, :, 0] * 0.95047
    mask_xyz[:, :, :, 2] = mask_xyz[:, :, :, 2] * 1.08883

    rgb_trans = torch.mm(mask_xyz.view(-1, 3), torch.from_numpy(rgb_from_xyz).type_as(xyz)).view(
        input.size(0), input.size(2), input.size(3), 3
    )
    rgb = rgb_trans.transpose(2, 3).transpose(1, 2)

    mask = rgb > 0.0031308
    mask_rgb = rgb.clone()
    mask_rgb[mask] = 1.055 * torch.pow(rgb[mask], 1 / 2.4) - 0.055
    mask_rgb[~mask] = rgb[~mask] * 12.92

    neg_mask = mask_rgb.data < 0
    large_mask = mask_rgb.data > 1
    mask_rgb[neg_mask] = 0
    mask_rgb[large_mask] = 1
    return mask_rgb

###### loss functions ######
def feature_normalize(feature_in):
    feature_in_norm = torch.norm(feature_in, 2, 1, keepdim=True) + sys.float_info.epsilon
    feature_in_norm = torch.div(feature_in, feature_in_norm)
    return feature_in_norm