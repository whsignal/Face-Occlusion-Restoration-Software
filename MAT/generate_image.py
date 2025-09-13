# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""
import cv2
import pyspng
import glob
import os
import re
import random
from typing import List, Optional
import dnnlib

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image

from legacy import load_network_pkl
from networks.mat import Generator
from datasets.mask_generator_512 import RandomMask

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, nn.Module)
    assert isinstance(dst_module, nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)

def named_params_and_buffers(module):
    assert isinstance(module, nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

class ImageGenerator:
    def __init__(self, network_pkl, resolution=512, truncation_psi=1, noise_mode='const'):
        """
        Initialize the image generation module
        
        Args:
            network_pkl: Path to the network pickle file
            resolution: Resolution of the input/output images
            truncation_psi: Truncation psi value
            noise_mode: Noise mode (const, random, or none)
        """
        self.seed = 240  # Random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network_pkl = network_pkl
        self.resolution = resolution
        self.truncation_psi = truncation_psi
        self.noise_mode = noise_mode
        
        # Load the network
        self._load_network()
    
    def _load_network(self):
        """Load the pretrained network"""
        print(f'Loading networks from: {self.network_pkl}')
        with dnnlib.util.open_url(self.network_pkl) as f:
            self.G_saved = load_network_pkl(f)['G_ema'].to(self.device).eval().requires_grad_(False)
        
        net_res = 512 if self.resolution > 512 else self.resolution
        self.G = Generator(z_dim=512, c_dim=0, w_dim=512, 
                          img_resolution=net_res, img_channels=3).to(self.device).eval().requires_grad_(False)
        copy_params_and_buffers(self.G_saved, self.G, require_all=True)
        
        # Create output directory if needed
        if not os.path.exists('output'):
            os.makedirs('output')
    
    def read_image(self, image_path):
        """Read an image from a file path"""
        with open(image_path, 'rb') as f:
            if pyspng is not None and image_path.endswith('.png'):
                image = pyspng.load(f.read())
            else:
                image = np.array(Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
            image = np.repeat(image, 3, axis=2)
        image = image.transpose(2, 0, 1)  # HWC => CHW
        image = image[:3]
        return image
    
    def generate_images(self, image_paths, mask_paths=None, output_dir='output', resolution=None):
        """
        Generate images from input images
        
        Args:
            image_paths: List of input image paths
            mask_paths: List of mask paths (optional)
            output_dir: Directory to save output images
            resolution: Resolution of output images (optional)
        """
        if resolution is None:
            resolution = self.resolution
            
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # No labels
        label = torch.zeros([1, self.G.c_dim], device=self.device)
        
        # Read mask if provided
        if mask_paths is not None:
            assert len(image_paths) == len(mask_paths), 'Image and mask counts must match'
        
        if resolution != 512:
            self.noise_mode = 'random'
        
        with torch.no_grad():
            for i, ipath in enumerate(image_paths):
                iname = os.path.basename(ipath).replace('.jpg', '.png')
                print(f'Processing: {iname}')
                
                # Read and preprocess image
                image = self.read_image(ipath)
                image = (torch.from_numpy(image).float().to(self.device) / 127.5 - 1).unsqueeze(0)
                
                # Get mask
                if mask_paths is not None:
                    mask = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                    mask = torch.from_numpy(mask).float().to(self.device).unsqueeze(0).unsqueeze(0)
                else:
                    mask = RandomMask(resolution)  # Adjust masking ratio using 'hole_range'
                    mask = torch.from_numpy(mask).float().to(self.device).unsqueeze(0)
                
                # Generate latent vector
                z = torch.from_numpy(np.random.randn(1, self.G.z_dim)).to(self.device)
                
                # Generate image
                output = self.G(image, mask, z, label, truncation_psi=self.truncation_psi, noise_mode=self.noise_mode)
                output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                output = output[0].cpu().numpy()
                
                # Save output image
                output_path = os.path.join(output_dir, iname)
                Image.fromarray(output, 'RGB').save(output_path)