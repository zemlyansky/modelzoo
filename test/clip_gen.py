import json
import os

import numpy as np
import torch
from torch import nn
from PIL import Image

CWD = os.path.dirname(os.path.abspath(__file__))
CLIP_PATH = os.path.join(CWD, '../py/clip')
DATA_PATH = os.path.join(CWD, 'clip_data.json')

import sys
sys.path.insert(0, CLIP_PATH)

from clip.simple_tokenizer import SimpleTokenizer
from clip.clip import tokenize, _transform
from clip.model import ResidualAttentionBlock, VisionTransformer

def get_state_dict(model):
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = v.tolist()
    return state_dict

test_data = {}
test_data['cases'] = [
    'hello world',
    'lorem ipsum dolor sit amet',
    'nikola tesla in new york'
]

# Test tokenizer
tokenizer = SimpleTokenizer()
test_data['tokenizer'] = [tokenize(case).numpy().tolist() for case in test_data['cases']]

# mask = torch.empty(4, 4)
# mask.fill_(float("-inf"))
# mask.triu_(1)  # zero out the lower diagonal

# Test attention block
test_data['attn_config'] = {
    'd_model': 8,
    'n_head': 2,
}
test_data['attn_input_shape'] = [1, 4, test_data['attn_config']['d_model']] # [batch, channels, dim]
attn = ResidualAttentionBlock(**test_data['attn_config'])
attn_input = torch.randn(*test_data['attn_input_shape'])
test_data['attn_input'] = attn_input.tolist()
attn_input = attn_input.permute(1, 0, 2)
attn_output = attn(attn_input)
attn_output = attn_output.permute(1, 0, 2)
test_data['attn_output'] = attn_output.tolist()
test_data['attn_weights'] = get_state_dict(attn)
print('Attn output shapes:', attn_output.shape)

# Test VisionTransformer
# VisionTransformer expects specific input shape (batch, channels, input_resolution, input_resolution)
# _transform from clip.py converts regular image shape to this shape
# For example: (1000, 1000, 3) -> (1, 3, 32, 32)
test_data['vit_config'] = {
    'input_resolution': 32,
    'patch_size': 2,
    'width': 4,
    'layers': 2,
    'heads': 2,
    'output_dim': 8
}
vit = VisionTransformer(**test_data['vit_config'])
test_data['vit_input_shape'] = [1, 3, test_data['vit_config']['input_resolution'], test_data['vit_config']['input_resolution']]
vit_input = torch.randn(*test_data['vit_input_shape'])
vit_output = vit(vit_input)
test_data['vit_input'] = vit_input.permute(0, 2, 3, 1).tolist()
test_data['vit_output'] = vit_output.tolist()
test_data['vit_weights'] = get_state_dict(vit)
print('Vit output shape:', vit_output.shape)

# Test image transform
test_data['transform_resolution'] = 32
# Generate random image [128, 64, 3] with values in range [0, 255]
# tr_input = np.random.randint(0, 255, size=(128, 64, 3), dtype=np.uint8)
# Generate grey square in the middle of white image
tr_input = np.ones((64, 128, 3), dtype=np.uint8) * 255
tr_input[16:48, 48:80] = 100

transform = _transform(n_px=test_data['transform_resolution'])
tr_input_img = Image.fromarray(tr_input, mode='RGB')
tr_input_img.save(os.path.join(CWD, '../temp/clip_input.png'))
tr_output = transform(tr_input_img).permute(1, 2, 0)
test_data['transform_input'] = tr_input.tolist()
test_data['transform_output'] = tr_output.tolist()

# Print weights with shapes
# for k, v in vit.state_dict().items():
    # print(k, v.shape)

with open(DATA_PATH, 'w') as f:
    json.dump(test_data, f)

print('Generated test data at', DATA_PATH)