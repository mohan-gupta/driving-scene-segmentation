import torch
import torch.nn as nn

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from PIL import Image
from skimage import color

import numpy as np

import os
import io
import base64

def double_convs(in_channels, out_channels):
    conv_layers = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

    return conv_layers

def expansion_block(upsample_layer, conv_layer, inp, concat_inp):
    mask = upsample_layer(inp)
    mask = torch.concat([concat_inp, mask], dim=1)
    mask = conv_layer(mask)

    return mask


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #contraction path
        self.contrac1 = double_convs(in_channels, 64)
        self.contrac2 = double_convs(64, 128)
        self.contrac3 = double_convs(128, 256)
        self.contrac4 = double_convs(256, 512)
        self.contrac5 = double_convs(512, 1024)

        #expansion path
        self.upsample1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.double_conv1 = double_convs(1024, 512)
        self.upsample2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.double_conv2 = double_convs(512, 256)
        self.upsample3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.double_conv3 = double_convs(256, 128)
        self.upsample4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.double_conv4 = double_convs(128, 64)

        #output layer
        self.out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.Conv2d(64, out_channels, kernel_size=1)
            )

    def forward(self, image):
        #contraction
        cntrc_out1 = self.contrac1(image) #->
        out1 = self.max_pool(cntrc_out1)
        
        cntrc_out2 = self.contrac2(out1) #->
        out2 = self.max_pool(cntrc_out2)

        cntrc_out3 = self.contrac3(out2) #->
        out3 = self.max_pool(cntrc_out3)

        cntrc_out4 = self.contrac4(out3) #->
        out4 = self.max_pool(cntrc_out4)

        cntrc_out5 = self.contrac5(out4)

        #expansion
        mask = expansion_block(self.upsample1, self.double_conv1, cntrc_out5, cntrc_out4)
        mask = expansion_block(self.upsample2, self.double_conv2, mask, cntrc_out3)
        mask = expansion_block(self.upsample3, self.double_conv3, mask, cntrc_out2)
        mask = expansion_block(self.upsample4, self.double_conv4, mask, cntrc_out1)

        #output
        output = self.out(mask)

        return output

model = UNet(3, 22)

#Loading model
dir = os.path.dirname(__file__)
model_path = os.path.join(dir, "../model/model.pt")

checkpoint = torch.load(model_path, map_location="cpu")
model_state = checkpoint['model_state_dict']

model.load_state_dict(model_state)

class config:
    N_COLORS = 22
    COLORS = np.array([[0.658463, 0.178962, 0.372748, 1.      ],
       [0.428768, 0.09479 , 0.432412, 1.      ],
       [0.615513, 0.161817, 0.391219, 1.      ],
       [0.846709, 0.297559, 0.244113, 1.      ],
       [0.447428, 0.101597, 0.43108 , 1.      ],
       [0.50973 , 0.123769, 0.422156, 1.      ],
       [0.002267, 0.00127 , 0.01857 , 1.      ],
       [0.980824, 0.572209, 0.028508, 1.      ],
       [0.64626 , 0.173914, 0.378359, 1.      ],
       [0.964394, 0.843848, 0.273391, 1.      ],
       [0.190367, 0.039309, 0.361447, 1.      ],
       [0.949545, 0.955063, 0.50786 , 1.      ],
       [0.60933 , 0.159474, 0.393589, 1.      ],
       [0.522206, 0.12815 , 0.419549, 1.      ],
       [0.087411, 0.044556, 0.224813, 1.      ],
       [0.013995, 0.011225, 0.071862, 1.      ],
       [0.004547, 0.003392, 0.030909, 1.      ],
       [0.453651, 0.103848, 0.430498, 1.      ],
       [0.962517, 0.851476, 0.285546, 1.      ],
       [0.379001, 0.076253, 0.432719, 1.      ],
       [0.553392, 0.139134, 0.411829, 1.      ],
       [0.851384, 0.30226 , 0.239636, 1.      ]])
    
    MODEL = model


def transform_inputs(image):
    transforms = A.Compose([
                            A.Resize(256, 256),
                            ToTensorV2()
                        ])

    trnsfrmd = transforms(image=image)

    image = trnsfrmd['image']
    
    return image

def get_output(model, image):
    image = image.unsqueeze(0)
    
    logits = model(image.float())
    probs = logits.softmax(axis=1)
    probs = probs[0]
    pred = torch.argmax(probs, axis=0)
    pred *= 10
    
    return pred.numpy()

def get_results(frame):
    md_input = transform_inputs(frame)
    
    pred_mask = get_output(config.MODEL, md_input)
    
    md_input = md_input.permute(1,2,0).numpy()
    
    result_image = color.label2rgb(pred_mask, md_input, colors=config.COLORS, alpha=0.8)

    #converting from float64 to uint8
    result_image = (result_image*255).astype(np.uint8)
    
    return result_image

def encode_image(array, size):
    pil_img = Image.fromarray(array, 'RGB')
    pil_img = pil_img.resize(size)

    arr = np.array(pil_img)
    arr = arr.transpose(1,0,2)
    arr = np.fliplr(arr)
    pil_img = Image.fromarray(arr, 'RGB')
    
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_bytes = base64.b64encode(buffered.getvalue())

    img_string = img_bytes.decode()
    
    return img_string

def decode_base64(base64_str):
    img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
    
    np_arr = np.array(img)
    
    return np_arr