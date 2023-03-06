import torch

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from PIL import Image
from skimage import color
import cv2
import moviepy.editor as moviepy

import numpy as np

import os
import io
import base64
import tempfile

from model import UNet

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
    
    #Loading model
    dir = os.path.dirname(__file__)
    model_path = os.path.join(dir, "../model/model.pt")

    checkpoint = torch.load(model_path, map_location="cpu")
    model_state = checkpoint['model_state_dict']
    
    MODEL = UNet(3, 22)
    MODEL.load_state_dict(model_state)


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
    
    result_image = color.label2rgb(pred_mask, md_input, colors=config.COLORS, alpha=0.6)

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

def predict_video(uploaded_file):
    output_frames = []
    dir_path = os.path.dirname(__file__)

    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())

    vf = cv2.VideoCapture(tfile.name)

    while vf.isOpened():
        ret, frame = vf.read()  # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        pred = get_results(frame)

        masked_image = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

        output_frames.append(masked_image)

    output_file = tempfile.NamedTemporaryFile(suffix='.avi', delete=False, dir=dir_path)

    height, width, _ = output_frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(output_file.name, fourcc, 7, (width, height))

    for frame in output_frames:
        video.write(frame)

    video.release()

    clip = moviepy.VideoFileClip(output_file.name)
    mp4_file = output_file.name[:-3]+"mp4"
    print(mp4_file)

    clip.write_videofile(mp4_file)
    
    return mp4_file