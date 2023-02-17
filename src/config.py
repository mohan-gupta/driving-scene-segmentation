import torch

LR = 1e-3
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (256, 256)
INP_CHANNELS = 3
NUM_CLASSES = 22
PIN_MEMORY = True
EPOCH = 3
MODEL_PATH = "../model/model.pt"  #"../model/model_74sgd.pt"