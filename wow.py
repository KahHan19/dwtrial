import cv2
import numpy as np
from PIL import Image
import torch

from utils import *
from dwpose import DWposeDetector

# set configs
det_config = './dwpose/yolox_config/yolox_l_8xb8-300e_coco.py'
det_ckpt = './ckpts/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
pose_config = './dwpose/dwpose_config/dwpose-l_384x288.py'
pose_ckpt = './ckpts/dw-ll_ucoco_384.pth'

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# init
dwprocessor = DWposeDetector(det_config, None, pose_config, None, device)

# infer
image_dir = "./assets/test.jpeg"
input_image = cv2.imread(image_dir)
input_image = HWC3(input_image)
input_image = resize_image(input_image, resolution=512)
H, W, C = input_image.shape

detected_map = dwprocessor(input_image)
detected_map = HWC3(detected_map)

detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
cv2.imwrite(image_dir.split('/')[-1], detected_map)