from segment_anything.utils import amg
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
import torch
from torchvision import transforms
from utils import mask_to_rle_pytorch, rle_to_mask
#
# convert_tensor = transforms.PILToTensor()
# path_img = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/Masks/2023-01-19/4/DALSA/Img_20230119_10-09-08_LBARQUILLA_N0000_ID1.png'
#
# # img_cv2 = cv2.imread(path_img)
# # img_tensor = torch.from_numpy(img_cv2)
#
# img = Image.open(path_img)
# img_tensor = convert_tensor(img)
#
# format_rle = mask_to_rle_pytorch(img_tensor)
# mask = rle_to_mask(format_rle[0])
# mask = 255*mask
# mask = mask.astype(np.uint8)
# cv2.imshow('mask reconstructed', mask)
# cv2.waitKey(0)
# print('p')

# COPY TO DIRECTORIES

import shutil
import json

data = json.load(open('/home/scasao/SEPARA/unizar_DL4HSI/separa/data/only_bboxes_labeled_2023-07-31_all_annotations_rle.json', 'r'))
data_dir = list(data.keys())

for d in data_dir:
    file_name = d.split('/')[-1]
    origin_dir = '/home/scasao/SEPARA/unizar_DL4HSI/separa/' + d
    target_dir = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/pending_to_label/'
    shutil.copy(origin_dir, target_dir + file_name)

