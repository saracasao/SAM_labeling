import os
import cv2
import re
import json
import numpy as np
from PIL import Image
from pathlib import Path
from utils import remove_small_regions
from torchvision import transforms
from utils import mask_to_rle_pytorch, rle_to_mask
from CocoFormat import CocoFormatSEPARA
from datetime import date

convert_tensor = transforms.PILToTensor()

SEPARA_LABELS = {
     0: "FONDO",
     1: "FILM",
     2: "BARQUILLA",
     3: "CARTON",
     4: "CINTA_VIDEO",
     5: "ELEMENTOS_FILIFORMES"
}

def process_mask(d_mask):
    mask = cv2.imread(d_mask, 0)
    mask_img_id  = re.search('Img_(.*)_L', d_mask).group(1)
    str_idx_mask = re.search('_N(.*).png', d_mask).group(1)
    label_ann    = int(re.search('_L(.*)_N', d_mask).group(1))
    label_ann = SEPARA_LABELS[label_ann]

    # label_ann = re.search('_L(.*)_N', d_mask).group(1)
    # label_ann = SEPARA_LABELS[int(label_ann)]

    mask_processed = remove_small_regions(mask, label_ann)
    folder_file, name_file = os.path.split(d_mask)
    dir_mask_processed = folder_file.replace('Masks_Bilbao', 'Masks_processed')
    dir_mask_processed = dir_mask_processed + '/2023-06-09'
    if not os.path.exists(dir_mask_processed):
        os.makedirs(dir_mask_processed)

    name_file = 'Img_' + mask_img_id + '_L' + label_ann + '_N' + str_idx_mask
    final_file_path = dir_mask_processed + '/' + name_file + '.png'
    cv2.imwrite(final_file_path, mask_processed)


def process_mask_and_order(d_mask, annotations, save = True):
    mask_img_id  = re.search('Img_(.*)_L', d_mask).group(1)
    id_ann       = re.search('_ID(.*).png', d_mask).group(1)
    str_idx_mask = re.search('_N(.*)_ID', d_mask).group(1)
    label_ann    = int(re.search('_L(.*)_N', d_mask).group(1))
    label_ann = SEPARA_LABELS[label_ann]

    ann = annotations['images']
    image_dir = [a['file_name'] for a in ann if a['id'] == mask_img_id][0]
    image_dir = image_dir.split('/')[2:5]

    subfolders_mask = os.path.join(*image_dir)
    name_file = 'Img_' + mask_img_id + '_L' + label_ann + '_N' + str_idx_mask + '_ID' + str(id_ann)

    mask = cv2.imread(d_mask, 0)
    mask_processed = remove_small_regions(mask, label_ann)
    if save:
        dir_mask_processed = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/Masks_SAM_points_order/' + subfolders_mask
        if not os.path.exists(dir_mask_processed):
            os.makedirs(dir_mask_processed)

        final_file_path = dir_mask_processed + '/' + name_file + '.png'
        cv2.imwrite(final_file_path, mask_processed)

#
# data = json.load(open('/home/scasao/SEPARA/unizar_DL4HSI/separa/data/annotations_update/2023-07-28_all_annotations_rle.json', 'r'))
# dir_masks = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/Masks_SAM_points'
#
# dir_masks_files = list(Path(dir_masks).rglob("*.png"))  # check extension of the masks
# dir_masks_files = [str(d) for d in dir_masks_files]
# for d_mask in dir_masks_files:
#     mask_processed = process_mask_and_order(d_mask, data)


# When the process has been correctly perform
dir_masks = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/Masks_Bilbao'
dir_hyper = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/disk/2023-06-09'

dir_hyper_files = list(Path(dir_hyper).rglob("*.tiff"))
dir_masks_files = list(Path(dir_masks).rglob("*.png"))  # check extension of the masks

dir_masks_files = [str(d) for d in dir_masks_files]
# dir_hyper_files = [str(d) for d in dir_hyper_files]
# images_id_masks = {re.search('Img_(.*)_L', d).group(1): d for d in dir_masks_files}
# images_id_hyper = [re.search('Specim/(.*).tiff', d).group(1) for d in dir_hyper_files]
#
# image_to_annotate = list(set(images_id_masks.keys()) & set(images_id_hyper))
for d_mask in dir_masks_files:
    process_mask(d_mask)
