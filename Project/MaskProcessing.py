import os
import cv2
import re
import json
import numpy as np
import shutil
from PIL import Image
from pathlib import Path
from utils import remove_small_regions
from torchvision import transforms
from utils import mask_to_rle_pytorch, rle_to_mask
from CocoFormat import CocoFormatSEPARA
from datetime import date
from collections import Counter

convert_tensor = transforms.PILToTensor()

SEPARA_LABELS = {
     0: "FONDO",
     1: "FILM",
     2: "BARQUILLA",
     3: "CARTON",
     4: "CINTA_VIDEO",
     5: "ELEMENTOS_FILIFORMES",
     6: "BOLSA",
     7: "ELECTRONICA"
}


def get_name_mask_parameters(dir_mask):
    mask_img_id  = re.search('Img_(.*)_L', dir_mask).group(1)
    label_ann    = re.search('_L(.*)_N', dir_mask).group(1)
    str_idx_mask = re.search('_N(.*).png', dir_mask).group(1)

    return mask_img_id, str_idx_mask, label_ann


def process_mask(d_mask):
    mask_not_to_save = ['20230119_10-09-08', '20230119_10-17-25','20230117_11-43-53']
    image_id = re.search('Img_(.*)_L', d_mask).group(1)
    if image_id not in mask_not_to_save:
        mask = cv2.imread(d_mask, 0)
        mask_img_id, str_idx_mask, label_ann = get_name_mask_parameters(d_mask)
        label_ann = SEPARA_LABELS[int(label_ann)]

        mask_processed = remove_small_regions(mask, label_ann)
        # img_to_show = cv2.hconcat([mask, mask_processed])

        folder_file, name_file = os.path.split(d_mask)
        dir_mask_processed = folder_file.replace('Masks_raw_unify', 'Masks_unify_processed')
        # dir_mask_processed = dir_mask_processed + '/2023-06-09'
        if not os.path.exists(dir_mask_processed):
            os.makedirs(dir_mask_processed)

        name_file = 'Img_' + mask_img_id + '_L' + label_ann + '_N' + str_idx_mask
        final_file_path = dir_mask_processed + '/' + name_file + '.png'
        cv2.imwrite(final_file_path, mask_processed)
    else:
        print('excluded')


def process_mask_and_order(d_mask, annotations, save = True):
    mask_img_id  = re.search('Img_(.*)_L', d_mask).group(1)
    id_ann       = re.search('_ID(.*).png', d_mask).group(1)
    str_idx_mask = re.search('_N(.*)_ID', d_mask).group(1)
    label_ann    = re.search('_L(.*)_N', d_mask).group(1)
    label_ann = SEPARA_LABELS[int(label_ann)]

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


def load_txt(path_file):
    with open(path_file) as f:
        lines = [line.rstrip() for line in f]
    return lines


def overlap_vhs_class(dir_masks_files, name_new_folder, label='0004'):
    dir_masks_processed = []
    id_image_masks = [re.search('Img_(.*)_L', f).group(1) for f in dir_masks_files]
    class_masks = [re.search('_L(.*)_N', f).group(1) for f in dir_masks_files]

    id_image_masks_array = np.array(id_image_masks)
    class_masks_array = np.array(class_masks)

    idx_vhs_class = np.where(class_masks_array == label)
    id_image_masks_array_vhs = id_image_masks_array[idx_vhs_class]

    images_repeated = [k for k, v in Counter(id_image_masks_array_vhs).items() if v > 1]
    idx_vhs_class_array = np.array(idx_vhs_class[0])
    dir_masks_files_array = np.array(dir_masks_files)
    for img_id in images_repeated:
        idx_masks_vhs = np.where(id_image_masks_array_vhs == img_id)[0]
        idx_img_masks = idx_vhs_class_array[idx_masks_vhs]

        dir_img_masks = dir_masks_files_array[idx_img_masks]
        masks = []
        for d_mask in dir_img_masks:
            # Check mask correspond to the correct image id and class
            assert re.search('Img_(.*)_L', d_mask).group(1) == img_id and re.search('_L(.*)_N', d_mask).group(1) == label

            mask_img = cv2.imread(d_mask)
            masks.append(mask_img)
            dir_masks_processed.append(d_mask)

        size_mask = np.shape(masks[0])
        ref_mask = False*np.ones(size_mask, dtype=np.uint8)
        for mask in masks:
            ref_mask = np.logical_or(ref_mask, mask)
        ref_mask = ref_mask.astype(np.uint8) * 255

        d_mask = dir_img_masks[0]
        mask_img_id, str_idx_mask, label_ann = get_name_mask_parameters(d_mask)

        folder_file, name_file = os.path.split(d_mask)
        new_dir_mask = folder_file.replace('Masks_raw', name_new_folder)
        if not os.path.exists(new_dir_mask):
            os.makedirs(new_dir_mask)

        name_file = 'Img_' + mask_img_id + '_L' + label_ann + '_N' + str_idx_mask
        final_file_path = new_dir_mask + '/' + name_file + '.png'
        cv2.imwrite(final_file_path, ref_mask)
    return dir_masks_processed


if __name__ == '__main__':
    unify_vhs_masks = False

    dir_masks = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/annotations_25_10/Masks_raw_unify'
    dir_masks_files = list(Path(dir_masks).rglob("*.png"))  # check extension of the masks
    dir_masks_files = [str(d) for d in dir_masks_files]

    if unify_vhs_masks:
        dir_masks_img = []
        name_new_folder = 'Masks_raw_unify'
        dir_masks_processed = overlap_vhs_class(dir_masks_files, name_new_folder)
        dir_masks_files = set(dir_masks_files).symmetric_difference(set(dir_masks_processed))
        dir_masks_files = list(dir_masks_files)
        for d_mask in dir_masks_files:
            assert d_mask not in dir_masks_processed
            folder_file, name_file = os.path.split(d_mask)
            new_dir_mask = folder_file.replace('Masks_raw', name_new_folder)
            if not os.path.exists(new_dir_mask):
                os.makedirs(new_dir_mask)
            new_file_path = new_dir_mask + '/' + name_file
            shutil.copy(d_mask, new_file_path)
    else:
        # When the process has been correctly perform
        # distorted_images = load_txt('/home/scasao/SEPARA/unizar_DL4HSI/separa/data/annotations_14_09/distorted_images_2023_09_14.txt')
        distorted_images = []
        print('{} image to process'.format(len(dir_masks_files)))

        for i, d_mask in enumerate(dir_masks_files):
            image_id  = re.search('Img_(.*)_L', d_mask).group(1)
            if i%50 == 0:
                print('Image {}'.format(i))
            if image_id not in distorted_images:
                process_mask(d_mask)



#####################################################

# data = json.load(open('/home/scasao/SEPARA/unizar_DL4HSI/separa/data/annotations_update/2023-07-28_all_annotations_rle.json', 'r'))
# dir_masks = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/Masks_SAM_points'
#

# CHECK RGB EXISTS IN HYPER
# dir_hyper = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/disk/'

# dir_hyper_files = list(Path(dir_hyper).rglob("*.tiff"))

# dir_hyper_files = [str(d) for d in dir_hyper_files]
# images_id_masks = {re.search('Img_(.*)_L', d).group(1): d for d in dir_masks_files}
# images_id_hyper = [re.search('Specim/(.*).tiff', d).group(1) for d in dir_hyper_files]
#
# image_to_annotate = list(set(images_id_masks.keys()) & set(images_id_hyper))

# dir_masks_files = list(Path(dir_masks).rglob("*.png"))  # check extension of the masks
# dir_masks_files = [str(d) for d in dir_masks_files]
# for d_mask in dir_masks_files:
#     mask_processed = process_mask_and_order(d_mask, data)

