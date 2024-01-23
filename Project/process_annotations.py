import copy
import json
import cv2
import numpy as np
from datetime import date

from pathlib import Path
from utils import load_txt, get_colors, get_mask_img
from manual_annotations import manually_annotated_to_delete, film_labeled_twice, repeated_masks
from utils import get_existing_masks, get_annotations, rle_to_mask


def mask_selection(annotations, colors):
    # Image dir
    folder_project = '/home/scasao/SEPARA/unizar_DL4HSI/separa/'
    images_info = annotations['images']
    partial_path_images = [img['file_name'] for img in images_info]

    # Get whole path of the rgb images
    path_images = [folder_project + p for p in partial_path_images]
    path_images, key_images = map(list, zip(*[(p, str(Path(p).stem)) for p in path_images if str(Path(p).stem) in repeated_masks]))

    annotation_images = get_annotations(annotations['annotations'],key_images)
    k = None
    masks_to_delete = {}
    print('{} images to check'.format(len(path_images)))
    for i, path_img_rgb in enumerate(path_images):
        print('{} path image to check {}'.format(i, path_img_rgb))
        existing_masks, ann_label = get_existing_masks(path_img_rgb, annotation_images, labels_included=True)
        if existing_masks is not None:
            key_img = str(Path(path_img_rgb).stem)
            masks_to_delete[key_img] = []
            next_mask = False
            print('{} mask in the image'.format(len(existing_masks)))

            image_rgb = cv2.imread(path_img_rgb)
            image_raw = copy.deepcopy(image_rgb)

            idx_mask_show = 0
            mask_bool = existing_masks[idx_mask_show]
            mask_img = get_mask_img(mask_bool, colors[ann_label[idx_mask_show]])
            image_rgb_mask = cv2.addWeighted(image_rgb, 1, mask_img, 1, 0.5)

            while not next_mask:
                cv2.imshow('rgb masks', image_rgb_mask)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
                elif k == ord('d') and idx_mask_show < len(existing_masks) - 1:
                    idx_mask_show += 1
                    mask_bool = existing_masks[idx_mask_show]
                    mask_img = get_mask_img(mask_bool, colors[ann_label[idx_mask_show]])
                    image_rgb_mask = cv2.addWeighted(image_raw, 1, mask_img, 1, 0.5)
                elif k == ord('d') and idx_mask_show == len(existing_masks) - 1:
                    print('All the masks in this image has been showed')
                elif k == ord('a') and idx_mask_show > 0:
                    idx_mask_show -= 1
                    mask_bool = existing_masks[idx_mask_show]
                    mask_img = get_mask_img(mask_bool, colors[ann_label[idx_mask_show]])
                    image_rgb_mask = cv2.addWeighted(image_raw, 1, mask_img, 1, 0.5)
                elif k == ord('a') and idx_mask_show == 0:
                    print('This is the 1º masks in this image has been showed')
                elif k == ord('r'):
                    print('Mask saved to delete')
                    mask_bool = mask_bool.tolist()
                    masks_to_delete[key_img].append(mask_bool)
                elif k == ord('g'):
                    next_mask = True

        if k == 27:
            break
    with open('double_masks_general_labels.json', 'w') as outfile:
        json.dump(masks_to_delete, outfile)


def delete_double_masks(annotations, json_info):
    ann_segmentations = annotations['annotations']
    ann_image_id = [a['image_id'] for a in ann_segmentations]

    ann_image_id_arr = np.array(ann_image_id)
    ann_segmentations_arr = np.array(ann_segmentations)
    for key_img in json_info.keys():
        ann_idx = np.where(ann_image_id_arr == key_img)
        ann = ann_segmentations_arr[ann_idx]
        current_masks = [rle_to_mask(a['segmentation'][0]) for a in ann]

        masks_to_delete = json_info[key_img]
        masks_to_delete = np.array(masks_to_delete)
        annotations_to_delete = []
        for j, mask in enumerate(masks_to_delete):
            for k, c_mask in enumerate(current_masks):
                c_mask = np.array(c_mask)
                same_mask = np.array_equal(mask, c_mask)
                if same_mask:
                    annotations_to_delete.append(ann[k])
                    break
        for ann_to_delete in annotations_to_delete:
            if ann_to_delete in ann_segmentations:
                ann_segmentations.remove(ann_to_delete)
    annotations['annotations'] = ann_segmentations
    return annotations


def add_empty_annotations(annotations, empty_images_id):
    annotation_id = len(annotations)
    for empty_img in empty_images_id:
        annotation_info = {'id': annotation_id + 1,
                           'image_id': empty_img,
                           'category_id': 0,
                           'segmentation': [],
                           'bbox': []}
        annotations.append(annotation_info)
    return annotations


def clean_annotations(json_file, empty_files_to_save_txt=None, mode='clean_fails'):
    annotations = json.load(open(json_file, 'r'))
    if 'empty' in mode and empty_files_to_save_txt is not None:
        empty_images = load_txt(empty_files_to_save_txt)
    else:
        empty_images = []

    # Clean none annotated images, distorted or corrupted images
    ann = annotations['annotations']
    if 'none' in mode:
        ann_segmentation = [a for a in ann if len(a['segmentation']) > 0 and a['image_id'] not in manually_annotated_to_delete]
    else:
        ann_segmentation = [a for a in ann if a['image_id'] not in manually_annotated_to_delete]
    ann_image_id = [a['image_id'] for a in ann_segmentation]

    image_info = annotations['images']
    image_info = [i for i in image_info if (i['id'] in ann_image_id or i['id'] in empty_images)]

    if 'empty' in mode and empty_files_to_save_txt is not None:
        # Add empty segmentations to annotations
        ann_segmentation = add_empty_annotations(ann_segmentation, empty_images)

    annotations['annotations'] = ann_segmentation
    annotations['images'] = image_info
    return annotations


def save_annotations(annotations):
    current_date = date.today()
    current_date = current_date.strftime("%Y-%m-%d_")
    name_file = current_date + 'all_annotations_rle.json'
    dir_file = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/annotations_update/' + name_file

    with open(dir_file, 'w') as outfile:
        json.dump(annotations, outfile)


if __name__ == '__main__':
    mode = 'clean_fails' # 'clean_none_and_fails' / 'add_emtpy_clean_none_and_fails'
    selected_masks, remove_double_masks = False, False
    path_json_annotations = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/annotations_update/2023-09-28_all_annotations_rle.json'
    # path_txt_empty_files  = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/empty_images2023_09_25_first_iteration.txt'

    colors = get_colors()
    # Clean image none labeled and include empty images
    annotations = clean_annotations(path_json_annotations)

    if selected_masks:
        # In case of double annotations -> mask selection
        mask_selection(annotations, colors)

    if remove_double_masks:
        # Once the better masks have been selected -> delete the double ones
        with open('double_masks_films.json') as json_file:
            double_masks_films = json.load(json_file)
        values = double_masks_films.values()
        num_double_masks = [len(v) for v in values]
        print('{} are gonna be deleted'.format(sum(num_double_masks)))
        annotations = delete_double_masks(annotations, double_masks_films)

        # Clean more masks repeated
        with open('double_masks_general_labels.json') as json_file:
            double_masks_general = json.load(json_file)
        values = double_masks_general.values()
        num_double_masks = [len(v) for v in values]
        print('{} are gonna be deleted'.format(sum(num_double_masks)))
        annotations = delete_double_masks(annotations, double_masks_general)

    save_annotations(annotations)