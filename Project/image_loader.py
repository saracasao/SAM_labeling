import re
import os
import json
from pathlib import Path
import numpy as np
from utils import load_txt
from manual_annotations import some_classes_none_labeled


def get_path_hyperspectral(rgb_images_path):
    path_hyperspectral = []
    for rgb_path in rgb_images_path:
        folder_file, name_file = os.path.split(rgb_path)
        folder_mask_processed = folder_file.replace('DALSA', 'Specim')
        dir_mask_processed = folder_mask_processed + '/' + str(Path(name_file).stem) + '.tiff'
        path_hyperspectral.append(dir_mask_processed)
    return path_hyperspectral


def get_masks_files(masks_path):
    names = []
    for path, subdirs, files in os.walk(masks_path):
        for file in files:
            if (file.endswith('.png')):
                names.append(file)
    return names


def clean_already_annotated(masks_path, path_images, all_name_files):
    try:
        # name_files = os.listdir(masks_path)
        name_files = get_masks_files(masks_path)
        masks_id = [re.search('Img_(.*)_L', n_mask).group(1) for n_mask in name_files]
        image_id = [re.search('(.*).jpg', n_img).group(1) for n_img in all_name_files]

        image_to_label = list(set(image_id) ^ set(masks_id))

        name_image_to_label = [all_name_files[i].split('.')[0] for i, name in enumerate(image_id) if name in image_to_label]
        path_images_to_label = [p for p in path_images if str(Path(p).stem) in name_image_to_label]
    except:
        path_images_to_label = path_images

        image_to_label = [img.split('.')[0] for img in all_name_files]
        name_image_to_label = image_to_label
    return path_images_to_label, name_image_to_label


def get_annotations(annotations,name_image_to_label, get_labels=True):
    ann = {name_img: [] for name_img in name_image_to_label}

    ann_image_ids = [ann['image_id'] for ann in annotations]
    ann_image_ids = np.array(ann_image_ids)
    for name in name_image_to_label:
        if name in ann_image_ids:
            idx_ann = np.where(name == ann_image_ids)[0]
            for idx in idx_ann:
                img_ann = annotations[idx]['segmentation']
                if get_labels:
                    label_ann = annotations[idx]['category_id']
                    info_ann = tuple((img_ann, label_ann))
                    ann[name].append(info_ann)
                else:
                    ann[name].append(img_ann)
    return ann


def loader_separa_data(path_to_save_masks, path_file_annotations, clean_files_annotated=True, only_labeled=False, path_processed_files=None, selected_keys=None, add_manual_labeled=False, include_labels=False):
    annotations = json.load(open(path_file_annotations, 'r'))

    images_info = annotations['images']
    partial_path_images = [img['file_name'] for img in images_info]

    folder_project = '/home/scasao/SEPARA/unizar_DL4HSI/separa/'
    path_images = [folder_project + p for p in partial_path_images]
    name_files = [p.split('/')[-1] for p in partial_path_images]

    # If clean images that has a corresponding mask in the 'path_to_save_masks' directory ~ already labeled
    if clean_files_annotated:
        path_images, name_image_to_label = clean_already_annotated(path_to_save_masks, path_images, name_files)
    else:
        name_image_to_label = [img.split('.')[0] for img in name_files]

    # If we want to revise some images
    if selected_keys is not None:
        name_image_to_label = load_txt(selected_keys)
        name_image_to_label = [n[:17] for n in name_image_to_label]
        if add_manual_labeled:
            name_image_to_label = name_image_to_label + some_classes_none_labeled
        name_image_to_label = list(np.unique(name_image_to_label))
        path_images = [p for p in path_images if str(Path(p).stem) in name_image_to_label]
        assert len(path_images) == len(name_image_to_label)

    # Get dictionary of images with their corresponding masks
    annotations_image_to_label = get_annotations(annotations['annotations'],name_image_to_label, include_labels)

    # Load only labeled images
    if only_labeled:
        annotations_non_empty = {k: v for k, v in annotations_image_to_label.items() if len(v) > 0}
        path_images_non_empty = [p for p in path_images if Path(p).stem in annotations_non_empty.keys()]
        annotations_image_to_label = annotations_non_empty
        path_images = path_images_non_empty
    # Clean images already processed
    if path_processed_files is not None:
        keys_processed = load_txt(path_processed_files)
        path_images_non_processed = [p for p in path_images if Path(p).stem not in keys_processed]
        annotations_non_processed = {k: v for k, v in annotations_image_to_label.items() if k not in keys_processed}
        path_images = path_images_non_processed
        annotations_image_to_label = annotations_non_processed

    ann = [len(a) for a in annotations_image_to_label.values() if len(a) > 0]
    print('numbers of annotations', sum(ann))
    return path_images, annotations_image_to_label


def get_selected_images():
    dir_image_selected = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/disk/preprocessed_data/Downsampling_bigObjects_annotations_PCA3/hyper'
    dir_image_selected = list(Path(dir_image_selected).rglob("*.npy"))  # check extension of the masks

    name_images = [Path(d).stem for d in dir_image_selected]
    key_images = [n.replace('hyper_', '') for n in name_images]
    return key_images


def loader_separa_data_unsupervised(path_file_annotations, only_labeled=False, include_labels=False):
    key_images = get_selected_images()

    annotations = json.load(open(path_file_annotations, 'r'))

    images_info = annotations['images']
    partial_path_images = [img['file_name'] for img in images_info]

    folder_project = '/home/scasao/SEPARA/unizar_DL4HSI/separa/'
    path_images = [folder_project + p for p in partial_path_images]
    name_files = [p.split('/')[-1] for p in partial_path_images]

    name_image_to_label = [img.split('.')[0] for img in name_files]

    # Get dictionary of images with their corresponding masks
    annotations_image_to_label = get_annotations(annotations['annotations'],name_image_to_label, include_labels)

    # Load only labeled images
    if only_labeled:
        annotations_non_empty = {k: v for k, v in annotations_image_to_label.items() if len(v) > 0}
        path_images_non_empty = [p for p in path_images if Path(p).stem in annotations_non_empty.keys()]
        annotations_image_to_label = annotations_non_empty
        path_images = path_images_non_empty

    path_images_selected = [p for p in path_images if Path(p).stem in key_images]
    annotations_selected = {k: v for k, v in annotations_image_to_label.items() if k in key_images}
    path_images = path_images_selected
    annotations_image_to_label = annotations_selected

    ann = [len(a) for a in annotations_image_to_label.values() if len(a) > 0]
    print('numbers of annotations', sum(ann))
    return path_images, annotations_image_to_label