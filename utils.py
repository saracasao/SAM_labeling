import cv2
import re
import numpy as np
import torch
import json
import os

from color_list import rgb_dict
from PIL import Image
from skimage import measure
from itertools import groupby
from pathlib import Path
from datetime import date
from typing import Any, Dict, Generator, ItemsView, List, Tuple

kernel = {'FILM': [np.ones((7, 7), np.uint8), 1],
          'BARQUILLA': [np.ones((7, 7), np.uint8), 1],
          'CARTON': [np.ones((7, 7), np.uint8), 1],
          'ELEMENTOS_FILIFORMES': [np.ones((1, 1), np.uint8), 2],
          'CINTA_VIDEO': [np.ones((1, 1), np.uint8), 2],
          'BOLSA': [np.ones((7, 7), np.uint8), 1],
          'ELECTRONICA': [np.ones((7, 7), np.uint8), 1]}


def mask_to_rle_pytorch(tensor: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = torch.nonzero(diff)

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [
                torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                cur_idxs + 1,
                torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})
    return out


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    assert type(rle) == dict, "Error in rle type {}".format(rle)
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


def remove_small_regions(mask, label):
    k = kernel[label][0]
    iter = kernel[label][1]

    clean_noise = cv2.erode(mask, k, iterations=iter)
    dilation = cv2.dilate(clean_noise, k, iterations=(2*iter))
    # final_correction = cv2.erode(dilation, kernel, iterations=iter)
    return dilation


def process_mask(d_mask, save = True):
    mask = cv2.imread(d_mask, 0)
    label_ann = re.search('_L(.*)_N', d_mask).group(1)

    mask_processed = remove_small_regions(mask, label_ann)
    if save:
        folder_file, name_file = os.path.split(d_mask)
        dir_mask_processed = folder_file.replace('Masks', 'Masks_processed')
        if not os.path.exists(dir_mask_processed):
            os.makedirs(dir_mask_processed)

        final_file_path = dir_mask_processed + '/' + name_file
        cv2.imwrite(final_file_path, mask_processed)
    return mask_processed


def get_mask_from_images(name_images, all_name_masks):
    name_image_wo_ext = [f.split('.')[0] for f in name_images]
    id_image_in_masks = [re.search('Img_(.*)_L', f).group(1) for f in all_name_masks]

    name_mask_of_images = []
    for name in name_image_wo_ext:
        if name in id_image_in_masks: # TODO if does not exist name define empty label
            idx_n_mask = id_image_in_masks.index(name)
            name_mask = all_name_masks[idx_n_mask]

            assert name in name_mask, "ID of image does not match between mask and original image"
            name_mask_of_images.append(name_mask)
        else:
            raise NameError('Image {} does not has corresponding mask'.format(name))
    return name_mask_of_images


def get_image_from_masks_no_filter(dir_masks, all_dir_images):
    mask_files = [os.path.split(d)[1] for d in dir_masks]
    img_files = [os.path.split(d)[1] for d in all_dir_images]

    name_image_wo_ext = [f.split('.')[0] for f in img_files]
    id_image_in_masks = [re.search('Img_(.*)_L', f).group(1) for f in mask_files]

    info_image_to_label = [(f, name_image_wo_ext[i]) for i,f in enumerate(all_dir_images) if name_image_wo_ext[i] in id_image_in_masks]
    return info_image_to_label


def get_image_from_masks(dir_masks, all_dir_images):
    mask_files = [os.path.split(d)[1] for d in dir_masks]
    # img_files = [os.path.split(d)[1] for d in all_dir_images]
    # name_image_wo_ext = [f.split('.')[0] for f in img_files]

    name_image_wo_ext = [f.stem for f in all_dir_images]
    id_image_in_masks = [re.search('Img_(.*)_L', f).group(1) for f in mask_files]
    keys_in_common = list(set(name_image_wo_ext) & set(id_image_in_masks)) # images belonging to both sets

    info_image_to_label = [(f, name_image_wo_ext[i]) for i,f in enumerate(all_dir_images) if name_image_wo_ext[i] in keys_in_common]
    dir_masks_in_common = [d_mask for i, d_mask in enumerate(dir_masks) if id_image_in_masks[i] in keys_in_common]
    return info_image_to_label, dir_masks_in_common


def get_common_data_by_keys(keys, list_dir_masks, list_dir_images, key_images):
    mask_files = [os.path.split(d)[1] for d in list_dir_masks]
    key_masks = [re.search('Img_(.*)_L', f).group(1) for f in mask_files]

    keys_in_common = list(set(key_masks) & set(keys)) # keys belonging to both sets

    assert len(list_dir_images) == len(key_images) and len(list_dir_masks) == len(key_masks)

    info_image_to_label = [(str(f), key_images[i]) for i,f in enumerate(list_dir_images) if key_images[i] in keys_in_common]
    dir_masks = [str(d_mask) for i, d_mask in enumerate(list_dir_masks) if key_masks[i] in keys_in_common]
    return info_image_to_label, dir_masks


def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)


# def binary_mask_to_rle(binary_mask):
#     rle = {'counts': [], 'size': list(binary_mask.shape)}
#     counts = rle.get('counts')
#     for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
#         if i == 0 and value == 1:
#             counts.append(0)
#         counts.append(len(list(elements)))
#     return rle


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """
    polygons, segmentations, length = [], [], []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask)

    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segmentation = [0 if i < 0 else int(i) for i in segmentation]

        polygons.append(contour)
        segmentations.append(segmentation)
    polygons = [np.array(p, np.int32) for p in polygons]
    return polygons, segmentations


def binary_mask_to_bbox(binary_mask):
    binary_mask = np.asarray(binary_mask, dtype=np.uint8)

    segmentation = np.where(binary_mask == 255)
    xmin = int(np.min(segmentation[1]))
    xmax = int(np.max(segmentation[1]))
    ymin = int(np.min(segmentation[0]))
    ymax = int(np.max(segmentation[0]))

    width = xmax - xmin
    height = ymax - ymin

    return xmin, ymin, width, height


def get_existing_masks(name_img, annotations, labels_included =False):
    name = str(Path(name_img).stem)
    ann_img = annotations[name]

    masks, labels = [], []
    for a in ann_img:
        if labels_included:
            rle = a[0]
            label = a[1]
            if len(rle) > 0 and len(rle[0]) > 0:
                mask = rle_to_mask(rle[0])
                masks.append(np.array(mask))
                labels.append(label)
        elif len(a) > 0 and len(a[0]) > 0:
            mask = rle_to_mask(a[0])
            masks.append(np.array(mask))

    if not labels_included and len(masks) > 1:
        size_mask = np.shape(masks[0])
        ref_mask = False*np.ones(size_mask)
        for mask in masks:
            ref_mask = np.logical_or(ref_mask, mask)
        masks = ref_mask
    elif not labels_included and len(masks) == 1:
        masks = masks[0]
    elif len(masks) == 0:
        masks = None

    if not labels_included:
        return masks
    else:
        return masks, labels


def get_mask_img(mask, color_label):
    color = np.array(color_label)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image *= 255
    mask_image = mask_image.astype(np.uint8)
    b_mask, g_mask, r_mask, alpha = cv2.split(mask_image)
    mask_image_3channels = cv2.merge((b_mask, g_mask, r_mask))
    return mask_image_3channels


def get_multiple_mask_img(masks):
    labels = []
    h, w = masks[0][0].shape[-2:]
    final_mask = np.zeros((h, w, 3)).astype(np.uint8)
    for bool_mask, l in zip(*masks):
        color = np.array(rgb_dict[l])
        labels.append(l)
        h, w = bool_mask.shape[-2:]
        mask_image = bool_mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image *= 255
        mask_image = mask_image.astype(np.uint8)
        b_mask, g_mask, r_mask, alpha = cv2.split(mask_image)
        mask_image_3channels = cv2.merge((b_mask, g_mask, r_mask))
        final_mask = cv2.bitwise_or(final_mask, mask_image_3channels)
    return final_mask, labels


def get_path_file(path, name):
    file_path = None
    for root, subdirs, files in os.walk(path):
        if name in files:
           file_path = os.path.join(root, name)
    return file_path


def load_txt(path_file):
    with open(path_file) as f:
        lines = [line.rstrip() for line in f]
    return lines


def get_annotations(annotations,name_images):
    ann = {name_img: [] for name_img in name_images}

    ann_image_ids = [ann['image_id'] for ann in annotations]
    ann_image_ids = np.array(ann_image_ids)
    for name in name_images:
        if name in ann_image_ids:
            idx_ann = np.where(name == ann_image_ids)[0]
            for idx in idx_ann:
                img_seg   = annotations[idx]['segmentation']
                img_label = annotations[idx]['category_id']
                img_ann   = (img_seg, img_label)
                ann[name].append(img_ann)
    return ann


def get_colors():
    rgb_dict = {i: np.concatenate([np.random.random(3), np.array([0.6])], axis=0) for i in list(range(30))}
    return rgb_dict


if __name__ == "__main__":
    name_file = '20220928_10-51-47.jpg'
    path = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/disk/'

    get_path_file(path, name_file)