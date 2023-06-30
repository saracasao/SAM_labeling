import re
import numpy as np

from PIL import Image
from skimage import measure
from itertools import groupby
from datetime import date


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


def get_image_from_masks(name_masks, all_name_images):
    name_image_wo_ext = [f.split('.')[0] for f in all_name_images]
    id_image_in_masks = [re.search('Img_(.*)_L', f).group(1) for f in name_masks]

    idx_name_image_with_labeling = [i for i,f in enumerate(name_image_wo_ext) if f in id_image_in_masks]
    name_image_with_labeling = [f for i,f in enumerate(all_name_images) if i in idx_name_image_with_labeling]
    return name_image_with_labeling


def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))


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
        segmentation = [0 if i < 0 else i for i in segmentation]

        polygons.append(contour)
        segmentations.append(segmentation)
        length.append(len(contour))
    idx = length.index(max(length))

    polygon = polygons[idx]
    polygon = np.array(polygon, np.int32)
    segmentation = segmentations[idx]
    segmentation = np.array(segmentation, np.int32)
    return polygon, segmentation


def binary_mask_to_bbox(binary_mask):
    segmentation = np.where(binary_mask == 255)
    xmin = int(np.min(segmentation[1]))
    xmax = int(np.max(segmentation[1]))
    ymin = int(np.min(segmentation[0]))
    ymax = int(np.max(segmentation[0]))

    width = xmax - xmin
    height = ymax - ymin

    return xmin, ymin, width, height