import os
import cv2
import copy
import json
import time
import numpy as np

from colors_list import rgb_dict
from segment_anything import SamPredictor, sam_model_registry
from utils import remove_small_regions

SEPARA_LABELS = {
    "FONDO": 0,
    "FILM": 1,
    "BARQUILLA": 2,
    "CARTON": 3,
    "CINTA_VIDEO": 4,
    "ELEMENTOS_FILIFORMES": 5,
}

# Configuration
mask_selector_size = False
mask_selector_score = True

# Drawing variables
width_label, height_label = 10, 10


def draw_labeling_process_bboxes(img, masks, bboxes, labels):
    masks_to_draw = [masks[-1]]
    bboxes_to_draw = [bboxes[-1]]
    labels_to_draw = [labels[-1]]
    assert len(masks_to_draw) == len(bboxes_to_draw)

    # width_label, height_label = 30, 30
    for i, mask in enumerate(masks_to_draw):
        bbox = bboxes_to_draw[i]
        color = rgb_dict[SEPARA_LABELS[labels_to_draw[i]]]
        img = cv2.addWeighted(img, 1, mask, 0.8, 0)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        height_bbox = bbox[3] - bbox[1]
        y_min_label = bbox[1] + int(height_bbox / 2)

        bbox_label = [bbox[2], y_min_label, bbox[2] + width_label, y_min_label + height_label]
        size, _ = cv2.getTextSize(labels_to_draw[i], cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        width_text, height_text = size

        cv2.rectangle(img, (bbox_label[0], bbox_label[1] - height_text), (bbox_label[0] + width_text, bbox_label[1]), color, -1)
        cv2.putText(img, labels_to_draw[i], (bbox_label[0], bbox_label[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    return img


def get_mask_img(mask, label, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image *= 255
    mask_image = mask_image.astype(np.uint8)
    b_mask, g_mask, r_mask, alpha = cv2.split(mask_image)
    mask_image_3channels = cv2.merge((b_mask, g_mask, r_mask))
    mask_image_3channels = remove_small_regions(mask_image_3channels, label)
    return mask_image_3channels


def save_labeling_process(masks, labels, dir, save_decision, list_of_id_ann):
    dir_mask = dir.split('/')
    img_name = dir_mask[-1]

    dir_mask.pop(-1)
    dir_mask.remove('disk')

    dir_mask.insert(7, 'Masks')
    dir_mask = os.path.join(*dir_mask)
    dir_mask = '/' + dir_mask
    if not os.path.exists(dir_mask):
        os.makedirs(dir_mask)

    for i, mask in enumerate(masks):
        label = labels[i]
        save = save_decision[i]
        id_ann = list_of_id_ann[i]
        if save == 1:
            str_idx_mask = str(i)
            str_idx_mask = str_idx_mask.zfill(4)

            img_name = img_name.split('.')[0]
            name_file = 'Img_' + img_name + '_L' + label + '_N' + str_idx_mask + '_ID' + str(id_ann)
            final_path = dir_mask + '/' + name_file + '.png'
            print('save mask', final_path)
            cv2.imwrite(final_path, mask)


def keep_mask(list_of_masks, mask_to_save):
    gray = cv2.cvtColor(mask_to_save, cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    list_of_masks.append(thresh1)
    return list_of_masks


def check_coord(bbox_coord):
    xmin = bbox_coord[0]
    ymin = bbox_coord[1]
    width = bbox_coord[2]
    height = bbox_coord[3]

    xmax = xmin + width
    ymax = ymin + height
    return np.array([int(xmin), int(ymin), int(xmax), int(ymax)])


def clean_existing_masks(masks_path, dir_images):
    if os.path.exists(masks_path):
        masks_folder = [f for f in sorted(os.listdir(masks_path)) if '.' not in f]

        mask_files = []
        for f in masks_folder:
            subfolder = sorted(os.listdir(masks_path + '/' + f))
            for subf in subfolder:
                file_names = sorted(os.listdir(masks_path + '/' + f + '/' + subf + '/DALSA/'))
                mask_files = mask_files + file_names

        mask_keys = [m[4:21] for m in mask_files]
        images_keys = [d.split('/')[-1][:17] for d in dir_images]

        assert len(images_keys) == len(dir_images)

        dir_images = [dir_images[i] for i, k in enumerate(images_keys) if k not in mask_keys]
    return dir_images


def clean_sam_failures(sam_failures_path, keys):
    if os.path.exists(sam_failures_path):
        with open(sam_failures_path, 'r') as file_failures:
            file_lines = [line.rstrip() for line in file_failures]
        clean_keys = [k for k in keys if k not in file_lines]
    else:
        file_lines = []
        clean_keys = keys
    return file_lines, clean_keys


# Load SAM model
device = "cuda"
model_type = "vit_h"
sam_checkpoint = "/home/scasao/SAM/segment-anything/checkpoints/sam_vit_h_4b8939.pth"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)


# Load bbox labels
data = json.load(open('/home/scasao/SEPARA/unizar_DL4HSI/separa/data/only_bboxes_labeled_all_annotations.json', 'r'))
keys = sorted(data.keys())

#Load files already processed
masks_path = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/Masks'
keys = clean_existing_masks(masks_path, keys)

sam_failures_path = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/Masks/labeling_failures.txt'
previous_files_failure_sam, keys_to_label = clean_sam_failures(sam_failures_path, keys)

files_sam_failures = []
for key in keys_to_label:
    label_ok = None
    path_img = '/home/scasao/SEPARA/unizar_DL4HSI/separa/' + key
    if '.npy' in path_img:
        image = np.load(path_img)
    else:
        image = cv2.imread(path_img)
    height, width, _ = image.shape

    # API TO SELECT MASK THAT WE WANT
    predictor.set_image(image)
    window_name = 'Image'
    cv2.namedWindow(winname=window_name)

    # set mouse callback function for window
    raw_image = copy.deepcopy(image)
    img_to_show = image.copy()

    list_of_masks_image, list_of_mask_showed, list_of_labels, list_of_bboxes, list_of_quality, list_of_id_ann = [], [], [], [], [], []
    annotations = data[key]
    for ann in annotations:
        label_ok = None
        bbox = ann['bbox']
        label = ann['label']
        id_ann = ann['id_annotation']
        print('LABEL:', label)

        bbox_arr = check_coord(bbox)
        list_of_bboxes.append([bbox_arr[0],bbox_arr[1], bbox_arr[2],bbox_arr[3]])

        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox_arr[None, :],
            multimask_output=False,
        )

        # Select biggest mask -> global object
        masks_size = []
        for i, mask in enumerate(masks):
            masks_size.append(mask.sum())

        scores = list(scores)
        if mask_selector_size:
            # 1. Select the mask by size
            idx_mask_selected = masks_size.index(max(masks_size))
        elif mask_selector_score:
            # 2. Select the mask by confidence
            idx_mask_selected = scores.index(max(scores))
        else:
            idx_mask_selected = scores.index(max(scores))

        # Final mask
        selected_mask = masks[idx_mask_selected]

        mask_overlap = get_mask_img(selected_mask, label)
        img_to_show = cv2.addWeighted(img_to_show, 1, mask_overlap, 0.9, 1)
        list_of_masks_image = keep_mask(list_of_masks_image, mask_overlap)

        list_of_mask_showed.append(mask_overlap)
        list_of_labels.append(label)
        list_of_id_ann.append(id_ann)

        # Draw process
        img_to_show = draw_labeling_process_bboxes(img_to_show, list_of_mask_showed, list_of_bboxes, list_of_labels)
        final_img = cv2.hconcat([img_to_show, raw_image])
        cv2.imshow(window_name, final_img)

        # Opencv fails combining cv2.imshow with input() functions from python -> through while loops to show the image it seems like it is working
        show = True
        while show:
            cv2.imshow(window_name, final_img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

            cv2.waitKey(2)
            print('Label ok? OR stop and save process (1 = ok, 0 = do not ok, 2 = exit) ')
            label_ok = int(input())
            if label_ok == 1:
                show = False
                list_of_quality.append(label_ok)
            elif label_ok == 0:
                show = False
                files_sam_failures.append(key)
                list_of_quality.append(label_ok)
            elif label_ok == 2:
                break

            if len(list_of_masks_image) == len(annotations) and label_ok is not None:
                show = False
                list_of_quality.append(label_ok)
                save_labeling_process(list_of_masks_image, list_of_labels, path_img, list_of_quality, list_of_id_ann)
        if label_ok == 2:
            break
    if label_ok == 2:
        break

# Save changes in files processed
files_sam_failures = list(set(files_sam_failures))
files_failure_sam = previous_files_failure_sam + files_sam_failures
with open(sam_failures_path, "w") as f:
    for line in files_failure_sam:
        f.write(line + "\n")
f.close()


cv2.destroyAllWindows()

