import os
import cv2
import copy
import numpy as np
import re
import imageio.v3 as imageio
from datetime import date
from color_list import rgb_dict
from pathlib import Path
from segment_anything import SamPredictor, sam_model_registry
from utils import remove_small_regions, binary_mask_to_bbox, get_existing_masks, get_multiple_mask_img
from image_loader import loader_separa_data

"""
Tool for labeling challenging objects with SAM. 
1. To introduce the label of the object that you will label press N 
2. In the terminal the code ask you for the label
3. Introduce label and press enter
4. Start labeling clicking on the object:
    4.1 Right button mouse = the pixel belongs to the object
    4.2 Press scroll wheel = the pixel does NOT belong to the object (if SAM includes some parts that are not from the object)
    4.3 Left button mouse = the mask is correct and you want to save it (sometimes opencv open a list of options, click again to save the mask). 
        When the mask has been saved, a message is shown in the terminal with the label and the path where the mask has been saved.
        If you don't click the left button mouse and continue to the next image or label, the mask is not going to be save

Different tools:
-> if you want to define a new label for the same image -> press N again, the code ask you again for the new label (steps 3. and 4.)
-> if you want to label the next image -> press D (you can change this letter in line 313)
-> if you want to finish the process press Esc
-> if you make a mistake during the labeling you can start the image again pressing C 
"""

n_annotation = 0

# Configuration
mask_selector_size = False
mask_selector_score = True

# GLOBAL VARIABLES USED BY THE MOUSE CALLBACK
clicks = list()
label_clicks = list()

signal, mask_ok = False, False
n_clicks = 0
current_mouse_coordinates = list()

# Drawing variables
width_label, height_label = 30, 30

# CHANGE PATH HERE
path_to_save_masks = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/annotations_25_10/Masks_raw/'


def draw_init_stickers(img, height, width, scale_stickers_height, scale_stickers_width):
    dims_ref_text = 2048
    scale_text = width / dims_ref_text

    center_w = int(width / 2)
    center_h = int(height / 3)
    dims_w = int(width * scale_stickers_width)
    dims_h = int(height * scale_stickers_height)

    init_coord = (center_w - int(dims_w/2), center_h - int(dims_h/2))
    end_coord = (init_coord[0] + dims_w, init_coord[1] + dims_h)
    coord_of_stickers = [init_coord, end_coord]
    offset_text = [int(0.05*init_coord[0]), int(0.1*init_coord[1])]

    # Text configuration
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    text = 'Press N to introduce \n a new label in the terminal'

    # Draw in image
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    img = cv2.rectangle(img, coord_of_stickers[0], coord_of_stickers[1], rectangle_color_sticker, -1)
    img = cv2.rectangle(img, coord_of_stickers[0], coord_of_stickers[1], text_color_sticker, 2)
    cv2.putText(img, 'Press N to introduce a new label in the terminal', (coord_of_stickers[0][0] + offset_text[0], coord_of_stickers[0][1] + offset_text[1]), cv2.FONT_HERSHEY_SIMPLEX, scale_text*1, text_color_sticker, 2)

    return img


def show_labels(img, mask, label):
    mask = mask*255
    color = rgb_dict[label]

    segmentation = np.where(mask == 255)
    x_min = int(np.min(segmentation[1]))
    x_max = int(np.max(segmentation[1]))
    y_min = int(np.min(segmentation[0]))
    y_max = int(np.max(segmentation[0]))

    width = x_max - x_min
    height = y_max - y_min
    y_min_label = y_min + int(height / 2)

    # bbox_label = [x_max, y_min_label, x_max + width_label, y_min_label + height_label]
    bbox_label = [x_min + width // 2, y_min + height // 2, x_min + width // 2 + width_label, y_min + height // 2 + height_label]
    cv2.rectangle(img, (bbox_label[0], bbox_label[1] - 10), (bbox_label[2],bbox_label[3]), color, -1)
    cv2.rectangle(img, (bbox_label[0], bbox_label[1] - 10), (bbox_label[2], bbox_label[3]), (0, 0, 0), 1)
    cv2.putText(img, str(label), (bbox_label[0] + 5, bbox_label[1] + int(height_label / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 0), 2)
    return img


def mouse_callback(event, x, y, flags, param):
    global signal
    global mask_ok
    global clicks
    global label_clicks
    global n_clicks
    global current_mouse_coordinates

    #middle-bottom-click event value is 2
    if event == cv2.EVENT_RBUTTONDOWN:
        #store the coordinates of the right-click event
        mask_ok = True
        signal = True
    #this just verifies that the mouse data is being collected
    #you probably want to remove this later
    if event == cv2.EVENT_MBUTTONDOWN:
        clicks.append([x, y])
        label_clicks.append(0)
        cv2.circle(image_hyper_to_show, (x, y), 5, (0, 255, 0), -1)
        signal = True
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append([x, y])
        label_clicks.append(1)
        cv2.circle(image_hyper_to_show, (x, y), 5, (0, 255, 0), -1)
        signal = True


def clean_already_annotated(dir, all_name_files):
    try:
        masks_path = dir + '/Masks'

        name_files = os.listdir(masks_path)
        masks_id = [re.search('Img_(.*)_L', n_mask).group(1) for n_mask in name_files]
        image_id = [re.search('(.*).jpg', n_img).group(1) for n_img in all_name_files]

        image_to_label = list(set(image_id) ^ set(masks_id))

        name_image_to_label = [all_name_files[i] for i, name in enumerate(image_id) if name in image_to_label]
    except:
        name_image_to_label = all_name_files
    return name_image_to_label


def get_mask_img_random_color(mask, random_color=True):
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

    # Clean small regions
    # mask_image_3channels = remove_small_regions(mask_image_3channels)
    return mask_image_3channels


def get_mask_name(mask_name):
    if '/' in mask_name:
        mask_name = mask_name.split('/')[-1]

    if '.' in mask_name:
        mask_name = mask_name.split('.')[0]
    return mask_name


def save_labeling_process(masks, labels, dir):
    global n_annotation
    folder = dir.split('/')[-4:-1]
    folder = os.path.join(*folder)

    masks_path = path_to_save_masks + folder
    img_name = get_mask_name(dir)

    if not os.path.exists(masks_path):
        os.makedirs(masks_path)

    img_name = get_mask_name(img_name)
    for i, mask in enumerate(masks):
        label = labels[i]

        label_str = str(label)
        label_str = label_str.zfill(4)

        str_idx_mask = str(n_annotation)
        str_idx_mask = str_idx_mask.zfill(4)

        name_file = 'Img_' + img_name + '_L' + label_str + '_N' + str_idx_mask
        final_path = masks_path + '/' + name_file + '.png'
        cv2.imwrite(final_path, mask)
        n_annotation += 1

        print('Masks labeled as {} save in {}'. format(list_of_labels, final_path))


def keep_mask(list_of_masks, mask_to_save):
    if mask_to_save is not None:
        gray = cv2.cvtColor(mask_to_save, cv2.COLOR_BGR2GRAY)
        _, thresh1 = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        list_of_masks.append(thresh1)
    return list_of_masks


def get_hyper_image(rgb_path):
    folder_file, name_file = os.path.split(rgb_path)
    folder_mask_processed = folder_file.replace('DALSA', 'Specim')
    hyper_path = folder_mask_processed + '/' + str(Path(name_file).stem) + '.tiff'
    hyper = imageio.imread(hyper_path)
    return hyper


def dims_reduction_random_bands(image_hyp):
    band50 = image_hyp[:, :, 50]
    band125 = image_hyp[:, :, 125]
    band200 = image_hyp[:, :, 200]
    hype_image_3channels = cv2.merge((band50, band125, band200))
    hype_image_3channels = hype_image_3channels.astype(np.float32)

    # Normalize hyper -> [0,1]
    # min_v = np.percentile(hype_image_3channels, 5, axis=(0, 1), keepdims=True)
    # max_v = np.percentile(hype_image_3channels, 95, axis=(0, 1), keepdims=True)
    # hype_image_3channels = (hype_image_3channels - min_v) / (max_v - min_v)
    # hype_image_3channels = np.clip(hype_image_3channels, 0, 1)
    hype_image_3channels /= 65535.
    hype_image_3channels *= 255.0
    hype_image_3channels = hype_image_3channels.astype(np.uint8)
    return hype_image_3channels


def alignment_multimodal(rgb_image, hyper_image=None):
    image_rgb_cropped = rgb_image[:1190, :]
    image_rgb_cropped = cv2.resize(image_rgb_cropped, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_CUBIC)
    if hyper_image is not None:
        image_hyp_cropped = hyper_image[:, :605, :]
        image_hyp_cropped_resize = cv2.resize(image_hyp_cropped, (image_rgb_cropped.shape[1], image_rgb_cropped.shape[0]), interpolation = cv2.INTER_CUBIC)
    else:
        image_hyp_cropped_resize = None
    return image_rgb_cropped, image_hyp_cropped_resize


# Load path images
path_file_annotations = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/annotations_update/2023-10-02_all_annotations_rle.json'
image_names, annotations_image_to_label = loader_separa_data(path_to_save_masks, path_file_annotations, clean_files_annotated=False, only_labeled=True,
                                                             path_processed_files=None, selected_keys='/home/scasao/COTR/test_matching/key_hyper_to_label.txt',include_labels=True)

print('Number of images for labeling', len(image_names))

device = "cuda"
model_type = "vit_h"
sam_checkpoint = "/home/scasao/SAM/segment-anything/checkpoints/sam_vit_h_4b8939.pth"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

scale_stickers_height, scale_stickers_width = 0.05, 0.4
text_color_sticker = (0, 0, 0)
rectangle_color_sticker = (127, 127, 127)

print('Total images to label {}'.format(len(image_names)))
for i, name_img in enumerate(image_names):
    print('N {} Image {}'.format(i, name_img))
    image = cv2.imread(name_img)

    # Get hyperspectral image and manual alignment of both sensors
    image_hyper_raw = get_hyper_image(name_img)
    image_hyper = image_hyper_raw.transpose(1, 2, 0)
    image, image_hyper = alignment_multimodal(image, image_hyper)
    image_hyper = dims_reduction_random_bands(image_hyper)

    height, width, _ = image.shape

    # API TO SELECT MASK THAT WE WANT
    predictor.set_image(image_hyper)

    # Set mouse callback function for window
    window_name = 'Image'
    cv2.namedWindow(winname=window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # USE TO LABEL ONLINE
    raw_image = copy.deepcopy(image)
    raw_image_hyper = copy.deepcopy(image_hyper)
    # Get existing masks
    existing_masks = get_existing_masks(name_img, annotations_image_to_label, labels_included=True)
    if existing_masks is not None:
        masks_merged, labels = get_multiple_mask_img(existing_masks)
        image = cv2.addWeighted(image, 1, masks_merged, 0.9, 1)
        for mask, l in zip(*existing_masks):
            image = show_labels(image, mask, l)
    img_to_show = draw_init_stickers(image, height, width, scale_stickers_height, scale_stickers_width)
    image_hyper_to_show = image_hyper.copy()

    list_of_masks_image, list_of_mask_showed, list_of_labels = [], [], []
    while True:
        save_process = True
        mask_ok = False
        mask_reference = cv2.hconcat([image_hyper_to_show, img_to_show])
        cv2.imshow(window_name, mask_reference)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: # Esc = end process
            break
        elif k == ord('n'): # N = new label
            image_hyper = raw_image_hyper.copy()
            print('Label of the next mask?')
            label = int(input())
        elif k == ord('c'): # C = clean segmentation process
            clicks = list()
            label_clicks = list()
            current_mouse_coordinates = list()

            list_of_masks_image, list_of_mask_showed, list_of_labels = [], [], []
            image_hyper_to_show = copy.deepcopy(raw_image_hyper)
            save_process = False
        elif k == ord('d'): # D = next image
            break

        if signal and not mask_ok:
            clicks_arr = np.array(clicks)
            label_clicks_arr = np.array(label_clicks)

            masks, scores, logits = predictor.predict(
                point_coords=clicks_arr,
                point_labels=label_clicks_arr,
                multimask_output=True,
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

            mask_overlap = get_mask_img_random_color(selected_mask)
            image_hyper_to_show = cv2.addWeighted(raw_image_hyper, 1, mask_overlap, 0.9, 1) # raw_image

        elif signal and mask_ok:
            list_of_masks_image = keep_mask(list_of_masks_image, mask_overlap)

            list_of_mask_showed.append(mask_overlap)
            list_of_labels.append(label)

            # Clean info for the new mask
            clicks = list()
            label_clicks = list()
            current_mouse_coordinates = list()
            save_labeling_process(list_of_masks_image, list_of_labels, name_img)

            list_of_masks_image, list_of_mask_showed, list_of_labels = [], [], []
            image_hyper_to_show = copy.deepcopy(raw_image_hyper)

        signal, mask_ok = False, False

    if k == 27:
        break

    cv2.destroyAllWindows()