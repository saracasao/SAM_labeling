import os
import cv2
import copy
import numpy as np
from color_list import rgb_dict
from segment_anything import SamPredictor, sam_model_registry
from utils import remove_small_regions, binary_mask_to_bbox

"""
Tool for labeling with SAM. 
1. The code ask you how do you want to label: points or bboxes
2. To introduce the label of the object that you will label press N 
3. In the terminal the code ask you for the label
4. Introduce label and press enter
5. Start labeling

Different tools:
-> if you want to define a new label for the same image -> press N again, the code ask you again for the new label (steps 3. and 4.)
-> if you want to label the next image -> press D (you can change this letter in line 313) -> the masks and its labels are saved in ./Mask folder and a new image will appear to start the process (from step 2.)
-> if you want to finish the process press Esc
-> if you make a mistake during the labeling you can start the image again pressing C 
"""


# Configuration
mask_selector_size = False
mask_selector_score = True

# GLOBAL VARIABLES USED BY THE MOUSE CALLBACK
mode = None
clicks = list()
label_clicks = list()
signal = False
n_clicks = 0
current_mouse_coordinates = list()

# Drawing variables
text_color_sticker = (0, 0, 0)
rectangle_color_sticker = (127, 127, 127)

width_label, height_label = 30, 30
scale_stickers_height, scale_stickers_width = 0.05, 0.4


# TODO DEFINE CONFIG TO SELECT: 1. THE TYPE OF LABELING METHOD (POINTS/BBOXES), 2. SELECTION OF MASK BY SIZE OR CONFIDENCE, 3. IMAGE PATH TO LOAD AND PATH TO SAVE MASKS, 4. FORMAT OF LABELS

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
    # offset_text = [int(0.05*init_coord[0]), int(0.2*init_coord[1])]

    # Text configuration
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    text = 'Press N to introduce \n a new label in the terminal'
    position = (coord_of_stickers[0][0] + offset_text[0], coord_of_stickers[0][1] + offset_text[1])

    # Draw in image
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    line_height = text_size[1] + 5

    img = cv2.rectangle(img, coord_of_stickers[0], coord_of_stickers[1], rectangle_color_sticker, -1)
    img = cv2.rectangle(img, coord_of_stickers[0], coord_of_stickers[1], text_color_sticker, 2)
    cv2.putText(img, 'Press N to introduce a new label in the terminal', (coord_of_stickers[0][0] + offset_text[0], coord_of_stickers[0][1] + offset_text[1]), cv2.FONT_HERSHEY_SIMPLEX, scale_text*1, text_color_sticker, 2)

    # xt, yt0 = position
    # for i, line in enumerate(text.split('\n')):
    #     yt = yt0 + i*line_height
    #     cv2.putText(img, line, (xt + offset_text[0], yt + offset_text[1]),font, font_scale, text_color_sticker, thickness)
    return img


def draw_set_label_stickers(img, height, width, scale_stickers_height, scale_stickers_width):
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

    img = cv2.rectangle(img, coord_of_stickers[0], coord_of_stickers[1], rectangle_color_sticker, -1)
    img = cv2.rectangle(img, coord_of_stickers[0], coord_of_stickers[1], text_color_sticker, 2)
    cv2.putText(img, 'Introduce the new label in the terminal', (coord_of_stickers[0][0] + offset_text[0], coord_of_stickers[0][1] + offset_text[1]), cv2.FONT_HERSHEY_SIMPLEX, scale_text*1, text_color_sticker, 2)
    return img


def mouse_callback(event, x, y, flags, param):
    global mode
    global signal
    global clicks
    global label_clicks
    global n_clicks
    global current_mouse_coordinates

    if mode == 'points':
        #middle-bottom-click event value is 2
        if event == cv2.EVENT_MBUTTONDOWN:
            #store the coordinates of the right-click event
            clicks.append([x, y])
            label_clicks.append(0)
            cv2.circle(img_to_show, (x, y), 5, (0, 0, 255), -1)
            signal = True
        #this just verifies that the mouse data is being collected
        #you probably want to remove this later
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append([x, y])
            label_clicks.append(1)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            signal = True
    elif mode == 'bboxes':
        #this just verifies that the mouse data is being collected
        #you probably want to remove this later
        if n_clicks == 0 and event == cv2.EVENT_LBUTTONDOWN:
            clicks.append([x,y])
            cv2.circle(img_to_show, clicks[0], 5, (0, 0, 255), -1)
            n_clicks += 1
        elif n_clicks == 1 and event == cv2.EVENT_MOUSEMOVE:
            current_mouse_coordinates = [x, y]
        elif n_clicks == 1 and event == cv2.EVENT_LBUTTONDOWN:
            clicks = clicks[0] + [x, y]
            signal = True
            n_clicks = 0


def draw_labeling_process_points(img, mask, label):
    color = rgb_dict[label]

    segmentation = np.where(mask == 255)
    x_min = int(np.min(segmentation[1]))
    x_max = int(np.max(segmentation[1]))
    y_min = int(np.min(segmentation[0]))
    y_max = int(np.max(segmentation[0]))

    height = y_max - y_min
    y_min_label = y_min + int(height / 2)

    bbox_label = [x_max, y_min_label, x_max + width_label, y_min_label + height_label]
    cv2.rectangle(img, (bbox_label[0], bbox_label[1] - 10), (bbox_label[2],bbox_label[3]), color, -1)
    cv2.rectangle(img, (bbox_label[0], bbox_label[1] - 10), (bbox_label[2], bbox_label[3]), (0, 0, 0), 1)
    cv2.putText(img, str(label), (bbox_label[0] + 5, bbox_label[1] + int(height_label / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 0), 2)
    # cv2.putText(img, str(label), (bbox_label[0], bbox_label[1] + int(height_label / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (0, 0, 0), 1)
    return img


def draw_current_info(img, xmin, ymin, xmax, ymax, label):
    color = rgb_dict[label]
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

    height = ymax - ymin
    y_min_label = ymin + int(height / 2)

    # width_label, height_label = 30, 30
    bbox_label = [xmax, y_min_label, xmax + width_label, y_min_label + height_label]
    cv2.rectangle(img, (bbox_label[0], bbox_label[1] - 10), (bbox_label[2],bbox_label[3]), color, -1)
    cv2.rectangle(img, (bbox_label[0], bbox_label[1] - 10), (bbox_label[2], bbox_label[3]), (0, 0, 0), 1)
    cv2.putText(img, str(label), (bbox_label[0], bbox_label[1] + int(height_label / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 0), 2)
    return img


def draw_labeling_process_bboxes(img, masks, bboxes, labels):
    masks_to_draw = masks[:-1]
    bboxes_to_draw = bboxes[:-1]

    assert len(masks_to_draw) == len(bboxes_to_draw)

    # width_label, height_label = 30, 30
    for i, mask in enumerate(masks_to_draw):
        bbox = bboxes_to_draw[i]
        color = rgb_dict[labels[i]]
        img = cv2.addWeighted(img, 1, mask, 0.8, 0)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        height_bbox = bbox[3] - bbox[1]
        y_min_label = bbox[1] + int(height_bbox / 2)

        bbox_label = [bbox[2], y_min_label, bbox[2] + width_label, y_min_label + height_label]
        cv2.rectangle(img, (bbox_label[0], bbox_label[1] - 10), (bbox_label[2], bbox_label[3]), color, -1)
        cv2.rectangle(img, (bbox_label[0], bbox_label[1] - 10), (bbox_label[2], bbox_label[3]), (0, 0, 0), 1)
        cv2.putText(img, str(labels[i]), (bbox_label[0], bbox_label[1] + int(height_label / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img


def drawing_bboxes_over_time(img, label):
    global current_mouse_coordinates
    color = rgb_dict[label]
    cv2.rectangle(img, clicks[0], current_mouse_coordinates, color, 2)
    return img


def get_mask_img(mask, random_color=True):
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

    # Clean mask to delete the small regions
    # mask_image_3channels = remove_small_regions(mask_image_3channels)
    return mask_image_3channels


def get_mask_name(mask_name):
    if '/' in mask_name:
        mask_name = mask_name.split('/')[-1]

    if '.' in mask_name:
        mask_name = mask_name.split('.')[0]
    return mask_name


def save_labeling_process(masks, labels, img_name, dir):
    masks_path = dir + '/Masks/'
    img_name = get_mask_name(img_name)

    if not os.path.exists(masks_path):
        os.makedirs(masks_path)

    for i, mask in enumerate(masks):
        label = labels[i]

        label_str = str(label)
        label_str = label_str.zfill(4)

        str_idx_mask = str(i)
        str_idx_mask = str_idx_mask.zfill(4)

        name_file = 'Img_' + img_name + '_L' + label_str + '_N' + str_idx_mask
        final_path = masks_path + '/' + name_file + '.png'
        cv2.imwrite(final_path, mask)


def save_labeling_process_bbox_as_ref(masks, labels, img_name, dir, annotations_ref):
    masks_path = dir + '/Masks/'
    if not os.path.exists(masks_path):
        os.makedirs(masks_path)

    img_name = get_mask_name(img_name)
    for i, mask in enumerate(masks):
        id_ann = None
        ann = annotations_ref[img_name]
        if len(ann) == 1:
            id_ann = ann[0]['id_annotation']
        elif len(ann) > 1:
            bbox_mask = binary_mask_to_bbox(mask)
            middle_point = [int(bbox_mask[0] + bbox_mask[2] / 2), int(bbox_mask[1] + bbox_mask[3] / 2)]
            for j, a in enumerate(ann):
                bbox_annotation = ann[j]['bbox']
                mask_inside = bbox_annotation[0] < middle_point[0] < bbox_annotation[0] + bbox_annotation[2] and bbox_annotation[1] < middle_point[1] < bbox_annotation[1] + bbox_annotation[3]
                if mask_inside:
                    id_ann = bbox_annotation[j]['id_annotation']
                    break

        assert id_ann is not None

        label = labels[i]

        label_str = str(label)
        label_str = label_str.zfill(4)

        str_idx_mask = str(i)
        str_idx_mask = str_idx_mask.zfill(4)

        name_file = 'Img_' + img_name + '_L' + label_str + '_N' + str_idx_mask + '_ID' + str(id_ann)
        final_path = masks_path + '/' + name_file + '.png'
        cv2.imwrite(final_path, mask)


def keep_mask(list_of_masks, mask_to_save):
    gray = cv2.cvtColor(mask_to_save, cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    list_of_masks.append(thresh1)
    return list_of_masks


def check_coord(bbox_coord):
    xmin = bbox_coord[0]
    ymin = bbox_coord[1]
    xmax = bbox_coord[2]
    ymax = bbox_coord[3]

    if xmin > xmax:
        new_xmax = xmin
        new_xmin = xmax
    else:
        new_xmax = xmax
        new_xmin = xmin

    if ymin > ymax:
        new_ymax = ymin
        new_ymin = ymax
    else:
        new_ymax = ymax
        new_ymin = ymin
    return np.array([new_xmin, new_ymin, new_xmax, new_ymax])


device = "cuda"
model_type = "vit_h"
sam_checkpoint = "/home/scasao/SAM/segment-anything/checkpoints/sam_vit_h_4b8939.pth"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# CHANGE PATH HERE
path_images = '/home/scasao/Documents/TEST_DATA'
# image_names = list(Path(path_images).rglob("*.jpg"))
image_names = [f for f in sorted(os.listdir(path_images)) if ('.jpg' in f or '.png' in f)]  # all images are in a single folder
print('Number of images for labeling', len(image_names))

print('HI! Which mode do you want for labeling? (points/bboxes)')
mode = str(input())
assert mode == 'points' or mode == 'bboxes', "Mode introduced does not exist"

# LOAD IMAGES TO LABEL -> DIFFERENT FOR EACH DATASET
for name_img in image_names:
    if '.npy' in name_img:
        image = np.load(path_images + '/' + name_img)
    else:
        image = cv2.imread(path_images + '/' + name_img)
    height, width, _ = image.shape

    # API TO SELECT MASK THAT WE WANT
    predictor.set_image(image)

    # set mouse callback function for window
    window_name = 'Image'
    cv2.namedWindow(winname=window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # USE TO LABEL ONLINE
    raw_image = copy.deepcopy(image)
    img_to_show = draw_init_stickers(image, height, width, scale_stickers_height, scale_stickers_width)
    img_to_show = image.copy()

    list_of_masks_image, list_of_mask_showed, list_of_labels, list_of_bboxes = [], [], [], []
    while True:
        save_process = True

        cv2.imshow(window_name, img_to_show)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord('n'):
            img_to_show = raw_image.copy()
            print('Label of the next mask?')
            label = int(input())
        elif k == ord('c'):
            list_of_masks_image, list_of_mask_showed, list_of_labels, list_of_bboxes = [], [], [], []
            img_to_show = copy.deepcopy(raw_image)
            save_process = False
        elif k == ord('d'):
            break

        # Draw when bboxes mode
        if mode == 'bboxes' and n_clicks == 1 and len(current_mouse_coordinates) > 0:
            img_to_draw = copy.deepcopy(raw_image)
            img_to_show = drawing_bboxes_over_time(img_to_draw, label)

        if signal:
            clicks_arr = np.array(clicks)
            label_clicks_arr = np.array(label_clicks)

            if mode == 'points':
                masks, scores, logits = predictor.predict(
                    point_coords=clicks_arr,
                    point_labels=label_clicks_arr,
                    multimask_output=True,
                )
            elif mode == 'bboxes':
                clicks_arr = check_coord(clicks_arr)
                img_to_show = draw_current_info(img_to_show, clicks_arr[0],clicks_arr[1], clicks_arr[2],clicks_arr[3], label)
                list_of_bboxes.append([clicks_arr[0],clicks_arr[1], clicks_arr[2],clicks_arr[3]])
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=clicks_arr[None, :],
                    multimask_output=False,
                )
            else:
                assert 'Mode introduce does not exist'

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

            mask_overlap = get_mask_img(selected_mask)
            img_to_show = cv2.addWeighted(img_to_show, 1, mask_overlap, 0.9, 1)
            list_of_masks_image = keep_mask(list_of_masks_image, mask_overlap)

            list_of_mask_showed.append(mask_overlap)
            list_of_labels.append(label)

            # Clean info for the new mask
            clicks = list()
            label_clicks = list()
            current_mouse_coordinates = list()

            # Draw process
            if mode == 'bboxes':
                img_to_show = draw_labeling_process_bboxes(img_to_show, list_of_mask_showed, list_of_bboxes, list_of_labels)
            elif mode == 'points':
                img_to_show = draw_labeling_process_points(img_to_show, list_of_masks_image[-1], label)
        signal = False

    if len(list_of_masks_image) > 0 and save_process:
        # New image without labels
        save_labeling_process(list_of_masks_image, list_of_labels, name_img, path_images)

    if k == 27:
        break
    print('Number of masks labeled', len(list_of_masks_image))
    print('Labels assgined', list_of_labels)
    cv2.destroyAllWindows()
