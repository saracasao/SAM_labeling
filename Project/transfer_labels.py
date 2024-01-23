import cv2
import numpy as np
import torch
import imageio.v3 as imageio
import copy
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def get_mask_img(mask, random_color=False):
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


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)


def get_masks(anns):
    if len(anns) == 0:
        return
    elif len(anns) > 1:
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    else:
        sorted_anns = anns

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.8]])
        img[m] = color_mask
    return img


def show_anns(anns, idx):
    if len(anns) == 0:
        return
    elif len(anns) > 1:
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    else:
        sorted_anns = anns
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.8]])
        img[m] = color_mask
    ax.imshow(img)


def get_pos(i, n):
    threshold = int(n / 3)
    if i <= threshold:
        r = 0
    elif i <= 2*threshold:
        r = 1
    else:
        r = 2

    c = i
    if threshold < i < 2*threshold:
        c = i - (threshold + 1)
    elif 2*threshold < i < 3*threshold:
        c = i - ((2*threshold) + 1)
    return r, c


def normalize_percentile_v0(hyper, q_min=5, q_max=95, per_channel=True, clip=True):
    if per_channel:
        min_v = np.percentile(hyper, q_min, axis=(0,1), keepdims=True)
        max_v = np.percentile(hyper, q_max, axis=(0,1), keepdims=True)
    else:
        min_v = np.percentile(hyper, q_min)
        max_v = np.percentile(hyper, q_max)

    hyper = (hyper - min_v) / (max_v - min_v)
    if clip: hyper = np.clip(hyper, 0, 1)
    return hyper


def dims_reduction_random_bands(image_hyp):
    band100 = image_hyp[:, :, 10]
    band150 = image_hyp[:, :, 130]
    band200 = image_hyp[:, :, 200]
    hype_image_3channels = cv2.merge((band100, band150, band200))

    # Normalize hyper -> [0,1]
    image_hyp_pca = normalize_percentile_v0(hype_image_3channels)
    image_hyp_pca *= 255.0
    image_hyp_pca = image_hyp_pca.astype(np.uint8)
    return image_hyp_pca


def alignment_multimodal(rgb_image, hyper_image= None):
    image_rgb_cropped = rgb_image[:1190, :]
    if hyper_image is not None:
        image_hyp_cropped = hyper_image[:, :605, :]
        image_hyp_cropped_resize = cv2.resize(image_hyp_cropped, (image_rgb_cropped.shape[1], image_rgb_cropped.shape[0]), interpolation = cv2.INTER_CUBIC)
    else:
        image_hyp_cropped_resize = None
    return image_rgb_cropped, image_hyp_cropped_resize


show_masks_individually, show_individual_mask_on_img = False, False

path_img = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/disk/2022-10-11/23/DALSA/20221011_11-19-54.jpg'
image_rgb = cv2.imread(path_img)

path_img = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/disk/2022-10-11/23/Specim/20221011_11-19-54.tiff'
image_hyp = imageio.imread(path_img).transpose(1, 2, 0)
image_hyp = dims_reduction_random_bands(image_hyp)

image_rgb, image_hyp = alignment_multimodal(image_rgb, image_hyp)
# image_rgb = image_rgb[400:, 600:]
# image_hyp = image_hyp[400:, 600:]

img_raw = copy.deepcopy(image_rgb)

device = "cuda"
model_type = "vit_h"
sam_checkpoint = "/home/scasao/SAM/segment-anything/checkpoints/sam_vit_h_4b8939.pth"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(model=sam,
                                           points_per_side=30,
                                           pred_iou_thresh=0.5, # 0.86
                                           stability_score_thresh=0.75, # 0.92
                                           crop_n_layers=1,
                                           crop_n_points_downscale_factor=2,
                                           min_mask_region_area=10,  # Requires open-cv to run post-processing)
                                           )

masks_rgb = mask_generator.generate(image_rgb)
sorted_ann_rgb = sorted(masks_rgb, key=(lambda x: x['area']), reverse=True)
rgb_mask = get_masks(sorted_ann_rgb)

masks_hyper = mask_generator.generate(image_hyp)
sorted_ann_hyper = sorted(masks_hyper, key=(lambda x: x['area']), reverse=True)
hyper_mask = get_masks(sorted_ann_hyper)

fig, axs = plt.subplots(1, 2, figsize=(40, 20))
axs[0].imshow(image_rgb)
axs[0].imshow(rgb_mask)

axs[1].imshow(image_hyp)
axs[1].imshow(hyper_mask)
# show_anns(sorted_ann_rgb, 0)
# show_anns(sorted_ann_hyper, 1)
# plt.axis('off')
plt.show()

if show_individual_mask_on_img:
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    print('{} masks generated'.format(len(sorted_anns)))
    for i, ann in enumerate(sorted_anns):
        ann = ann['segmentation']
        mask = get_mask_img(ann)

        image_mask = cv2.addWeighted(img_raw, 1, mask, 0.9, 1)
        image_to_show = cv2.hconcat([img_raw, image_mask])

        show = True
        while show:
            cv2.imshow('masks', image_to_show)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif k == ord('d'):
                break

        if k == 27:
            break

if show_masks_individually:
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    print('{} masks generated'.format(len(sorted_anns)))
    rows = 3
    ncols = int(len(masks) / 3)
    masks_row1, masks_row2, masks_row3 = [], [], []
    for i, ann in enumerate(sorted_anns):
        img_size = (ann['segmentation'].shape[0], ann['segmentation'].shape[1])
        img = np.ones((img_size[0], img_size[1], 4))
        img[:, :, 3] = 0

        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
        img = cv2.rectangle(img, (0,0), (img_size[1],img_size[0]), (0,0,0), 2)
        r, c = get_pos(i, len(masks))
        if r == 0:
            masks_row1.append(img)
        elif r == 1:
            masks_row2.append(img)
        elif r == 2:
            masks_row3.append(img)

    row1_img = hconcat_resize_min(masks_row1)
    row2_img = hconcat_resize_min(masks_row2)
    row3_img = hconcat_resize_min(masks_row3)
    final_img = vconcat_resize_min([row1_img, row2_img, row3_img])

    width_i, height_i = 240,240
    height_img = 3*height_i
    width_img = ncols*width_i
    resize_img = cv2.resize(final_img,(width_img, height_img), interpolation = cv2.INTER_AREA)

    cv2.imshow('masks', resize_img)
    cv2.waitKey(0)

# c = 0
# rows = 2
# ncols = int(len(masks) / 2)
# fig, axs = plt.subplots(rows, ncols + 1, figsize=(30, 20))
# for i, ann in enumerate(sorted_anns):
#     img = np.ones((ann['segmentation'].shape[0], ann['segmentation'].shape[1], 4))
#     img[:,:,3] = 0
#
#     r,c = get_pos(i, len(masks))
#     print(r,c)
#     m = ann['segmentation']
#     color_mask = np.concatenate([np.random.random(3), [0.35]])
#     img[m] = color_mask
#     axs[r,c].imshow(img)
#
# if c < ncols:
#     axs[1,ncols].plot(np.array(None))