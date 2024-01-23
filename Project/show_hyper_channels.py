import cv2
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np


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


path_img = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/disk/2022-10-11/23/Specim/20221011_11-19-54.tiff'
image_hyp = imageio.imread(path_img).transpose(1, 2, 0)
image_hyp = dims_reduction_random_bands(image_hyp)
cv2.imshow('img', image_hyp)
cv2.waitKey(0)

# px_vhs_hyp = image_hyp[374,382]
# x = list(range(len(px_vhs_hyp)))
# plt.figure()
# plt.plot(x, px_vhs_hyp)
# plt.show()


