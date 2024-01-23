import os
import json
import torch
import pickle
import pandas as pd
import seaborn as sns
import cv2
import numpy as np
import imageio.v3 as imageio
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from image_loader import loader_separa_data_unsupervised, loader_separa_data, get_path_hyperspectral
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import cluster
from mycolorpy import colorlist as mcp
from PIL import Image, ImageDraw
from utils import binary_mask_to_bbox, get_existing_masks
import matplotlib.patches as mpatches

path_to_save_obj = '/home/scasao/SEPARA/unizar_DL4HSI/separa/unsupervised/objects/'

SEPARA_LABELS_COLORS = {
    0: '#000000', #black
    1: '#daf706', #yellow
    2: '#33ddff', #cyan
    3: '#3432dd', #darkblue
    4: '#ca98c3', #pink
    5: '#008000', #green
    6: '#ffa500', #orange
    7: '#c12869', #darkpink
}


SEPARA_LABELS = {"FILM": 1,
                 "BARQUILLA": 2,
                 "CARTON": 3,
                 "CINTA_VIDEO": 4,
                 "ELEMENTOS_FILIFORMES": 5,
                 "BOLSA": 6}


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def get_segmentations():
    device = "cuda"
    model_type = "vit_h"
    sam_checkpoint = "/home/scasao/SAM/segment-anything/checkpoints/sam_vit_h_4b8939.pth"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Load path images
    path_annotations = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/final_annotations_2023-11-10.json'
    image_names, annotations_image_to_label = loader_separa_data_unsupervised(path_annotations, only_labeled=True)

    print('Number of images for labeling', len(image_names))
    for i, name_img in enumerate(image_names):
        if i % 20 == 0:
            print('img {}'.format(i))

        key_img = Path(name_img).stem
        image = cv2.imread(name_img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        # GET MASKS
        masks = mask_generator.generate(image)
        masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        for j, mask in enumerate(masks):
            seg = 255*mask['segmentation']
            bbox = mask['bbox'] #XYWH
            x, y, w, h = bbox
            seg = seg.astype(np.uint8)

            res = cv2.bitwise_and(image,image,mask = seg)
            obj_cropped = res[y:y+h, x:x+w]

            n_obj = str(j)
            n_obj.zfill(3)
            path_obj = path_to_save_obj + str(key_img) + '_N' + str(n_obj) + '.png'
            cv2.imwrite(path_obj, obj_cropped)
        # plt.figure(figsize=(20, 20))
        # plt.imshow(image)
        # show_anns(masks)
        # plt.axis('off')
        # plt.show()


def get_colors(n_clusters):
    colors_hsv = mcp.gen_color(cmap="hsv", n=n_clusters)
    color_dict = {}
    for i in range(n_clusters):
        color_dict[i] = colors_hsv[i]
        # color_dict[i] = tuple(np.random.choice(range(256), size=3))

    # color_dict = dict({0:'red',1:'firebrick',2:'tomato',3:'green',4:'lawngreen',5:'lightgreen',6:'blue',7:'darkblue',8:'aquamarine',
    #      9:'black',10:'gray',11:'silver',12:'yellow',13:'khaki',14:'gold',
    #      15:'magenta',16:'pink',17:'deeppink',18:'darkviolet',19:'blueviolet',20:'rebeccapurple'})
    return color_dict


def TSNE_points(tsne_pca_results, all_ids, name_file):
    plt.figure()

    x = tsne_pca_results[:, 0]
    y = tsne_pca_results[:, 1]

    x_n, y_n, ids_n = [], [], []
    for q_id, x_i, y_i in zip(all_ids, x, y):
        x_n.append(x_i)
        y_n.append((-1) * y_i)
        ids_n.append(q_id)
    cluster = {'pos_x': x_n, 'pos_y': y_n}
    cluster['y'] = ids_n

    df = pd.DataFrame(data=cluster)
    sns.scatterplot(x='pos_x', y='pos_y',
                        hue='y',
                        palette=SEPARA_LABELS_COLORS,
                        data=df,
                        legend=False,
                        alpha=0.7)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('/home/scasao/SEPARA/unizar_DL4HSI/separa/unsupervised/' + name_file, dpi=300)



def preprocess_img(img, color):
    coord = [(0,0),(img.size[0],0),(img.size[0],img.size[1]),(0,img.size[1])]
    draw = ImageDraw.Draw(img)
    for i in range(len(coord)):
        if i < len(coord)-1:
            p = (coord[i], coord[i+1])
        else:
            p = (coord[i], coord[0])
        draw.line(p, fill = color, width = 20)
    return img


def TSNE_images_clusters(tsne_pca_results, dir_images, all_ids, n_clusters, name_file, load_images = True):
    colors = get_colors(n_clusters)
    tx, ty = tsne_pca_results[:, 0], tsne_pca_results[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    width = 4000
    height = 3000
    max_dim = 100

    full_image = Image.new('RGBA', (width, height))
    i = 0
    for img, x, y in zip(dir_images, tx, ty):
        if load_images:
            tile = Image.open(img)
        else:
            tile = Image.fromarray(img)
        pid = all_ids[i]
        color = SEPARA_LABELS_COLORS[pid]
        tile = preprocess_img(tile, color)

        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize((int(tile.width / rs), int(tile.height / rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width - max_dim) * x), int((height - max_dim) * y)), mask=tile.convert('RGBA'))
        i += 1

    # legend = []
    # for i, v in SEPARA_LABELS.items():
    #     c = SEPARA_LABELS_COLORS[v]
    #     patch = mpatches.Patch(color=c, label=i)
    #     legend.append(patch)
    #
    # plt.imshow(full_image)
    # plt.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, -0.05), prop={'size':5}, fancybox=True, ncol=6, columnspacing=1)
    # plt.axis('off')
    # plt.savefig('/home/scasao/SEPARA/unizar_DL4HSI/separa/unsupervised/sam_annotated.png', dpi =1200)
    full_image.save('/home/scasao/SEPARA/unizar_DL4HSI/separa/unsupervised/' + name_file)


def TSNE_images(tsne_pca_results, dir_images, load_images = True):
    tx, ty = tsne_pca_results[:, 0], tsne_pca_results[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    width = 4000
    height = 3000
    max_dim = 100

    full_image = Image.new('RGBA', (width, height))
    i = 0
    for img, x, y in zip(dir_images, tx, ty):
        if load_images:
            tile = Image.open(img)
        else:
            tile = Image.fromarray(img)
            # tile = img
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize((int(tile.width / rs), int(tile.height / rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width - max_dim) * x), int((height - max_dim) * y)), mask=tile.convert('RGBA'))
        i += 1
    full_image.save('/home/scasao/SEPARA/unizar_DL4HSI/separa/unsupervised/sam_labels.png')


def dims_reduction_pca(pca_model, img):
    height, width, bands = img.shape

    features = img.reshape(-1, bands)
    hyper_reduction = pca_model.transform(features)
    hyper_reduction = hyper_reduction.reshape(height, width, 3)
    hyper_reduction /= 65536
    hyper_reduction = np.clip(hyper_reduction, 0, 1)

    hyper_reduction = 255 * hyper_reduction
    return hyper_reduction.astype(np.uint8)


def clustering_annotated_objects_hyper():
    print('Loading labeled objects...')

    device = "cuda"
    model_type = "vit_h"
    sam_checkpoint = "/home/scasao/SAM/segment-anything/checkpoints/sam_vit_h_4b8939.pth"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    model = SamPredictor(sam)
    pca_model = pickle.load(open('/home/scasao/SEPARA/unizar_DL4HSI/separa/data/disk/preprocessed_data/Downsampling_bigObjects_annotations_PCA3/pca_fit_3.pkl', 'rb'))

    # Load path images
    path_annotations = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/final_hyper_transfer_annotations_2023-11-10.json'
    path_images, annotations = loader_separa_data_unsupervised(path_annotations, only_labeled=True, include_labels=True)

    labels, data, list_images = [], [], []
    for i, path_img in enumerate(path_images):
        if i % 20 == 0:
            print(i)
        image_hyp = imageio.imread(path_img).transpose(1, 2, 0)
        image = dims_reduction_pca(pca_model, image_hyp)
        image = cv2.resize(image, (1184, 1200))

        existing_masks, ann_label = get_existing_masks(path_img, annotations, labels_included=True)
        if existing_masks is not None:
            for j, mask_bool in enumerate(existing_masks):
                binary_mask = 255*mask_bool
                binary_mask = binary_mask.astype(np.uint8)

                x,y,w,h = binary_mask_to_bbox(binary_mask)
                res = cv2.bitwise_and(image,image,mask = binary_mask)

                obj_cropped = res[y:y+h, x:x+w]

                model.set_image(obj_cropped)
                embbeding = model.get_image_embedding().cpu().numpy()
                emb_flatten = embbeding.reshape(-1)

                # MEDIAN EMBBEDING
                # emb_flatten = embbeding.reshape(256,-1)
                # median_arr = np.median(emb_flatten, axis=1)

                labels.append(ann_label[j])
                data.append(emb_flatten)
                list_images.append(obj_cropped)
    print(np.unique(labels))
    data_raw = np.array(data)

    # pca
    print('fitting PCA...')
    pca = PCA(100)
    data_dims_red = pca.fit_transform(data_raw)

    # tsne
    print('fitting TSNE...')
    model = TSNE(n_components=2, random_state=0)
    tsne_pca_data = model.fit_transform(data_dims_red)

    print('VISUALIZATION')
    TSNE_images_clusters(tsne_pca_data, list_images, labels, 7,name_file='hyper_all_feat.png', load_images=False)
    TSNE_points(tsne_pca_data, labels, name_file='hyper_all_feat_points.png')


def clustering_annotated_objects():
    print('Loading labeled objects...')

    device = "cuda"
    model_type = "vit_h"
    sam_checkpoint = "/home/scasao/SAM/segment-anything/checkpoints/sam_vit_h_4b8939.pth"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    model = SamPredictor(sam)

    # Load path images
    path_annotations = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/final_annotations_2023-11-10.json'
    path_images, annotations = loader_separa_data_unsupervised(path_annotations, only_labeled=True, include_labels=True)

    labels, data, list_images = [], [], []
    for i, path_img_rgb in enumerate(path_images):
        if i % 20 == 0:
            print(i)
        image_rgb = cv2.imread(path_img_rgb)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        existing_masks, ann_label = get_existing_masks(path_img_rgb, annotations, labels_included=True)
        if existing_masks is not None:
            for j, mask_bool in enumerate(existing_masks):
                binary_mask = 255*mask_bool
                binary_mask = binary_mask.astype(np.uint8)

                x,y,w,h = binary_mask_to_bbox(binary_mask)
                res = cv2.bitwise_and(image_rgb,image_rgb,mask = binary_mask)

                obj_cropped = res[y:y+h, x:x+w]

                model.set_image(obj_cropped)
                embbeding = model.get_image_embedding().cpu().numpy()
                emb_flatten = embbeding.reshape(-1)

                # MEDIAN EMBBEDING
                # emb_flatten = embbeding.reshape(256,-1)
                # median_arr = np.median(emb_flatten, axis=1)

                labels.append(ann_label[j])
                data.append(emb_flatten)
                list_images.append(obj_cropped)
    print(np.unique(labels))
    data_raw = np.array(data)

    # pca
    print('fitting PCA...')
    pca = PCA(100)
    data_dims_red = pca.fit_transform(data_raw)

    # tsne
    print('fitting TSNE...')
    model = TSNE(n_components=2, random_state=0)
    tsne_pca_data = model.fit_transform(data_dims_red)

    print('VISUALIZATION')
    TSNE_images_clusters(tsne_pca_data, list_images, labels, 7, name_file='rgb_all_feat.png', load_images=False)
    TSNE_points(tsne_pca_data, labels, name_file='rgb_all_feat_points.png',)


def clustering_unsupervised(kmeans=False):
    N_clusters = 100
    device = "cuda"
    model_type = "vit_h"
    sam_checkpoint = "/home/scasao/SAM/segment-anything/checkpoints/sam_vit_h_4b8939.pth"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    model = SamPredictor(sam)

    list_images_name = os.listdir(path_to_save_obj)
    list_dir_images = [path_to_save_obj + n for n in list_images_name][0:5000]

    data = [] # X
    for i, d_img in enumerate(list_dir_images):
        if i % 20 == 0:
            print(i)
        img = cv2.imread(d_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        model.set_image(img)
        embbeding = model.get_image_embedding().cpu().numpy()
        emb_flatten = embbeding.reshape(256,4096)
        median_arr = np.median(emb_flatten, axis=1)
        data.append(median_arr)
    if kmeans:
        # assign clusters
        print('fitting kmeans...')
        data_raw = np.array(data)
        k_means = cluster.KMeans(N_clusters)
        clusters = k_means.fit_predict(data_raw) # Y
    else:
        data_raw = np.array(data)

    # pca
    print('fitting PCA...')
    pca = PCA(100)
    data_dims_red = pca.fit_transform(data_raw)

    # tsne
    print('fitting TSNE...')
    model = TSNE(n_components=2, random_state=0)
    tsne_pca_data = model.fit_transform(data_dims_red)

    print('VISUALIZATION')
    if kmeans:
        TSNE_images_clusters(tsne_pca_data, list_dir_images, clusters, N_clusters)
    else:
        TSNE_images(tsne_pca_data, list_dir_images)


if __name__ == '__main__':
    # clustering_unsupervised(True)
    clustering_annotated_objects()
    clustering_annotated_objects_hyper()