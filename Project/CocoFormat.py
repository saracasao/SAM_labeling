import os
import re
import cv2
import copy
import json
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import date
from torchvision import transforms
from utils import get_path_file, get_common_data_by_keys, get_mask_from_images, binary_mask_to_polygon, binary_mask_to_bbox, get_image_from_masks, mask_to_rle_pytorch, rle_to_mask


class CocoFormatSEPARA:
    def __init__(self, dir_dataset, dir_masks, dir_previous_annotations=None):
        self.dir_data = dir_dataset
        self.dir_masks = dir_masks

        # ---------------------General label references in the annotated data----------------------------------------
        self.SEPARA_NUM_LABELS = 6
        self.SEPARA_LABELS_E = {"BACKGROUND": 0,
                                "FILM": 1,
                                "BASKET": 2,
                                "CARDBOARD": 3,
                                "VHS": 4,
                                "THREADLIKE": 5}

        self.SEPARA_LABELS = {"FONDO": 0,
                              "FILM": 1,
                              "BARQUILLA": 2,
                              "CARTON": 3,
                              "CINTA_VIDEO": 4,
                              "ELEMENTOS_FILIFORMES": 5,
                              "BOLSA": 6,
                              "ELECTRONICA": 7}

        self.SEPARA_LABELS_COLORS = {0: '#000000',
                                     1: '#daf706',
                                     2: '#33ddff',
                                     3: '#3432dd',
                                     4: '#ca98c3',
                                     5: '#3068df'}

        # ----------------------Required dictionaries for COCO labeling format------------------------------------
        self.info = {'description': 'SEPARA Dataset',
                     'url': '',
                     'version': '',
                     'year': '2023',
                     'date_created': ''}

        self.categories = [{'id': 0, 'name': 'FONDO'},  # background = the rest of the object
                           {'id': 1, 'name': 'FILM'},
                           {'id': 2, 'name': 'BARQUILLA'},
                           {'id': 3, 'name': 'CARTON'},
                           {'id': 4, 'name': 'CINTA_VIDEO'},
                           {'id': 5, 'name': 'ELEMENTOS_FILIFORMES'},
                           {'id': 6, 'name': 'BOLSA'},
                           {'id': 7, 'name': 'ELECTRONICA'}]


        self.categories_e = [{'id': 0, 'name': 'background'}, # background = the rest of the object
                             {'id': 1, 'name': 'film'},
                             {'id': 2, 'name': 'basket'},
                             {'id': 3, 'name': 'cardboard'},
                             {'id': 4, 'name': 'vhs'},
                             {'id': 5, 'name': 'threadlike'}]

        self.licenses = [{'id': 0,
                          'name': 'No known copyright restrictions',
                          'url': ''}]

        self.convert_tensor = transforms.PILToTensor()
        self.images = []
        self.images_keys = ['id', 'file_name', 'width', 'height', 'data_captured']

        self.annotation_id = 0
        self.annotations = []
        self.annotations_keys = ['id', # annotation id -> each annotation is unique
                                 'image_id', # id of the labeled image
                                 'category_id', # int label (defined in category dict)
                                 'segmentation', # coordinates of the polynom
                                 'bbox'] # bbox coordinates [xmin, ymin, width, height]
        if dir_previous_annotations is not None:
            self.dir_annotation_file = dir_previous_annotations
            self.previous_annotations = self.set_dataset_labels()
        else:
            self.init_new_annotations()

    def init_new_annotations(self):
        current_date = date.today()
        self.info['date_created'] = current_date.strftime("%Y/%m/%d/")

    def set_dataset_labels(self):
        with open(self.dir_annotation_file) as f:
            data = json.load(f)

        # Update version
        current_date = date.today()
        data['info']['version'] = current_date.strftime("%Y/%m/%d/")
        self.images = data['images']
        self.annotations = data['annotations']
        self.annotation_id = self.annotations[-1]['id'] + 1
        return data

    def get_image_id(self):
        images_id = [f['id'] for f in self.images]
        images_id = sorted(images_id)
        return images_id

    def get_files_dir_from_path(self):
        """Load the masks and look for the corresponding images in the global folder of the dataset"""
        # Get all the new masks
        dir_masks_png = list(Path(self.dir_masks).rglob("*.png")) # check extension of the masks
        dir_masks_jpg = list(Path(self.dir_masks).rglob("*.jpg")) # check extension of the masks

        list_dir_masks  = dir_masks_png + dir_masks_jpg

        # Get all the images corresponding to the masks
        list_dir_images = []
        for d_mask in list_dir_masks:
            d_mask_str = str(d_mask)
            img_id = re.search('Img_(.*)_L', d_mask_str).group(1)
            subfolder = os.path.join(*d_mask_str.split('/')[-4:-1])
            dir_image_png = self.dir_data + subfolder + '/' + img_id + '.jpg'
            assert os.path.isfile(dir_image_png)
            list_dir_images.append(Path(dir_image_png))

        # Get info images and masks as str
        info_image_to_label = [(str(f), str(f.stem)) for i, f in enumerate(list_dir_images)]
        list_dir_masks = list(map(str, list_dir_masks))

        return sorted(info_image_to_label), sorted(list_dir_masks)

    def get_files_dir_from_path_adding_new_files(self):
        """Add new images to the annotation files and needs to be checked if hyper exists or not.
           All the image and masks are in a unique subfolder"""
        # Get all the new masks
        dir_masks_png = list(Path(self.dir_masks).rglob("*.png")) # check extension of the masks
        dir_masks_jpg = list(Path(self.dir_masks).rglob("*.jpg")) # check extension of the masks

        list_dir_masks  = dir_masks_png + dir_masks_jpg

        # Get all the images in the folder
        dir_image_png = list(Path(self.dir_data).rglob("*.png")) # check extension of the masks
        dir_image_jpg = list(Path(self.dir_data).rglob("*.jpg")) # check extension of the masks
        list_dir_images = dir_image_png + dir_image_jpg

        # Get all the hyper images in the folder
        dir_hyper = list(Path(self.dir_data).rglob("*.tiff")) # check extension of the hyper
        name_files_hyper = [n.stem for n in dir_hyper]
        name_files_image = [n.stem for n in list_dir_images]
        keys_in_common_rgb_hyper = list(set(name_files_hyper) & set(name_files_image)) # check files in common

        info_image_to_label, list_dir_masks = get_common_data_by_keys(keys_in_common_rgb_hyper, list_dir_masks, list_dir_images, name_files_image)
        return sorted(info_image_to_label), sorted(list_dir_masks)

    def get_name_files_original_as_ref(self, files='all', range_images=None, starting_image=None):
        all_name_images = [f for f in sorted(os.listdir(self.dir_data)) if ('.png' in f or '.jpg' in f)] # TODO change to find the file paths
        all_name_masks = [f for f in sorted(os.listdir(self.dir_masks)) if ('.png' in f or '.jpg' in f)] # TODO change to find the file paths

        if files == 'start_from':
            assert starting_image is not None, "Error in argument, range must indicate the start file"
            idx_file_start = all_name_images.index(starting_image)
            name_images = all_name_images[idx_file_start:]

            name_masks = get_mask_from_images(name_images, all_name_masks)
        elif files == 'range':
            assert range_images is not None and len(range_images) == 2, "Error in argument, range must indicate the start and end file, i.e., len 2"
            file_start = range_images[0]
            file_end = range_images[1]

            idx_file_start = all_name_images.index(file_start)
            idx_file_end = all_name_images.index(file_end)
            name_images = all_name_images[idx_file_start:idx_file_end + 1]
            name_masks = get_mask_from_images(name_images, all_name_masks)
        else:
            name_images = all_name_images
            name_masks = all_name_masks
        return name_images, name_masks

    def get_name_files_masks_as_ref(self, files='all', range_masks=None, starting_masks=None):
        info_images, list_dir_masks = self.get_files_dir_from_path()
        if files == 'start_from':
            assert starting_masks is not None, "Error in argument, range must indicate the start file"
            idx_file_start = list_dir_masks.index(starting_masks)
            dir_masks = list_dir_masks[idx_file_start:]
            # info_images = get_image_from_masks(dir_masks, all_dir_images)
        elif files == 'range':
            assert range_masks is not None and len(range_masks) == 2, "Error in argument, range must indicate the start and end file, i.e., len 2"
            file_start = range_masks[0]
            file_end = range_masks[1]

            idx_file_start = list_dir_masks.index(file_start)
            idx_file_end = list_dir_masks.index(file_end)

            dir_masks = list_dir_masks[idx_file_start:idx_file_end + 1]
            # info_images = get_image_from_masks(dir_masks, all_dir_images)
        else:
            info_images = info_images
            dir_masks = list_dir_masks
            # info_images = get_image_from_masks(dir_masks, all_dir_images)
        return info_images, dir_masks

    def create_image_info(self, image_id, file_name, width, height, hyper):
        img_dir_split    = file_name.split('/')
        img_dir_standard = img_dir_split[img_dir_split.index('data'):]
        img_dir_standard = os.path.join(*img_dir_standard)

        if hyper:
            extension = Path(img_dir_standard).suffix
            img_dir_standard = img_dir_standard.replace('DALSA', 'Specim')
            img_dir_standard = img_dir_standard.replace(extension,'.tiff')

        image_info = {"id": image_id,
                      "file_name": img_dir_standard,
                      "width": width,
                      "height": height,
                      "date_captured": self.info['date_created'],
                      "license": 0}
        return image_info

    def create_annotation_info(self, mask, name_file_mask, image_id):
        # _, segmentation = binary_mask_to_polygon(mask, tolerance=1)
        # From mask to rle
        mask_tensor = self.convert_tensor(mask)
        segmentation = mask_to_rle_pytorch(mask_tensor)
        # mask = rle_to_mask(segmentation[0])

        bbox = binary_mask_to_bbox(mask)

        category_str = re.search('_L(.*)_N', name_file_mask).group(1)
        category = self.SEPARA_LABELS[category_str]

        # Save new info
        annotation_info = {'id': self.annotation_id,
                           'image_id': image_id,
                           'category_id': category,
                           'segmentation': segmentation,
                           'bbox': bbox}
        self.annotation_id += 1
        return annotation_info

    def merge_vhs_segmentations(self, img_id, mask):
        ann = copy.deepcopy(self.annotations)
        ann_img = [a for a in ann if a['image_id'] == img_id]
        ann_vhs = [a for a in ann if a['image_id'] == img_id and a['category_id'] == 4]

        if len(ann_vhs) > 0:
            merge = True
            mask_arr = np.array(mask)
            bool_mask = mask_arr > 200
            for i, a in enumerate(ann_vhs):
                current_rle_mask = a['segmentation'][0]
                current_mask = rle_to_mask(current_rle_mask)
                bool_mask = np.logical_or(bool_mask, current_mask)
                self.annotations.remove(a)
            ref_mask = bool_mask.astype(np.uint8) * 255
            ref_mask = Image.fromarray(ref_mask)
            mask_tensor = self.convert_tensor(ref_mask)
            segmentation = mask_to_rle_pytorch(mask_tensor)
            bbox = binary_mask_to_bbox(mask)

            new_annotation_info = {'id': self.annotation_id,
                                   'image_id': img_id,
                                   'category_id': 4,
                                   'segmentation': segmentation,
                                   'bbox': bbox}
            self.annotations.append(new_annotation_info)
            self.annotation_id += 1
        else:
            merge = False
        return merge

    def generate_new_coco_labels(self, merge_vhs=False, hyper=False):
        info_images, dir_masks = self.get_name_files_masks_as_ref()
        dir_images, name_images_wo_ext = zip(*info_images)

        for i, d_mask in enumerate(dir_masks):
            print(i)
            label  = re.search('_L(.*)_N', d_mask).group(1)
            img_id = re.search('Img_(.*)_L', d_mask).group(1)
            idx_name_image = name_images_wo_ext.index(img_id)
            dir_img = dir_images[idx_name_image]

            mask = Image.open(d_mask)
            images_id = self.get_image_id()
            if img_id not in images_id:
                image_info = self.create_image_info(img_id, dir_img, mask.size[0], mask.size[1], hyper)
                self.images.append(image_info)

                annotation_info = self.create_annotation_info(mask, d_mask, img_id)
                self.annotations.append(annotation_info)
            elif merge_vhs and label == 'CINTA_VIDEO' and img_id in images_id:
                merge = self.merge_vhs_segmentations(img_id, mask)
                if not merge:
                    annotation_info = self.create_annotation_info(mask, d_mask, img_id)
                    self.annotations.append(annotation_info)
            else:
                annotation_info = self.create_annotation_info(mask, d_mask, img_id)
                self.annotations.append(annotation_info)

    def set_empty_segm_from_bbox(self, format_seg = 'rle'):
        dir_masks = list(Path(self.dir_masks).rglob("*.png")) # check extension of the masks
        dir_masks = [str(d) for d in dir_masks]

        n_segm_annotation = 0
        for d_mask in dir_masks:
            info_annotation = d_mask.split('/')[-1]
            img_id    = re.search('Img_(.*)_L', info_annotation).group(1)
            label_ann = re.search('_L(.*)_N', info_annotation).group(1)
            ann_id    = re.search('_ID(.*).png', info_annotation).group(1)

            ann_id = int(ann_id)
            ann = self.annotations[ann_id]
            if len(ann['segmentation'][0]) == 0:
                assert ann['image_id'] == img_id and ann['id'] == ann_id and ann['category_id'] == self.SEPARA_LABELS[label_ann]
                if format_seg == 'rle':
                    mask = Image.open(d_mask)
                    mask_tensor = self.convert_tensor(mask)
                    segmentation = mask_to_rle_pytorch(mask_tensor)
                elif format_seg == 'points':
                    mask = cv2.imread(d_mask, 0)
                    _, segmentation = binary_mask_to_polygon(mask, tolerance=1)
                else:
                    raise AssertionError('Annotation format does not exist')

                assert type(segmentation) == list
                ann['segmentation'] = segmentation
                self.annotations[ann_id] = ann
                n_segm_annotation += 1
        print(n_segm_annotation, 'segmentation annotations have been added.')

    def reformat_annotations_to_rle(self):
        for ann in self.annotations:
            segmentation = ann['segmentation']
            if type(segmentation[0]) == list and len(segmentation[0]) > 0:
                image_subpath = [img['file_name'] for img in self.images if img['id'] == ann['image_id']][0]
                image_path = self.dir_data + '/' + image_subpath.replace('data', '')
                list_points = segmentation[0]
                points = list(map(int, list_points))
                points = list(zip(points[::2], points[1::2]))

                img = cv2.imread(image_path)
                h, w, c = img.shape

                empty_mask = np.zeros((h, w))
                mask_array = cv2.fillPoly(empty_mask, [np.array(points).astype(np.int32)], color=255)

                mask = Image.fromarray(np.uint8(mask_array))
                mask_tensor = self.convert_tensor(mask)
                segmentation = mask_to_rle_pytorch(mask_tensor)
                assert type(segmentation) == list
                ann['segmentation'] = segmentation
                self.annotations[ann['id']] = ann

    def save_annotations(self, dir_file):
        coco_annotations = {'info' : self.info,
                            'licenses': self.licenses,
                            'categories': self.categories,
                            'images': self.images,
                            'annotations': self.annotations}

        with open(dir_file, 'w') as outfile:
            json.dump(coco_annotations, outfile)
        print('ANNOTATIONS SAVE IN:', dir_file)

