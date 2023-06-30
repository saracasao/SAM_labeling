import os
import re
import cv2
import json
import numpy as np

from datetime import date
from utils import get_mask_from_images, binary_mask_to_polygon, binary_mask_to_bbox, get_image_from_masks


class CocoFormatSEPARA:
    def __init__(self, dir_dataset, masks=True, dir_previous_annotations=None):
        self.dir_data = dir_dataset
        self.masks = masks
        if self.masks:
            self.dir_masks = dir_dataset + '/Masks'

        # ---------------------General label references in the annotated data----------------------------------------
        self.SEPARA_NUM_LABELS = 6
        self.SEPARA_LABELS = {"BACKGROUND": 0,
                              "FILM": 1,
                              "BASKET": 2,
                              "CARDBOARD": 3,
                              "VHS": 4,
                              "THREADLIKE": 5}

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

        self.categories = [{'id': 0, 'name': 'background'}, # background = the rest of the object
                           {'id': 1, 'name': 'film'},
                           {'id': 2, 'name': 'basket'},
                           {'id': 3, 'name': 'cardboard'},
                           {'id': 4, 'name': 'vhs'},
                           {'id': 5, 'name': 'threadlike'}]

        self.licenses = [{'id': 0,
                          'name': 'No known copyright restrictions',
                          'url': ''}]

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
            self.set_dataset_labels()
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
        self.annotation_id = len(self.annotations)

    def get_image_id(self):
        images_id = [f['id'] for f in self.images]
        images_id = sorted(images_id)
        return images_id

    def get_name_files_original_as_ref(self, files='all', range_images=None, starting_image=None):
        all_name_images = [f for f in sorted(os.listdir(self.dir_data)) if ('.png' in f or '.jpg' in f)]
        all_name_masks = [f for f in sorted(os.listdir(self.dir_masks)) if ('.png' in f or '.jpg' in f)]

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
        all_name_images = [f for f in sorted(os.listdir(self.dir_data)) if ('.png' in f or '.jpg' in f)]
        all_name_masks = [f for f in sorted(os.listdir(self.dir_masks)) if ('.png' in f or '.jpg' in f)]

        if files == 'start_from':
            assert starting_masks is not None, "Error in argument, range must indicate the start file"
            idx_file_start = all_name_masks.index(starting_masks)
            name_masks = all_name_masks[idx_file_start:]
            name_images = get_image_from_masks(name_masks, all_name_images)
        elif files == 'range':
            assert range_masks is not None and len(range_masks) == 2, "Error in argument, range must indicate the start and end file, i.e., len 2"
            file_start = range_masks[0]
            file_end = range_masks[1]

            idx_file_start = all_name_masks.index(file_start)
            idx_file_end = all_name_masks.index(file_end)

            name_masks = all_name_masks[idx_file_start:idx_file_end + 1]
            name_images = get_image_from_masks(name_masks, all_name_images)
        else:
            name_masks = all_name_masks
            name_images = get_image_from_masks(all_name_masks, all_name_images)
        return name_images, name_masks

    def create_image_info(self, image_id, file_name, width, height):
        image_info = {"id": image_id,
                      "file_name": file_name,
                      "width": width,
                      "height": height,
                      "date_captured": self.info['date_created'],
                      "license": 0}
        return image_info

    def create_annotation_info(self, mask, name_file_mask, id_annotation, image_id):
        _, segmentation = binary_mask_to_polygon(mask, tolerance=1)
        bbox = binary_mask_to_bbox(mask)

        category_str = name_file_mask.split('_')[3]
        assert 'L' in category_str, "Error in component selected {}".format(category_str)
        category_str = category_str[4]

        assert int(category_str) in self.SEPARA_LABELS.values()
        if not self.masks:
            segmentation = []
        # Save new info
        annotation_info = {'id': id_annotation,
                           'image_id': image_id,
                           'category_id': int(category_str),
                           'segmentation': segmentation,
                           'bbox': bbox}
        return annotation_info

    def generate_coco_labels(self): # TODO if does not exist name define empty label
        # name_images, name_masks = self.get_name_files_original_as_ref() # all image are labeled
        name_images, name_masks = self.get_name_files_masks_as_ref()
        name_image_wo_ext = [f.split('.')[0] for f in name_images]

        for n_mask in name_masks:
            img_id = re.search('Img_(.*)_L', n_mask).group(1)
            idx_name_image = name_image_wo_ext.index(img_id)
            name_img = name_images[idx_name_image]

            dir_file_image = self.dir_data + '/' + name_img
            dir_file_mask = self.dir_masks + '/' + n_mask
            mask = cv2.imread(dir_file_mask, 0)

            images_id = self.get_image_id()
            if img_id not in images_id:
                image_info = self.create_image_info(img_id, dir_file_image, mask.shape[1], mask.shape[0])
                self.images.append(image_info)
            annotation_info = self.create_annotation_info(mask, n_mask, self.annotation_id, img_id)
            self.annotations.append(annotation_info)
            self.annotation_id += 1


if __name__ == '__main__':
    dir_images = '/home/scasao/Documents/TEST_DATASET/SEPARA/WD_BLACK/2022-05-10/DALSA'
    masks = True

    coco_labels = CocoFormatSEPARA(dir_images, masks, dir_previous_annotations='/home/scasao/Documents/TEST_DATASET/SEPARA/all_annotations.json')
    coco_labels.generate_coco_labels()












