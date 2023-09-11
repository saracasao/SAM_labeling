import json
import cv2

color = (0,0,255)


def load_path_images(info_files_path):
    file = open(info_files_path, 'r')
    file_lines = file.readlines()
    dir_images = [f.strip() for f in file_lines]

    return dir_images


def load_files_annotations(path_annotations):
    # LOAD EXISTING ANNOTATIONS
    data = json.load(open(path_annotations, 'r'))
    return data


def draw_ref_annotations(img, name_file, annotations):
    ann = annotations[name_file]

    # Drawing variables
    width_label, height_label = 10, 10
    for a in ann:
        label = a['label']
        bbox = a['bbox']
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        width_bbox = int(bbox[2])
        height_bbox = int(bbox[3])

        y_min_label = ymin + int(height_bbox / 2)
        bbox_label = [xmin + width_bbox, y_min_label, xmin + width_bbox + width_label, y_min_label + height_label]

        cv2.rectangle(img, (xmin,ymin), (xmin + width_bbox, ymin + height_bbox), color, 2)
        cv2.putText(img, label, (bbox_label[0], bbox_label[1] + int(height_label / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img
