from CocoFormat import CocoFormatSEPARA, CocoFormatLOGOS
from datetime import date


def main(name_dataset):
    current_date = date.today()
    current_date = current_date.strftime("%Y-%m-%d_")
    if name_dataset == 'SEPARA':
        dir_images = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/disk/2023-06-09'
        dir_masks = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/Masks/2023-06-09'

        name_file = current_date + 'all_annotations_rle.json'

        coco_labels = CocoFormatSEPARA(dir_images, dir_masks, dir_previous_annotations='/home/scasao/SEPARA/unizar_DL4HSI/separa/data/annotations_update/2023-07-31_all_annotations_rle.json')
        coco_labels.init_new_annotations()
        coco_labels.generate_new_coco_labels()

        coco_labels.save_annotations('/home/scasao/SEPARA/unizar_DL4HSI/separa/data/annotations_update/' + name_file)

    elif name_dataset == 'LOGOS':
        dir_files = '/home/scasao/SAM/Detector_Logos/dir_files.txt' # paths of the images labeled
        dir_bbox_annotations = '/home/scasao/SAM/Detector_Logos/annotations.json'

        name_file = current_date + 'annotations_logos.json'

        coco_labels = CocoFormatLOGOS(dir_files, dir_bbox_annotations)
        coco_labels.init_new_annotations()
        coco_labels.from_bbox_to_coco()

        coco_labels.save_annotations('/home/scasao/SAM/Detector_Logos/' + name_file)


if __name__ == '__main__':
    main('LOGOS')

