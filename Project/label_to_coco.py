from CocoFormat import CocoFormatSEPARA
from datetime import date


def main(name_dataset):
    current_date = date.today()
    current_date = current_date.strftime("%Y-%m-%d_")
    if name_dataset == 'SEPARA':
        dir_images = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/disk/'
        dir_masks = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/annotations_25_10/Masks_unify_processed/'

        name_file = current_date + 'hyper_annotations_rle.json'

        # coco_labels = CocoFormatSEPARA(dir_images, dir_masks, dir_previous_annotations='/home/scasao/SEPARA/unizar_DL4HSI/separa/data/annotations_update/2023-09-27_all_annotations_rle.json')
        coco_labels = CocoFormatSEPARA(dir_images, dir_masks)
        coco_labels.init_new_annotations()
        coco_labels.generate_new_coco_labels()

        # coco_labels.save_annotations('/home/scasao/SEPARA/unizar_DL4HSI/separa/data/annotations_update/' + name_file)
        # print('p')


if __name__ == '__main__':
    main('SEPARA')

