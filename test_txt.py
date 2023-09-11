# sam_failures_path = '/home/scasao/SEPARA/unizar_DL4HSI/separa/data/Masks/labeling_failures.txt'
# with open(sam_failures_path, 'a+') as file_failures:
#     file_lines = [line.rstrip() for line in file_failures]
#
# files_sam_failures = file_lines + ['a', 'b', 'c']
# with open(sam_failures_path, "w") as f:
#     for line in files_sam_failures:
#         f.write(line + "\n")
# f.close()
###################################################33

import json
path_ann = '/home/scasao/Downloads/annotations_trainval2017/annotations/instances_train2017.json'
ann = json.load(open(path_ann, 'r'))
print('p')
