# LABEL WITH SAM AS ASSINTANT

The python file sam_labeling.py is recommended for labeling easy and common objects such as bottles, trees, cars etc. The code asks you for the mode of labeling (points/bounding boxes) and the label of the mask.

For more challenging and uncommon data, the file sam_labeling_challenge.py allows an interactive labeling process. It only works with the points mode and allows to refine the masks by defining which pixels do and do not belong to the mask. 

More concrete instructions are included at the beginning of the codes. The output of both codes is a .jpg file with the binary mask.

SAM's information and repository
[SAM web](https://segment-anything.com/) 

[SAM repo](https://github.com/facebookresearch/segment-anything)
