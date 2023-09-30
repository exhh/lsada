# Low-Resource Adversarial Domain Adaptation
This code implements a low-resource advaersarial domain adaptation method [1][2] for cross-domain cell/nuclei detection in digital pathology and microscopy images, when target training data is limited. The code is implemented with PyTorch (version 0.4.1) on a Ubuntu Linux machine. <br /> 


**Training:** <br /> 
Step 1: Given the gold standard annotations of cells/nuclei in each traing image, generate the correpsonding proximity map for each image by following the description in [1][2] and store the images and proximity maps in proper folders.

Step 2: Train the source model (specify a proper path for the dataset): ./train_source.sh

Step 3: Train the low-resource adversrial domain adaptation model (specify proper paths for the source model and datasets): ./train.sh


**Testing:** <br /> 
Given new target images, run the script for model predition (specify proper paths for models and datasets): ./test.sh

<br /> 
Relevant References:<br /> 
[1] F. Xing and T. C. Cornish. Low-Resource Adversarial Domain Adaptation for Cross-Modality Nucleus Detection. Proceedings of International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), vol. 13437, pp. 639–649, 2022. <br />
[2] F. Xing, X. Yang, T. C. Cornish and D. Ghosh. Learning with limited target data to detect cells in cross-modality images. Medical Image Analysis, 2023. <br />
