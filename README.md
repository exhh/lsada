# Low-Resource Adversarial Domain Adaptation
This code implements a low-resource advaersarial domain adaptation method [1] for cross-domain cell/nuclei detection in digital pathology and microscopy images, when target training data is limited. The code is written with PyTorch (version 0.4.1) on a Ubuntu Linux machine.

**Training:** 
Step 1: Given the gold standard annotations of cells/nuclei in each traing image, generate the correpsonding proximity map for each image by following the description in [1].

Step 2: Train the source model: ./train_source.sh

Step 3: Train the low-resource adversrial domain adaptation model (specify proper paths for the source model and datasets): ./train.sh


**Testing:**
Given new target images, run the script for model predition (specify proper paths for models and datasets): ./test.sh
