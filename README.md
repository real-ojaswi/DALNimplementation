# DALNimplementation
This is the only full implementation of [Reusing the Task-Specific Classifier as a Discriminator: Discriminator-free Adversarial Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Reusing_the_Task-Specific_Classifier_as_a_Discriminator_Discriminator-Free_Adversarial_Domain_CVPR_2022_paper.html) The implementation is performed on TensorFlow.

The 'DALNCustom.ipynb' shows experiments on MNIST and USPS dataset. The model is trained on MNIST and its accuracy is checked on USPS dataset with and without Domain Adaptation. The 'grl.py' and 'nwd.py' execute gradient reversal layer and nuclear wasserstein discrepancy, and the code has been adapted from the [official implementation repository] (https://github.com/xiaoachen98/DALN) which provides the code for PyTorch. 'DALNModel.py' has the code for the model. Other files have codes to aid training process and experimentation.

The results shows significant increase in accuracy when its trained using DALN.

|Method|Source accuracy|Target accuracy|
|-----|----------------|---------------|
|Source only|0.9998|0.417|
|DALN|0.9341|0.838|
