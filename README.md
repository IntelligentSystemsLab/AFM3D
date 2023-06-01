# AFM3D
This is the Pytorch code of the paper entitled "AFM3D: An Asynchronous Federated Meta-learning Framework for Driver Distraction Detection".


#AUC-preprocess-train.py#
Preprocessing training data of AUC dataset.

#AUC-preprocess-test.py#
Preprocessing testing data of AUC dataset.

#SFD-preprocess.py#
Preprocessing data of SFD dataset.

#main_AFM3DTW_SFD_densenet121.py#
Running this main file to use AFM3DTW train densenet121 in SFD dataset.

#learner_AFM3DTW_SFD_densenet121.py#
Used by 'main_AFM3DTW_SFD_densenet121.py'. Note that you can change models, datasets, and methods here conveniently.

#client_AFM3DTW_SFD_densenet121.py#
Used by 'learner_AFM3DTW_SFD_densenet121.py'. Note that various training functions are provided here.
