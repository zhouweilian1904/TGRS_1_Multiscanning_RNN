# TGRS_1_Multiscanning_RNN
The accepted manuscript in TGRS.

"W. Zhou, S. -i. Kamata, Z. Luo and H. Wang, "Multiscanning Strategy-Based Recurrent Neural Network for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-18, 2022, Art no. 5521018, doi: 10.1109/TGRS.2021.3138742." (https://ieeexplore.ieee.org/document/9663304)

**Please kindly cite the papers if this code is useful and helpful for your research.**

Scheme 1
----------------------------------------
![image](https://github.com/zhouweilian1904/TGRS_1_Multiscanning_RNN/blob/main/whole%20framework.jpg)

Scheme 2
----------------------------------------
![image](https://github.com/zhouweilian1904/TGRS_1_Multiscanning_RNN/blob/main/whole%20framework%202.jpg)

The demo of scheme 1 can be checked in *model.py*, named zhouEightDRNN_kamata_LSTM.

**Abstract:**

Most methods based on the convolutional neural network show satisfying performance for hyperspectral image (HSI) classification. However, the spatial dependence among different pixels is not well learned by CNNs. A recurrent neural network (RNN) can effectively establish the dependence of nonadjacent pixels and ensure that each feature activation in its output is an activation at the specific location concerning the whole image, in contrast to the usual local context window in the CNNs. However, recent limited conversion schemes in RNN-based methods for HSI classification cannot fully capture the complete spatial dependence of an HSI patch. In this study, a novel multiscanning strategy with RNN is proposed to feature the sequential character of the HSI pixel and fully consider the spatial dependence in the HSI patch. By investigating different scanning forms, eight scanning orders are considered spatially, which flattens one local HSI patch into eight neighboring continuous pixel sequences. Moreover, considering that eight scanning orders complement one local patch with correlative dependence, the concatenated features from all scanning orders are fed into the RNN again for complementarity. As a result, the network can achieve competitive classification performance on three publicly accessible datasets using fewer parameters than other state-of-the-art methods.

--------------------------------
**Datasets:**

We have uploaded several datasets: https://drive.google.com/drive/folders/1IQxuz4jpwf6goB_2ZwVLEZz1ILScymdO?usp=share_link
1. Indian Pines, 
2. PaviaU, 
3. PaviaC, 
4. Botswana, 
5. Houston 2013, 
6. KSC, 
7. Mississippi_Gulfport, 
8. Salinas, 
9. Simulate_database, 
10. Augmented_IndianPines, 
11. Augmented_Mississippi_Gulfport, 
12. Augmented_PaviaU
13. The disjoint datasets (IP, PU, HU) can be referred in https://github.com/danfenghong/IEEE_TGRS_SpectralFormer.


--------------------------------
**How to use:**

you can find and add some arguments in *main.py* for your own testing.

For example:

python main.py --model zhouEightDRNN_kamata_LSTM  --dataset IndianPines --training_sample 0.1 --cuda 0 --epoch 200 --batch_size 100 --patch_size 9

--------------------------------
**Models:**

In the *model.py*, we have implemented many types of different designs for HSI classification. You can try it with your debug because we are still modifying them. There may exist some mistakes. 

--------------------------------
**Env settings:**

Pytorch:1.11.0

Cuda:11.7

Others: you can direct install the packages through "pip install xxx" or use our *pytorch_new.yml*.
