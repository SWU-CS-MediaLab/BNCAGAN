## Image Enhancement with Bi-directional Normalization and Color Attention-guided GAN

## This paper is submitted to Knowledge-Based Systems, and the extended and updated version of BNCAGAN will be released when it is accepted by journal of Knowledge-Based Systems.

The old version of BNCAGAN can be used as following:

## Requirements and Installation
We recommended the following dependencies.
*  Python 3.7
*  PyTorch 1.7.1
*  tqdm 4.42.1
*  munch 2.5.0
*  torchvision 0.8.2

## Training
Prepare the training, testing, and validation data. The folder structure should be:
```
data
└─── fiveK
	├─── train
	|	├─── exp
	|	|	├──── a1.png                  
	|	|	└──── ......
	|	└─── raw
	|		├──── b1.png                  
	|		└──── ......
	├─── val
	|	├─── label
	|	|	├──── c1.png                  
	|	|	└──── ......
	|	└─── raw
	|		├──── c1.png                  
	|		└──── ......
	└─── test
		├─── label
		| 	├──── d1.png                  
		| 	└──── ......
		└─── raw
			├──── d1.png                  
			└──── ......
```
```raw/```contains low-quality images, ```exp/``` contains unpaired high-quality images, and ```label/``` contains corresponding ground truth.

To train BNCAGAN on FiveK, run the training script below.
```
python main.py --mode train --version BNCAGAN-FiveK --use_tensorboard False \
--is_test_nima True --is_test_psnr_ssim True
```

To test BNCAGAN on FiveK, run the test script below.
```
python main.py --mode test --version BNCAGAN-FiveK --pretrained_model xx (best epoch, e.g., 100) \
--is_test_nima True --is_test_psnr_ssim True

```



