# SegNet

The algorithm used for segmentation is SegNet (Nov 2015), I followed the instruction of the tutorial on their website to set it up:
[SegNet Tutorial](http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html)

## My set up:

- Ubuntu 14.04
- Architecture: x86_64
- Nvidia GEFORCE 840M (2 Go memory)

## SegNet-Caffe installation

Download the source code from [github](https://github.com/alexgkendall/caffe-segnet)

The steps are for the installation are the same as for Caffe described [here](http://caffe.berkeleyvision.org/install_apt.html)

- Installation of the general dependencies
- BLAS: I installed Atlas
- CUDA: downloaded from this [link](https://developer.nvidia.com/cuda-downloads) and used this [guide](http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Installation_Guide_Linux.pdf) (do the post-installation steps to change PATH variables)

- For SegNet-Caffe you will also need Python dependencies as described in the **Python and/or MATLAB Caffe (optional)** paragraph [here](http://caffe.berkeleyvision.org/installation.html)

- Then proceed with the compilation after editing **Makefile.config** to specify the correct paths for Python

```bash
make all
make test
make runtest
make pycaffe
```

### Troubleshooting

- If you install **boost 1.6**, you will need to patch it (change one line of the code) to compile caffe: [link](https://svn.boost.org/trac/boost/attachment/ticket/11852/float128GccNvcc.patch)

- I got an error on *make runtest* : "Error 127", fixed it by adding "/home/sahbi/miniconda2/lib" to **LD_DIRECTORY_PATH** (I use miniconda for Python)

## Running SegNet

- I was not able to use it with CPU only

- When using GPU, even with a batch size of 1 I got a memory error. Solved it by subsampling the images

- To do so, I use **imagemagick**:

	- make a copy of the original image folder
	- do:

```bash
cd train
mogrify -resize 25% *jpg
```
it resizes to 25% of the original size (*mogrify* replaces the images)

- As described in the tutorial pages you will have to change the paths in *segnet_train.prototxt* and *segnet_inference.prototxt* and *segnet_solver.prototxt*

- In order to generate automatically train.txt and test.txt from the images in folders train and test I wrote a python script in *SegNet/ISBI/getfiles.py* (change the paths if you use it)

- I modified the code in *segnet_train.prototxt* and *segnet_inference.prototxt* to take inputs of size 256x192 (25% of 1022x767) and output 2 classes

- I added the computation of the test error (Jaccard Index) in *test_segmentation_isbi.py*

- **Problem**: images of different sizes in the dataset !

### Steps:

- To train run:

```bash
cd /Programs/caffe-segnet/build/tools # change with your path to caffe-segnet
# run training
./caffe train -gpu 0 -solver /home/sahbi/Projects/SegNet/Models/segnet_solver.prototxt -weights ~/Projects/SegNet/Models/VGG_ILSVRC_16_layers.caffemodel
```

- To test:

```bash
# First
# run test
python /home/sahbi/Projects/SegNet/Scripts/compute_bn_statistics.py /home/sahbi/Projects/SegNet/Models/segnet_train.prototxt /home/sahbi/Projects/SegNet/Models/Training/segnet_iter_300.caffemodel /home/sahbi/Projects/SegNet/Models/Inference/
# Then
# visualize and compute accuracy
python /home/sahbi/Projects/SegNet/Scripts/test_segmentation_isbi.py --model /home/sahbi/Projects/SegNet/Models/segnet_inference.prototxt --weights /home/sahbi/Projects/SegNet/Models/Inference/test_weights.caffemodel
--iter 2 #number of test images to test
```

### Troubleshooting:

- I think that it's also a boost 1.6 bug, but I had to do the following [fix](https://github.com/BVLC/caffe/pull/3575/files)

- In *test_segmentation_isbi.py* I modified *Scripts/test_segmentation_camvid.py* by adding this line on line 39 to be able to run the tests:

```python
label = np.squeeze(label[0,:,:,:])
```

- If you did not do it yet, you have to modify your path variables:

```bash
export PATH=$PATH:/home/sahbi/Prgrams/caffe-segnet/build/tools
export PYTHONPATH=/home/sahbi/Programs/caffe-segnet/python:$PYTHONPATH
```

- For the training, the labeled images must have pixels 0 and 1, not 0 and 255 (Error *Check failed: status == CUBLAS_STATUS_SUCCESS (11 vs. 0) CUBLAS_STATUS_INTERNAL_ERROR*), used **imagemagick** for that. However, you must do that **before** resizing the images (else you have other pixel values that appear)

```bash
cd trainannot
mogrify -fill "rgb(1,1,1)" -opaque "rgb(255,255,255)" *png
mogrify -resize 25% *jpg
```