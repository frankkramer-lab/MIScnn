## The aim of this tutorial script is to show, how you can use the NIFTI slicer
## IO interface.
##
## This interface automatically slices a data set of NIfTI 3D volumes into 2D
## images for using specific 2D models
##
## Based on the KITS 19 data set (Kidney Tumor Segmentation Challenge 2019)
## Data Set: https://github.com/neheller/kits19

# Import all libraries we need
from miscnn import Data_IO, Preprocessor, Neural_Network
from miscnn.data_loading.interfaces import NIFTIslicer_interface
from miscnn.processing.subfunctions import Resize
import numpy as np

# Initialize the NIfTI interface IO slicer variant
interface = NIFTIslicer_interface(pattern="case_0000[0-3]", channels=1, classes=3)

# Initialize the Data IO class
data_path = "/home/mudomini/projects/KITS_challenge2019/kits19/data.interpolated/"
data_io = Data_IO(interface, data_path, delete_batchDir=False)

# Obtain the list of samples from our data set
## A sample is defined as a single slice (2D image)
samples_list = data_io.get_indiceslist()
samples_list.sort()

# Let's test out, if the the NIfTI slicer interface works like we want
# and output the image and segmentation shape of a random slice
sample = data_io.sample_loader("case_00002:#:42", load_seg=True)
print(sample.img_data.shape, sample.seg_data.shape)

## As you hopefully noted, the index of a slice is defined
## as the volume file name and the slice number separated with a ":#:"

# Specify subfunctions for preprocessing
## Here we are using the Resize subfunctions due to many 2D models
## want a specific shape (e.g. DenseNet for classification)
sf = [Resize(new_shape=(224, 224))]

# Initialize the Preprocessor class
pp = Preprocessor(data_io, data_aug=None, batch_size=1, subfunctions=sf,
                  prepare_subfunctions=True, prepare_batches=False,
                  analysis="fullimage")
## We are using fullimage analysis due to a 2D image can easily fit completely
## in our GPU

# Initialize the neural network model
model = Neural_Network(preprocessor=pp)

# Start the fitting on some slices
model.train(samples_list[30:50], epochs=3, iterations=10, callbacks=[])

# Predict a generic slice with direct output
pred = model.predict(["case_00002:#:42"], return_output=True)
print(np.asarray(pred).shape)
## Be aware that the direct prediction output, has a additional batch axis

# Predict a generic slice and save it as a NumPy pickle on disk
model.predict(["case_00002:#:42"], return_output=False)

# Load the slice via sample-loader and output also the new prediction, now
sample = data_io.sample_loader("case_00000:#:89", load_seg=True, load_pred=True)
print(sample.img_data.shape, sample.seg_data.shape, sample.pred_data.shape)


## Final words
# I hope that this usage example / tutorial on the new NIfTI slicer IO
# interface, helps on understanding how it works and how you can use it
#
# If you have any questions or feedback, please do not hesitate to write me an
# email or create an issue on GitHub!
