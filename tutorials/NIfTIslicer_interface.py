## The aim of this tutorial script is to show, how you can use the NIFTI slicer
## IO interface.
##
## This interface automatically slices a data set of NIfTI 3D volumes into 2D
## images for using specific 2D models
##
## Based on the KITS 19 data set (Kidney Tumor Segmentation Challenge 2019)

# Import all libraries we need
from miscnn import Data_IO, Preprocessor, Neural_Network
from miscnn.data_loading.interfaces import NIFTIslicer_interface
from miscnn.processing.subfunctions import Resize

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
print(sample.img_data.shape)
print(sample.seg_data.shape)

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
pred = model.predict(["case_00002:#:42"], direct_output=True)
print(pred.shape)

# Predict a generic slice and save it as grayscale JPEG on disk
model.predict(["case_00002:#:42"], direct_output=False)
