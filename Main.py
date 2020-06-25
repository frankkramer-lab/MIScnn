import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import nibabel as nib
import matplotlib.pyplot as plt

# Library import
from miscnn.data_loading.interfaces.dicom_io import DICOM_interface
from miscnn.data_loading.interfaces.nifti_io import NIFTI_interface
from miscnn.data_loading.data_io import Data_IO

structure_dict = {"Lung_L": 1,
                  "Lung_R": 2}

# Initialize the NIfTI I/O interface and configure the images as one channel (grayscale) and three segmentation classes (background, kidney, tumor)
interface = DICOM_interface(structure_dict = structure_dict, classes=3)
#interface = NIFTI_interface(pattern="case_00[0-9]*", 
                            #channels=1, classes=3)
# Specify the kits19 data directory ('kits19/data' was renamed to 'kits19/data.original')
data_path = "/home/mluser1/Desktop/LCTSC/"
#data_path = '/home/mluser1/Desktop/kits19/data/'
# Create the Data I/O object 
data_io = Data_IO(interface, data_path)

sample_list = data_io.get_indiceslist()
sample_list.sort()
print("All samples: " + str(sample_list))

# Library import
from miscnn.processing.data_augmentation import Data_Augmentation

# Create and configure the Data Augmentation class
data_aug = Data_Augmentation(cycles=2, scaling=True, rotations=True, elastic_deform=True, mirror=True,
                             brightness=True, contrast=True, gamma=True, gaussian_noise=True)


from miscnn.processing.subfunctions.normalization import Normalization
from miscnn.processing.subfunctions.clipping import Clipping
from miscnn.processing.subfunctions.resampling import Resampling

# Create a pixel value normalization Subfunction through Z-Score 
sf_normalize = Normalization()

sf_resample = Resampling((3.22, 1.62, 1.62))
# Create a clipping Subfunction between -79 and 304
sf_clipping = Clipping(min=-79, max=304)
# Create a resampling Subfunction to voxel spacing 3.22 x 1.62 x 1.62
# Assemble Subfunction classes into a list
# Be aware that the Subfunctions will be exectued according to the list order!
subfunctions = [sf_resample, sf_clipping, sf_normalize]


# Library import
from miscnn.processing.preprocessor import Preprocessor

# Create and configure the Preprocessor class
pp = Preprocessor(data_io, data_aug=data_aug, batch_size=2, subfunctions=subfunctions, prepare_subfunctions=False, 
                  prepare_batches=False, analysis="patchwise-crop", patch_shape=(80, 160, 160))

# Adjust the patch overlap for predictions
#pp.patchwise_overlap = (40, 80, 80)

# Library import
from miscnn.neural_network.model import Neural_Network
from miscnn.neural_network.metrics import dice_soft, dice_crossentropy, tversky_loss

# Create the Neural Network model
model = Neural_Network(preprocessor=pp, loss=tversky_loss, metrics=[dice_soft, dice_crossentropy],
                       batch_queue_size=5, workers=8, learninig_rate=0.0001, gpu_number=1)
from miscnn.utils.tensor_board_logger import TrainValTensorBoard

from keras.callbacks import ModelCheckpoint
cb_model = ModelCheckpoint(os.path.join('/mnt/md1/Micha/Projects/MIScnn/training/', 'DICOM_interface_test' + ".hdf5"),
                                   monitor="val_loss", verbose=1,
                                   save_best_only=True, mode="min")


callbacks=[TrainValTensorBoard('{}_training_2D'.format('seg_' + 'DICOM_interface_test'), '{}_validation_2D'.format('seg_' + 'DICOM_interface_test')), cb_model]


from miscnn.evaluation import split_validation

split_validation(sample_list, model,epochs = 200, callbacks = callbacks)

