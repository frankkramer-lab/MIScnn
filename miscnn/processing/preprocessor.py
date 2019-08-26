#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2019 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import numpy as np
from keras.utils import to_categorical
# Internal libraries/scripts
from miscnn.processing.data_augmentation import Data_Augmentation
from miscnn.processing.batch_creation import create_batches
from miscnn.utils.patch_operations import slice_3Dmatrix

#-----------------------------------------------------#
#                 Preprocessor class                  #
#-----------------------------------------------------#
# Class to handle all preprocessing functionalities
class Preprocessor:
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    """ Initialization function for creating a Preprocessor object.
    This class provides functionality for handling all preprocessing methods. This includes diverse
    optional processing subfunctions like resampling, clipping, normalization or custom subfcuntions.
    This class processes the data into batches which are ready to be used for training, prediction and validation.

    The user is only required to create an instance of the Preprocessor class with the desired specifications
    and Data IO instance (optional also Data Augmentation instance).

    Args:
        data_io (Data_IO):                      Data IO class instance which handles all I/O operations according to the user
                                                defined interface.
        batch_size (integer):                   Number of samples inside a single batch.
        subfunctions (list of Subfunctions):    List of Subfunctions class instances which will be SEQUENTIALLY executed on the data set.
                                                (clipping, normalization, resampling, ...)
        data_aug (Data_Augmentation):           Data Augmentation class instance which performs diverse data augmentation techniques.
                                                If no Data Augmentation is provided, an instance with default settings will be created.
                                                Use data_aug=None, if you want no data augmentation at all.
        prepare_subfunctions (boolean):         Should all subfunctions be prepared and backup to disk before starting the batch generation
                                                (True), or should the subfunctions preprocessing be performed during runtime? (False).
        prepare_batches (boolean):              Should all batches be prepared and backup to disk before starting the training (True),
                                                or should the batches be created during runtime? (False).
        analysis (string):                      Modus selection of analysis type. Options:
                                                - "fullimage":      Analysis of complete image data
                                                - "patchwise-crop": Analysis of random cropped patches from the image
                                                - "patchwise-grid": Analysis of patches by splitting the image into a grid
        patch_shape (integer tuple):            Size and shape of a patch. The variable has to be defined as a tuple.
                                                For Example: (64,128,128) for 64x128x128 patch cubes.
                                                Be aware that the x-axis represents the number of slices in 3D volumes.
                                                This parameter will be redundant if fullimage or patchwise-crop analysis is selected!!
    """
    def __init__(self, data_io, batch_size, subfunctions=[],
                 data_aug=Data_Augmentation(), prepare_subfunctions=False,
                 prepare_batches=False, analysis="patchwise-crop",
                 patch_shape=None):
        # Parse Data Augmentation
        if isinstance(data_aug, Data_Augmentation):
            self.data_augmentation = data_aug
        else:
            self.data_augmentation = None
        # Exception: Analysis parameter check
        analysis_types = ["patchwise-crop", "patchwise-grid", "fullimage"]
        if not isinstance(analysis, str) or analysis not in analysis_types:
            raise ValueError('Non existent analysis type in preprocessing.')
        # Exception: Patch-shape parameter check
        if (analysis == "patchwise-crop" or analysis == "patchwise-grid") and \
            not isinstance(patch_shape, tuple):
            raise ValueError("Missing or wrong patch shape parameter for " + \
                             "patchwise analysis.")
        # Parse parameter
        self.data_io = data_io
        self.batch_size = batch_size
        self.subfunctions = subfunctions
        self.prepare_subfunctions = prepare_subfunctions
        self.prepare_batches = prepare_batches
        self.analysis = analysis
        self.patch_shape = patch_shape

    #---------------------------------------------#
    #               Class variables               #
    #---------------------------------------------#
    patchwise_grid_overlap = (0,0,0)        # In patchwise_analysis and without random cropping, an overlap can be defined
                                            # between adjuncted patches.
    patchwise_grid_skip_blanks = True       # In patchwise_analysis and without random cropping, patches, containing only the
                                            # background annotation, can be skipped with this option. This result into only
                                            # training on relevant patches and ignore patches without any information.
    patchwise_grid_skip_class = 0           # Class, which will be skipped if patchwise_grid_skip_blanks is True
    img_queue = []                          # Intern queue of already processed and data augmentated images or segmentations.
                                            # The function create_batches will use this queue to create batches
    shape_cache = dict()                    # Cache the shape of an image for patch assembling after patchwise prediction

    #---------------------------------------------#
    #               Prepare Batches               #
    #---------------------------------------------#
    # Preprocess data and prepare the batches for a given list of indices
    def run(self, indices_list, training=True, validation=False):
        # Initialize storage type
        if self.prepare_batches : batchpointer = -1     # Batch pointer for later batchpointer list (references to batch disk backup)
        else : all_batches = []                         # List of batches from all samples (can sum up large amount of memory with wrong usage)
        # Iterate over all samples
        for index in indices_list:
            # Load sample and process provided subfunctions on image data
            if not self.prepare_subfunctions:
                sample = self.data_io.sample_loader(index, load_seg=training)
                for sf in self.subfunctions:
                    sf.preprocessing(sample, training=training)
            # Load sample from file with already processed subfunctions
            else : sample = self.data_io.sample_loader(index, backup=True)
            # Transform digit segmentation classes into categorical
            if training:
                sample.seg_data = to_categorical(sample.seg_data,
                                                 num_classes=sample.classes)
            # Decide if data augmentation should be performed
            if training and not validation and self.data_augmentation is not None:
                data_aug = True
            else:
                data_aug = False
            # Run Fullimage analysis
            if self.analysis == "fullimage":
                ready_data = self.analysis_fullimage(sample, training,
                                                     data_aug)
            # Run patchwise cropping analysis
            elif self.analysis == "patchwise-crop" and training:
                ready_data = self.analysis_patchwise_crop(sample, data_aug)
            # Run patchwise grid analysis
            else:
                if not training: self.shape_cache[index] = sample.img_data.shape
                ready_data = self.analysis_patchwise_grid(sample, training,
                                                          data_aug)
            # Put the preprocessed data at the image queue end
            self.img_queue.extend(ready_data)
            # Identify if current index is the last one
            if index == indices_list[-1]: last_index = True
            else : last_index = False
            # Identify if incomplete_batches are allowed for batch creation
            if training : incomplete_batches = False
            else : incomplete_batches = True
            # Create batches by gathering images from the img_queue
            batches = create_batches(self.img_queue, self.batch_size,
                                     incomplete_batches, last_index)
            # Backup batches to disk
            if self.prepare_batches:
                for batch in batches:
                    batchpointer += 1
                    if not training:
                        batch_address = "prediction" + "." + str(batchpointer)
                    elif validation:
                        batch_address = "validation" + "." + str(batchpointer)
                    else:
                        batch_address = "training" + "." + str(batchpointer)
                    self.data_io.backup_batches(batch[0], batch[1],
                                                batch_address)
            # Backup batches to memory
            else : all_batches.extend(batches)
        # Return prepared batches
        if self.prepare_batches : return batchpointer
        else : return all_batches

    #---------------------------------------------#
    #               Run Subfunctions              #
    #---------------------------------------------#
    # Preprocess data through subfunctions and backup samples
    def run_subfunctions(self, indices_list, training=True):
        # Iterate over all samples
        for index in indices_list:
            # Load sample
            sample = self.data_io.sample_loader(index, load_seg=training)
            # Run provided subfunctions on imaging data
            for sf in self.subfunctions:
                sf.preprocessing(sample, training=training)
            # Backup sample as pickle to disk
            self.data_io.backup_sample(sample)

    #---------------------------------------------#
    #           Patch-wise grid Analysis          #
    #---------------------------------------------#
    def analysis_patchwise_grid(self, sample, training, data_aug):
        # Slice image into patches
        patches_img = slice_3Dmatrix(sample.img_data,
                                     self.patch_shape,
                                     self.patchwise_grid_overlap)
        if training:
            # Slice segmentation into patches
            patches_seg = slice_3Dmatrix(sample.seg_data,
                                         self.patch_shape,
                                         self.patchwise_grid_overlap)
        else : patches_seg = None
        # Skip blank patches (only background)
        if training and self.patchwise_grid_skip_blanks:
            # Iterate over each patch
            for i in reversed(range(0, len(patches_seg))):
                # IF patch DON'T contain any non background class -> remove it
                if not np.any(patches_seg[i][...,self.patchwise_grid_skip_class] != 1):
                    del patches_img[i]
                    del patches_seg[i]
        # Concatenate a list of patches into a single numpy array
        img_data = np.stack(patches_img, axis=0)
        if training : seg_data = np.stack(patches_seg, axis=0)
        # Run data augmentation
        if data_aug:
            img_data, seg_data = self.data_augmentation.run(img_data, seg_data)
        # Create tuple of preprocessed data
        if training:
            ready_data = list(zip(img_data, seg_data))
        else:
            ready_data = list(zip(img_data))
        # Return preprocessed data tuple
        return ready_data

    #---------------------------------------------#
    #           Patch-wise crop Analysis          #
    #---------------------------------------------#
    def analysis_patchwise_crop(self, sample, data_aug):
        # Access image and segmentation data
        img = sample.img_data
        seg = sample.seg_data
        # If no data augmentation should be performed
        # -> create Data Augmentation instance without augmentation methods
        if not data_aug or self.data_augmentation is None:
            cropping_data_aug = Data_Augmentation(cycles=1,
                                        scaling=False, rotations=False,
                                        elastic_deform=False, mirror=False,
                                        brightness=False, contrast=False,
                                        gamma=False, gaussian_noise=False)
        else : cropping_data_aug = self.data_augmentation
        # Configure the Data Augmentation instance to cropping
        cropping_data_aug.cropping = True
        cropping_data_aug.cropping_patch_shape = self.patch_shape
        # Expand image dimension to simulate a batch with one image
        img_data = np.expand_dims(img, axis=0)
        seg_data = np.expand_dims(seg, axis=0)
        # Run data augmentation and cropping
        img_data, seg_data = cropping_data_aug.run(img_data, seg_data)
        # Create tuple of preprocessed data
        ready_data = list(zip(img_data, seg_data))
        # Return preprocessed data tuple
        return ready_data

    #---------------------------------------------#
    #             Full-Image Analysis             #
    #---------------------------------------------#
    def analysis_fullimage(self, sample, training, data_aug):
        #-> optional data_aug

        # # Expand image dimension to simulate a batch with one image
        # self.image = np.expand_dims(self.image, axis=0)
        # self.segmentation = np.expand_dims(self.segmentation, axis=0)

        return None
