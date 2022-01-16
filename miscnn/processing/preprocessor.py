#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2020 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  Contributions: Michael Lempart, 2020 Department of Radiation Physics,       #
#                                  Lund University                             #
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
from tensorflow.keras.utils import to_categorical
import threading
import multiprocessing as mp
from functools import partial
# Internal libraries/scripts
from .data_augmentation import Data_Augmentation
from .batch_creation import create_batches

from .patching.patch_handler import Patch_Handler

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
        use_multiprocessing (boolean):          Uses multi-threading to prepare subfunctions if True (parallelized).
    """
    def __init__(self, data_io, batch_size, subfunctions=[],
                 data_aug=Data_Augmentation(), prepare_subfunctions=False,
                 prepare_batches=False, use_multiprocessing=False, *argv, **kwargs):
        # Parse Data Augmentation
        if isinstance(data_aug, Data_Augmentation):
            self.data_augmentation = data_aug
        else:
            self.data_augmentation = None
        # Parse parameter
        self.data_io = data_io
        self.batch_size = batch_size
        self.subfunctions = subfunctions
        self.prepare_subfunctions = prepare_subfunctions
        self.prepare_batches = prepare_batches
        self.use_multiprocessing = use_multiprocessing
        
        self.patch_handler = Patch_Handler(*argv, **kwargs)

    #---------------------------------------------#
    #               Class variables               #
    #---------------------------------------------#
    img_queue = []                          # Intern queue of already processed and data augmentated images or segmentations.
                                            # The function create_batches will use this queue to create batches
    thread_lock = threading.Lock()          # Create a threading lock for multiprocessing
    mp_threads = 5                          # Number of threads used to prepare subfunctions if use_multiprocessing is set to True

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
            # Cache sample object for prediction
            # Transform digit segmentation classes into categorical
            if training:
                sample.seg_data = to_categorical(sample.seg_data,
                                                 num_classes=sample.classes)
            # Decide if data augmentation should be performed on data
            if training and not validation and self.data_augmentation is not None:
                data_aug = True
            elif not training and self.data_augmentation is not None and \
                self.data_augmentation.infaug:
                data_aug = True
            else:
                data_aug = False
            
            ready_data = self.patch_handler.patch(sample, training, data_aug)
            
            # Identify if current index is the last one
            if index == indices_list[-1]: last_index = True
            else : last_index = False
            # Identify if incomplete_batches are allowed for batch creation
            if training : incomplete_batches = False
            else : incomplete_batches = True
            # Create threading lock to avoid parallel access
            with self.thread_lock:
                # Put the preprocessed data at the image queue end
                self.img_queue.extend(ready_data)
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
    #          Prediction Postprocessing          #
    #---------------------------------------------#
    # Postprocess prediction data
    def postprocessing(self, sample, prediction, activation_output=False):
        # Apply back-flipping if inference augmentation is active
        if self.data_augmentation is not None and self.data_augmentation.infaug:
            prediction = self.data_augmentation.run_infaug(prediction)
        # Reassemble patches into original shape for patchwise analysis
        
        prediction = self.patch_handler.unpatch(sample, prediction)
        
        # Transform probabilities to classes
        if not activation_output : prediction = np.argmax(prediction, axis=-1)
        # Run Subfunction postprocessing on the prediction
        for sf in reversed(self.subfunctions):
            prediction = sf.postprocessing(sample, prediction, activation_output)
        # Return postprocessed prediction
        return prediction

    #---------------------------------------------#
    #               Run Subfunctions              #
    #---------------------------------------------#
    # Iterate over all samples, process subfunctions on each and backup
    # preprocessed samples to disk
    def run_subfunctions(self, indices_list, training=True):
        # Prepare subfunctions using single threading
        if not self.use_multiprocessing or not training:
            for index in indices_list:
                self.prepare_sample_subfunctions(index, training)
        # Prepare subfunctions using multiprocessing
        else:
            pool = mp.Pool(int(self.mp_threads))
            pool.map(partial(self.prepare_sample_subfunctions,
                             training=training),
                     indices_list)
            pool.close()
            pool.join()

    # Wrapper function to process subfunctions for a single sample
    def prepare_sample_subfunctions(self, index, training):
        # Load sample
        sample = self.data_io.sample_loader(index, load_seg=training)
        # Run provided subfunctions on imaging data
        for sf in self.subfunctions:
            sf.preprocessing(sample, training=training)
        # Transform array data types in order to save disk space
        sample.img_data = np.array(sample.img_data, dtype=np.float32)
        if training:
            sample.seg_data = np.array(sample.seg_data, dtype=np.uint8)
        # Backup sample as pickle to disk
        self.data_io.backup_sample(sample)
    
    def batch_cleanup(self):
        if self.prepare_batches or self.prepare_subfunctions:
            self.data_io.batch_cleanup()
    