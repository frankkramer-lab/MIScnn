#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2020 IT-Infrastructure for Translational Medical Research,    #
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
#External libraries
from tensorflow.keras.utils import Sequence as Keras_Sequence
import math
import numpy as np
from miscnn.utils.visualizer import visualize_sample

#-----------------------------------------------------#
#                 Keras Data Generator                #
#-----------------------------------------------------#
# Data Generator for generating batches (WITH-/OUT segmentation)
## Returns a batch containing one or multiple images for training/prediction
class DataGenerator(Keras_Sequence):
    # Class Initialization
    def __init__(self, sample_list, preprocessor, training=False,
                 validation=False, shuffle=False, iterations=None):
        # Parse sample list
        if isinstance(sample_list, list) : self.sample_list = sample_list.copy()
        elif type(sample_list).__module__ == np.__name__ :
            self.sample_list = sample_list.tolist()
        else : raise ValueError("Sample list have to be a list or numpy array!")
        # Create a working environment from the handed over variables
        self.preprocessor = preprocessor
        self.training = training
        self.validation = validation
        self.shuffle = shuffle
        self.iterations = iterations
        self.batch_queue = []
        # If samples with subroutines should be preprocessed -> do it now
        if preprocessor.prepare_subfunctions:
            preprocessor.run_subfunctions(sample_list, training)
        # If batches should be prepared before runtime -> do it now
        if preprocessor.prepare_batches:
            batches_count = preprocessor.run(sample_list, training, validation)
            self.batchpointers = list(range(0, batches_count+1))
        elif not training:
            self.batch_queue = preprocessor.run(sample_list, False, False)

    # Return the next batch for associated index
    def __getitem__(self, idx):
        # Load a batch by generating it or by loading an already prepared
        if self.preprocessor.prepare_batches : batch = self.load_batch(idx)
        else : batch = self.generate_batch(idx)
        # Return the batch containing only an image or an image and segmentation
        if self.training:
            return batch[0], batch[1]
        else:
            return batch[0]

    # Return the number of batches for one epoch
    def __len__(self):
        # Number of batches is preprocessed for the single sample to predict
        if not self.training:
            return len(self.batch_queue)
        # IF number of samples is specified in the parameters take it
        elif self.iterations is not None : return self.iterations
        # Number of batches is the number of preprocessed batch files
        elif self.preprocessor.prepare_batches:
            return len(self.batchpointers)
        # Else number of samples is dynamic -> calculate it
        else:
            if self.preprocessor.data_augmentation is not None and not \
                self.validation:
                cycles = self.preprocessor.data_augmentation.cycles
            else:
                cycles = 1
            return math.ceil((len(self.sample_list) * cycles) / \
                             self.preprocessor.batch_size)

    # At every epoch end: Shuffle batchPointer list and reset sample_list
    def on_epoch_end(self):
        if self.shuffle and self.training:
            if self.preprocessor.prepare_batches:
                np.random.shuffle(self.batchpointers)
            else:
                np.random.shuffle(self.sample_list)

    #-----------------------------------------------------#
    #                     Subroutines                     #
    #-----------------------------------------------------#
    # Load an already prepared batch from disk
    def load_batch(self, idx):
        # Handle index adjustment for iterations > number of batches
        if idx >= len(self.batchpointers) : idx = idx % len(self.batchpointers)
        # Identify batch address
        if not self.training:
            batch_address = "prediction" + "." + str(self.batchpointers[idx])
        elif self.validation:
            batch_address = "validation" + "." + str(self.batchpointers[idx])
        else:
            batch_address = "training" + "." + str(self.batchpointers[idx])
        # Load next batch containing images
        batch_img = self.preprocessor.data_io.batch_load(batch_address,
                                                         img=True)
        # IF batch is for training -> return next img & seg batch
        if self.training:
            # Load next batch containing segmentations
            batch_seg = self.preprocessor.data_io.batch_load(batch_address,
                                                             img=False)
            # Return image and segmentation batch
            return (batch_img, batch_seg)
        # IF batch is for predicting -> return only next image batch
        else : return (batch_img, None)

    # Generate a batch during runtime
    def generate_batch(self, idx):
        # output an already generated batch if there are still batches in the queue
        if self.batch_queue : return self.batch_queue.pop(0)
        # otherwise generate a new batch
        else:
            # identify number of images required for a single batch
            if self.preprocessor.data_augmentation is not None and not \
                self.validation:
                cycles = self.preprocessor.data_augmentation.cycles
            else:
                cycles = 1
            # Create threading lock to avoid parallel access
            with self.preprocessor.thread_lock:
                sample_size = math.ceil(self.preprocessor.batch_size / cycles)
                # access samples
                samples = self.sample_list[:sample_size]
                # move samples from top to bottom in the sample queue
                del self.sample_list[:sample_size]
                self.sample_list.extend(samples)
            # create a new batch
            batches = self.preprocessor.run(samples, self.training,
                                            self.validation)
            # Create threading lock to avoid parallel access
            with self.preprocessor.thread_lock:
                # Access a newly created batch
                next_batch = batches.pop(0)
                # Add remaining batches to batch queue
                if len(batches) > 0 : self.batch_queue.extend(batches)
            # output a created batch
            return next_batch
