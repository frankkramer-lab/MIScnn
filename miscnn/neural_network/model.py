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
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
import numpy as np
# Internal libraries/scripts
from miscnn.neural_network.metrics import dice_classwise, tversky_loss
from miscnn.neural_network.architecture.unet.standard import Architecture
from miscnn.neural_network.data_generator import DataGenerator
from miscnn.utils.patch_operations import concat_3Dmatrices

#-----------------------------------------------------#
#            Neural Network (model) class             #
#-----------------------------------------------------#
# Class which represents the Neural Network and which run the whole pipeline
class Neural_Network:
    """ Initialization function for creating a Neural Network (model) object.
    This class provides functionality for handling all model methods.
    This class runs the whole pipeline and uses a Preprocessor instance to obtain batches.

    With an initialized Neural Network model instance, it is possible to run training, prediction
    and evaluations.

    Args:
        preprocessor (Preprocessor):            Preprocessor class instance which provides the Neural Network with batches.
        architecture (Architecture):            Instance of a neural network model Architecture class instance.
                                                By default, a standard U-Net is used as Architecture.
        loss (Metric Function):                 The metric function which is used as loss for training.
                                                Any Metric Function defined in Keras, in miscnn.neural_network.metrics or any custom
                                                metric function, which follows the Keras metric guidelines, can be used.
        metrics (List of Metric Functions):     List of one or multiple Metric Functions, which will be shown during training.
                                                Any Metric Function defined in Keras, in miscnn.neural_network.metrics or any custom
                                                metric function, which follows the Keras metric guidelines, can be used.
        epochs (integer):                       Number of epochs. A single epoch is defined as one iteration through the complete data set.
        learning_rate (float):                  Learning rate in which weights of the neural network will be updated.
        batch_queue_size (integer):             The batch queue size is the number of previously prepared batches in the cache during runtime.
        gpu_number (integer):                   Number of GPUs, which will be used for training.
    """
    def __init__(self, preprocessor, architecture=Architecture(),
                 loss=tversky_loss, metrics=[dice_classwise],
                 epochs=20, learninig_rate=0.0001,
                 batch_queue_size=2, gpu_number=1):
        # Identify data parameters
        three_dim = preprocessor.data_io.interface.three_dim
        channels = preprocessor.data_io.interface.channels
        classes = preprocessor.data_io.interface.classes
        # Assemble the input shape
        input_shape = (None,)
        # Initialize model for 3D data
        if three_dim:
            input_shape = (None, None, None, channels)
            self.model = architecture.create_model_3D(input_shape=input_shape,
                                                      n_labels=classes)
         # Initialize model for 2D data
        else:
             input_shape = (None, None, channels)
             self.model = architecture.create_model_2D(input_shape=input_shape,
                                                       n_labels=classes)
        # Transform to Keras multi GPU model
        if gpu_number > 1:
            self.model = multi_gpu_model(self.model, gpu_number)
        # Compile model
        self.model.compile(optimizer=Adam(lr=learninig_rate),
                           loss=loss, metrics=metrics)
        # Cache parameter
        self.preprocessor = preprocessor
        self.epochs = epochs
        self.batch_queue_size = batch_queue_size

    #---------------------------------------------#
    #               Class variables               #
    #---------------------------------------------#
    shuffle_batches = True                  # Option whether batch order should be shuffled or not

    #---------------------------------------------#
    #                  Training                   #
    #---------------------------------------------#
    # Train the Neural Network model using the MIScnn pipeline
    def train(self, sample_list=None):
        # Without sample list, use all samples from the Preprocessor
        if not isinstance(sample_list, list):
            sample_list = self.preprocessor.data_io.get_indiceslist()
        # Initialize Keras Data Generator for generating batches
        dataGen = DataGenerator(sample_list, self.preprocessor, training=True,
                                validation=False, shuffle=self.shuffle_batches)
        # Run training process with Keras fit_generator
        self.model.fit_generator(generator=dataGen,
                                 epochs=self.epochs,
                                 max_queue_size=self.batch_queue_size)
        # Clean up temporary npz files if necessary
        if self.preprocessor.prepare_batches:
            self.preprocessor.data_io.batch_npz_cleanup()

    #---------------------------------------------#
    #                 Prediction                  #
    #---------------------------------------------#
    # Predict with the fitted Neural Network model
    def predict(self, sample_list=None, direct_output=False):
        # Without sample list, use all samples from the Preprocessor
        if not isinstance(sample_list, list):
            sample_list = self.preprocessor.data_io.get_indiceslist()
        # Initialize result array for direct output
        if direct_output : results = []
        # Iterate over each sample
        for sample in sample_list:
            # Initialize Keras Data Generator for generating batches
            dataGen = DataGenerator([sample], self.preprocessor,
                                    training=False, validation=False,
                                    shuffle=False)
            # Run prediction process with Keras predict_generator
            pred_seg = self.model.predict_generator(
                                     generator=dataGen,
                                     max_queue_size=self.batch_queue_size)

            # Reassemble patches into original shape for patchwise analysis
            if self.preprocessor.analysis == "patchwise-crop" or \
                self.preprocessor.analysis == "patchwise-grid":
                # Load cached shape
                seg_shape = self.preprocessor.shape_cache.pop(sample)
                # Concatenate patches into original shape
                pred_seg = concat_3Dmatrices(
                               patches=pred_seg,
                               image_size=seg_shape,
                               window=self.preprocessor.patch_shape,
                               overlap=self.preprocessor.patchwise_grid_overlap)
            # Transform probabilities to classes
            pred_seg = np.argmax(pred_seg, axis=-1)
            # Run Subfunction postprocessing on the prediction
            for sf in self.preprocessor.subfunctions:
                sf.postprocessing(pred_seg)
            # Backup predicted segmentation
            if direct_output : results.append(pred_seg)
            else : self.preprocessor.data_io.save_prediction(pred_seg, sample)
            # Clean up temporary npz files if necessary
            if self.preprocessor.prepare_batches:
                self.preprocessor.data_io.batch_npz_cleanup()
        # Output predictions results if direct output modus is active
        if direct_output : return results

    #---------------------------------------------#
    #                 Evaluation                  #
    #---------------------------------------------#
    #
    def evaluate(self, sample_list=None):
        # Without sample list, use all samples from the Preprocessor
        if not isinstance(sample_list, list):
            sample_list = self.preprocessor.data_io.get_indiceslist()
