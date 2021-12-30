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
# External libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import copy
# Internal libraries/scripts
from miscnn.multi_model.model import Model as BaseModel
from miscnn.neural_network.metrics import dice_soft, tversky_loss
from miscnn.neural_network.architecture.unet.standard import Architecture
from miscnn.neural_network.data_generator import DataGenerator

#-----------------------------------------------------#
#            Neural Network (model) class             #
#-----------------------------------------------------#
# Class which represents the Neural Network and which run the whole pipeline
class Neural_Network(BaseModel):
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
        learning_rate (float):                  Learning rate in which weights of the neural network will be updated.
        batch_queue_size (integer):             The batch queue size is the number of previously prepared batches in the cache during runtime.
        Number of workers (integer):            Number of workers/threads which preprocess batches during runtime.
        multi_gpu (boolean):                    Parameter which decides, if multiple gpus will be used for training (Distributed training).
                                                By default, false.
    """
    def __init__(self, preprocessor, architecture=Architecture(),
                 loss=tversky_loss, metrics=[dice_soft],
                 learning_rate=0.0001, batch_queue_size=2,
                 workers=1, multi_gpu=False):
        BaseModel.__init__(self, preprocessor)
        # Cache parameter
        self.loss = loss
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.batch_queue_size = batch_queue_size
        self.workers = workers
        self.architecture = architecture
        # Build model with multiple GPUs (MirroredStrategy)
        if multi_gpu:
            strategy = tf.distribute.MirroredStrategy(
                    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
            with strategy.scope() : self.build_model(architecture)
        # Build model with single GPU
        else : self.build_model(architecture)
        # Cache starting weights
        self.initialization_weights = self.model.get_weights()


    #---------------------------------------------#
    #               Class variables               #
    #---------------------------------------------#
    shuffle_batches = True                  # Option whether batch order should be shuffled or not
    initialization_weights = None           # Neural Network model weights for weight reinitialization

    #---------------------------------------------#
    #                Model Creation               #
    #---------------------------------------------#
    def build_model(self, architecture):
        # Assemble the input shape
        input_shape = (None,)
        # Initialize model for 3D data
        if self.three_dim:
            input_shape = (None, None, None, self.channels)
            if not self.preprocessor.analysis == "fullimage":
                input_shape = self.preprocessor.patch_shape + (self.channels,)
            self.model = architecture.create_model_3D(input_shape=input_shape,
                                                      n_labels=self.classes)
         # Initialize model for 2D data
        else:
            input_shape = (None, None, self.channels)
            if not self.preprocessor.analysis == "fullimage":
                input_shape = self.preprocessor.patch_shape + (self.channels,)
            self.model = architecture.create_model_2D(input_shape=input_shape,
                                                       n_labels=self.classes)
        # Compile model
        self.model.compile(optimizer=Adam(lr=self.learning_rate),
                           loss=self.loss, metrics=self.metrics)

    #---------------------------------------------#
    #                  Training                   #
    #---------------------------------------------#
    """ Fitting function for the Neural Network model using the provided list of sample indices.

    Args:
        sample_list (list of indices):          A list of sample indicies which will be used for training
        epochs (integer):                       Number of epochs. A single epoch is defined as one iteration through
                                                the complete data set.
        iterations (integer):                   Number of iterations (batches) in a single epoch.
        callbacks (list of Callback classes):   A list of Callback classes for custom evaluation.
        class_weight (dictionary or list):      A list or dictionary of float values to handle class unbalance.
    """
    def train(self, sample_list, epochs=20, iterations=None, callbacks=[],
              class_weight=None):
        # Initialize Keras Data Generator for generating batches
        dataGen = DataGenerator(sample_list, self.preprocessor, training=True,
                                validation=False, shuffle=self.shuffle_batches,
                                iterations=iterations)
        # Run training process with Keras fit
        self.model.fit(dataGen,
                       epochs=epochs,
                       callbacks=callbacks,
                       class_weight=class_weight,
                       workers=self.workers,
                       max_queue_size=self.batch_queue_size)
        # Clean up temporary files if necessary
        if self.preprocessor.prepare_batches or self.preprocessor.prepare_subfunctions:
            self.preprocessor.data_io.batch_cleanup()

    #---------------------------------------------#
    #                 Prediction                  #
    #---------------------------------------------#
    """ Prediction function for the Neural Network model. The fitted model will predict a segmentation
        for the provided list of sample indices.

    Args:
        sample_list (list of indices):  A list of sample indicies for which a segmentation prediction will be computed.
        return_output (boolean):        Parameter which decides, if computed predictions will be output as the return of this
                                        function or if the predictions will be saved with the save_prediction method defined
                                        in the provided Data I/O interface.
        activation_output (boolean):    Parameter which decides, if model output (activation function, normally softmax) will
                                        be saved/outputed (if FALSE) or if the resulting class label (argmax) should be outputed.
    """
    def predict(self, sample_list, return_output=False,
                activation_output=False):
        # Initialize result array for direct output
        if return_output : results = []
        # Iterate over each sample
        for sample in sample_list:
            # Initialize Keras Data Generator for generating batches
            dataGen = DataGenerator([sample], self.preprocessor,
                                    training=False, validation=False,
                                    shuffle=False, iterations=None)
            # Run prediction process with Keras predict
            pred_list = []
            for batch in dataGen:
                pred_batch = self.model.predict_on_batch(batch)
                pred_list.append(pred_batch)
            pred_seg = np.concatenate(pred_list, axis=0)
            # Postprocess prediction
            sampleObj = self.preprocessor.cache.pop(sample)
            pred_seg = self.preprocessor.postprocessing(sampleObj, pred_seg,
                                                        activation_output)
            # Backup predicted segmentation
            if return_output : results.append(pred_seg)
            else :
              sampleObj.add_prediction(pred_seg, activation_output)
              self.preprocessor.data_io.save_prediction(sampleObj)
            # Clean up temporary files if necessary
            if self.preprocessor.prepare_batches or self.preprocessor.prepare_subfunctions:
                self.preprocessor.data_io.batch_cleanup()
        # Output predictions results if direct output modus is active
        if return_output : return results


    #---------------------------------------------#
    #            Augmentated Prediction           #
    #---------------------------------------------#
    """ Prediction function for the Neural Network model, which utilizes augmentation on prediction data.
        The model will compute multiple predictions for a single image via flipping.

        In contrast to the standard prediction function, this one will always return a list
        of augmentated predictions with acvtivation output for a single sample.

    Args:
        sample (string):                A sample index for which a segmentation prediction will be computed.
    """
    def predict_augmentated(self, sample):
        if self.preprocessor.data_augmentation is None:
            raise ValueError("Inference Augmentation requires a " + \
                             "Data Augmentation class instance!")
        else : data_aug = self.preprocessor.data_augmentation
        # Initialize result array for the augmentated predictions
        results = []
        # Activate augmentation inferene
        data_aug.infaug = True
        if self.three_dim : flip_list = data_aug.infaug_flip_list
        else : flip_list = data_aug.infaug_flip_list[:-1]
        # Compute inference for each flip augmentation / for each axis
        for flip_axis in flip_list:
            # Update flip axis
            data_aug.infaug_flip_current = flip_axis
            # Initialize Keras Data Generator for generating batches
            dataGen = DataGenerator([sample], self.preprocessor,
                                    training=False, validation=False,
                                    shuffle=False, iterations=None)
            # Run prediction process with Keras predict
            pred_list = []
            for batch in dataGen:
                pred_batch = self.model.predict_on_batch(batch)
                pred_list.append(pred_batch)
            pred_seg = np.concatenate(pred_list, axis=0)
            # Postprocess prediction
            sampleObj = self.preprocessor.cache.pop(sample)
            pred_seg = self.preprocessor.postprocessing(sampleObj, pred_seg,
                                                        activation_output=True)
            # Backup predicted segmentation for current augmentation
            results.append(pred_seg)
        # Reset inference augmentation modus
        data_aug.infaug = False
        data_aug.infaug_flip_current = None
        # Return result array
        return results

    #---------------------------------------------#
    #                 Evaluation                  #
    #---------------------------------------------#
    """ Evaluation function for the Neural Network model using the provided lists of sample indices
        for training and validation. It is also possible to pass custom Callback classes in order to
        obtain more information.

    Args:
        training_samples (list of indices):     A list of sample indicies which will be used for training
        validation_samples (list of indices):   A list of sample indicies which will be used for validation
        epochs (integer):                       Number of epochs. A single epoch is defined as one iteration through the complete data set.
        iterations (integer):                   Number of iterations (batches) in a single epoch.
        callbacks (list of Callback classes):   A list of Callback classes for custom evaluation
    Return:
        history (Keras history object):         Gathered fitting information and evaluation results of the validation
    """
    # Evaluate the Neural Network model using the MIScnn pipeline
    def evaluate(self, training_samples, validation_samples, epochs=20,
                 iterations=None, callbacks=[], class_weight=None):
        # Initialize a Keras Data Generator for generating Training data
        dataGen_training = DataGenerator(training_samples, self.preprocessor,
                                         training=True, validation=False,
                                         shuffle=self.shuffle_batches,
                                         iterations=iterations)
        # Initialize a Keras Data Generator for generating Validation data
        dataGen_validation = DataGenerator(validation_samples,
                                           self.preprocessor,
                                           training=True, validation=True,
                                           shuffle=self.shuffle_batches)
        print("constructed data generation")
        # Run training & validation process with the Keras fit
        history = self.model.fit(dataGen_training,
                                 validation_data=dataGen_validation,
                                 callbacks=callbacks,
                                 epochs=epochs,
                                 class_weight=class_weight,
                                 workers=self.workers,
                                 max_queue_size=self.batch_queue_size)
        # Clean up temporary files if necessary
        if self.preprocessor.prepare_batches or self.preprocessor.prepare_subfunctions:
            self.preprocessor.data_io.batch_cleanup()
        # Return the training & validation history
        return history

    #---------------------------------------------#
    #               Model Management              #
    #---------------------------------------------#
    def reset(self):
        self.reset_weights()
        self.model.compile(optimizer=Adam(lr=self.learning_rate),
                           loss=self.loss, metrics=self.metrics)

    def copy(self):
        new_model = Neural_Network(self.preprocessor, self.architecture, self.loss, copy.deepcopy(self.metrics),
            self.learning_rate, self.batch_queue_size, self.workers, False) #assume multi_gpu false because mirroring wqould be expensive with multiple models
        new_model.model.set_weights(self.model.get_weights())
        return new_model

    # Re-initialize model weights
    def reset_weights(self):
        self.model.set_weights(self.initialization_weights)

    # Dump model to file
    def dump(self, file_path):
        self.model.save(file_path)

    # Load model from file
    def load(self, file_path, custom_objects={}):
        # Create model input path
        self.model = load_model(file_path, custom_objects, compile=False)
        # Compile model
        self.model.compile(optimizer=Adam(lr=self.learning_rate),
                           loss=self.loss, metrics=self.metrics)
