#==============================================================================#
#  Author:       Philip Meyer                                                  #
#  Copyright:    2021 IT-Infrastructure for Translational Medical Research,    #
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
from miscnn.multi_model.model import Model
from miscnn.neural_network.model import Neural_Network
import json
from miscnn.data_loading.data_io import create_directories
from tensorflow.keras.callbacks import ModelCheckpoint
import os

#-----------------------------------------------------#
#                  Model Group class                  #
#-----------------------------------------------------#
# This class exposes the model functionality using multiple sub-models as components.
class Model_Group(Model):

    """ Initialization function for creating a Model Group object. This object will train and predict sub models.
        The predictions are merged using an aggregation function.

    Args:
        models (Collection of Models):          The list of models that the Model Group should use.
        preprocessor (Preprocessor):            Preprocessor class that the Model Group should refer to for pipeline structure.
                                                This does not necissarily mean that all models share that preprocessor.
        verify_preprocessor (Boolean):          Enable checking whether all models share the preprocessor of the model group.
                                                EWnabled by default. Disable to use models in combination with different preprocessing methods.
    """
    def __init__ (self, models, preprocessor, verify_preprocessor=True):
        Model.__init__(self, preprocessor)

        self.models = models

        #Verify the properties of the list.
        for model in self.models:
            #Check early if all items in the collection are indeed models.
            if (not isinstance(model, Model)):
                raise RuntimeError("Model Groups can only be comprised of objects inheriting from Model")
            if verify_preprocessor and not model.preprocessor == self.preprocessor:
                raise RuntimeError("not all models use the same preprocessor. This can have have unintended effects and instabilities. To disable this warning pass \"verify_preprocessor=False\"")

    #---------------------------------------------#
    #                  Training                   #
    #---------------------------------------------#
    """ Fitting function for the Model Group using the provided list of sample indices.

    Args:
        sample_list (list of indices):          A list of sample indicies which will be used for training
        epochs (integer):                       Number of epochs. A single epoch is defined as one iteration through
                                                the complete data set.
        iterations (integer):                   Number of iterations (batches) in a single epoch.
        callbacks (list of Callback classes):   A list of Callback classes for custom evaluation.
        class_weight (dictionary or list):      A list or dictionary of float values to handle class unbalance.
    """
    def train(self, sample_list, epochs=20, iterations=None, callbacks=[], class_weight=None):
        #Get the root path from model group preprocessor
        tmp = self.preprocessor.data_io.output_path

        for model in self.models:
            #Create subdirectories and reroute each model to their respective one
            out_dir = create_directories(tmp, "group_" + str(model.id))
            model.preprocessor.data_io.output_path = out_dir
            cb_list = []
            #If the child is not an instance of model group it is a single model. So a storage callback can be registered to this leaf.
            if (not isinstance(model, Model_Group)):
                #this child is a leaf. ensure correct storage.
                cb_model = ModelCheckpoint(os.path.join(out_dir, "model.hdf5"),
                                           monitor="val_loss", verbose=1,
                                           save_best_only=True, mode="min")
                cb_list = callbacks + [cb_model]
            else:
                cb_list = callbacks

            # Reset Neural Network model weights. This is to ensure tensorflow state of the model. As the same model can be used multiple times as a child.
            model.reset()
            model.train(sample_list, epochs=epochs, iterations=iterations, callbacks=cb_list, class_weight=class_weight)

    #---------------------------------------------#
    #                 Prediction                  #
    #---------------------------------------------#
    """ Prediction function for the Neural Network model. The fitted model will predict a segmentation
        for the provided list of sample indices. Due to the limited capacity of memory this involves a considerable amount of loading and saving.

    Args:
        sample_list (list of indices):  A list of sample indicies for which a segmentation prediction will be computed.
        return_output (boolean):        Parameter which decides, if computed predictions will be output as the return of this
                                        function or if the predictions will be saved with the save_prediction method defined
                                        in the provided Data I/O interface.
        aggregation_func (function):    Function that accepts a sample object and a list of predictions. It is expected to return
                                        a merged prediction.
        activation_output (boolean):    Parameter which decides, if model output (activation function, normally softmax) will
                                        be saved/outputed (if FALSE) or if the resulting class label (argmax) should be outputed.
    """
    def predict(self, sample_list, aggregation_func, activation_output=False):
        #Get the root path from model group preprocessor
        tmp = self.preprocessor.data_io.output_path
        for model in self.models:
            #Execute a predictions on all submodels. it is assumed that they store their result.
            out_dir = create_directories(tmp, "group_" + str(model.id))
            model.preprocessor.data_io.output_path = out_dir
            model.predict(sample_list, activation_output=activation_output)

        #Handle merging of all samples after prediction.
        for sample in sample_list:
            s = None
            prediction_list = []

            #Load predictions for all models. and collect them
            for model in self.models:
                out_dir = os.path.join(tmp, "group_" + str(model.id))
                model.preprocessor.data_io.output_path = out_dir
                s = model.preprocessor.data_io.sample_loader(sample, load_seg=False, load_pred=True)
                prediction_list.append(s.pred_data)
            #Aggregate the predictions. Then store the result.
            s = self.preprocessor.data_io.sample_loader(sample, load_seg=False, load_pred=False)
            res = aggregation_func(s, prediction_list)
            self.preprocessor.data_io.output_path = tmp #preprocessor is likely a reference so this needs to be reset
            s.pred_data = res
            self.preprocessor.data_io.save_prediction(s)

    #---------------------------------------------#
    #                 Evaluation                  #
    #---------------------------------------------#
    """ Evaluation function for the model group using the provided lists of sample indices
        for training and validation. It is also possible to pass custom Callback classes in order to
        obtain more information.

    Args:
        training_samples (list of indices):     A list of sample indicies which will be used for training
        validation_samples (list of indices):   A list of sample indicies which will be used for validation
        evaluation_path (string):               The base path for the evaluation.
        epochs (integer):                       Number of epochs. A single epoch is defined as one iteration through the complete data set.
        iterations (integer):                   Number of iterations (batches) in a single epoch.
        callbacks (list of Callback classes):   A list of Callback classes for custom evaluation
    """
    def evaluate(self, training_samples, validation_samples, evaluation_path="evaluation", epochs=20, iterations=None, callbacks=[]):
        for model in self.models:
            #Select path for each submodel and executte evaluation in.
            out_dir = create_directories(evaluation_path, "group_" + str(model.id))
            model.preprocessor.data_io.output_path = out_dir
            cb_list = []
            #If the child is not an instance of model group it is a single model. So a storage callback can be registered to this leaf.
            if (not isinstance(model, Model_Group)):
                #this child is a leaf. ensure correct storage.
                cb_model = ModelCheckpoint(os.path.join(out_dir, "model.hdf5"),
                                           monitor="val_loss", verbose=1,
                                           save_best_only=True, mode="min")
                cb_list = callbacks + [cb_model]
            else:
                cb_list = callbacks
            model.reset()
            model.evaluate(training_samples, validation_samples, evaluation_path=out_dir, epochs=epochs, iterations=iterations, callbacks=cb_list)
        self.preprocessor.data_io.output_path = evaluation_path

    def reset(self):
        for model in self.models: #propagate
            model.reset()

    def copy(self):
        return ModelGroup([model.copy() for model in self.models], self.preprocessor, False) #validity is implied and it accelerates cloning

    # Dump model to file
    def dump(self, file_path):
        subdir = create_directories(file_path, "group_" + str(self.id))

        with open(os.path.join(subdir, "metadata.json"), "w") as f:
            json.dump({"type": "group"}, f)

        for model in self.models:
            model.dump(subdir)

    # Load model from file
    def load(self, file_path, custom_objects={}):
        collection = [dir for dir in os.listdir(file_path) if os.path.isdir(dir)]

        for obj in collection:
            metadata = {}
            metadata_path = os.path.join(obj, "metadata.json")
            if os.exists(metadata_path):
                with open(metadata_path, "w") as f:
                    metadata = json.load(f)
            model = None
            if "type" in metadata.keys() and metadata["type"] == "group":
                model = ModelGroup([], self.preprocessor)
            else:
                model = Neural_Network(self.preprocessor) #needs handling of arbitrary types. some sort of model agent
            model.load(obj)
            self.models.append(model)
