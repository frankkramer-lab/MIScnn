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


#==============================================================================#
#  Abstract Model Type. The purpose of this class is to define shared          #
#  properties that a prediction component of MIScnn should expose. It          #
#  functions as an interface for the the pipeline, separating it from more     #
#  in depth prediction logic and structure.                                    #
#==============================================================================#
class Model():
    #Static Counter providing a unique identifier for each initialized model. this is to account for multiple models and predictions per run.
    #Furthermore this allows for models to be moved without issue as the uniqueness is global.
    __CNTR = 0
    
    """ Initialization function for creating a Model object. This object essentially just wraps the preprocessor.

    Args:
        preprocessor (Preprocessor):            Preprocessor class instance which provides the Neural Network with batches.
    """
    def __init__ (self, preprocessor):
        self.preprocessor = preprocessor
        # Identify data parameters
        self.three_dim = preprocessor.data_io.interface.three_dim
        self.channels = preprocessor.data_io.interface.channels
        self.classes = preprocessor.data_io.interface.classes
        
        self.id = Model.__CNTR
        Model.__CNTR += 1 #technically not thread safe
        
    
    #train using a collection of samples
    def train(self, sample_list, epochs=20, iterations=None, callbacks=[], class_weight=None):
        raise NotImplementedError()
    
    #predict on a collection of samples
    def predict(self, sample_list, activation_output=False):
        raise NotImplementedError()
    # Evaluate the Model using the MIScnn pipeline
    def evaluate(self, training_samples, validation_samples, evaluation_path="evaluation", epochs=20, iterations=None, callbacks=[], store=True):
        raise NotImplementedError()
    
    #reset model
    def reset(self):
        raise NotImplementedError()
    
    #clone model. This is assumed to be a deep copy
    def copy(self):
        raise NotImplementedError()
    
    # Dump model to file
    def dump(self, file_path):
        raise NotImplementedError()
    # Load model from file
    def load(self, file_path, custom_objects={}):
        raise NotImplementedError()
    