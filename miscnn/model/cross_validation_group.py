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

from miscnn.model.model_group import Model_Group
import numpy as np
from miscnn.data_loading.data_io import create_directories
from tensorflow.keras.callbacks import ModelCheckpoint
import os

class CrossValidationGroup(Model_Group):
    
    def __init__(self, model, preprocessor, folds, verify_preprocessor=True):
        modelList = [model] + [model.copy() for i in range(folds)]
        Model_Group.__init__(self, modelList, preprocessor, verify_preprocessor)
        self.folds = folds
    
    def evaluate(self, samples, evaluation_path="evaluation", epochs=20, iterations=None, callbacks=[], store=True, *args, **kwargs):
        samples_permuted = np.random.permutation(samples)
        # Split sample list into folds
        folds = np.array_split(samples_permuted, self.folds)
        fold_indices = list(range(len(folds)))
        # Start cross-validation
        
        self.preprocessor.data_io.output_path = evaluation_path
        
        for i in range(self.folds): #code is redundant to model group somehow clean
            model = self.models[i]
            training = np.concatenate([folds[x] for x in fold_indices if x!=i],
                                      axis=0)
            validation = folds[i]
            print(training, validation)
            out_dir = create_directories(evaluation_path, "group_" + str(model.id))
            model.preprocessor.data_io.output_path = out_dir
            cb_list = []
            if (not isinstance(model, Model_Group)):
                #this child is a leaf. ensure correct storage.
                path = os.path.join(out_dir, "model.hdf5")
                cb_model = ModelCheckpoint(path,
                                           monitor="val_loss", verbose=1,
                                           save_best_only=True, mode="min")
                cb_list = callbacks + [cb_model]
                print("registering model store path to: " + str(path))
            else:
                cb_list = callbacks
            
            model.reset()
            model.evaluate(training, validation, evaluation_path=out_dir, epochs=epochs, iterations=iterations, callbacks=cb_list)
        
        self.preprocessor.data_io.output_path = evaluation_path