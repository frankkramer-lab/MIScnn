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
                cb_model = ModelCheckpoint(os.path.join(out_dir, "model.hdf5"),
                                           monitor="val_loss", verbose=1,
                                           save_best_only=True, mode="min")
                cb_list = callbacks + [cb_model]
            else:
                cb_list = callbacks
            model.reset()
            model.evaluate(training, validation, evaluation_path=out_dir, epochs=epochs, iterations=iterations, callbacks=callbacks)