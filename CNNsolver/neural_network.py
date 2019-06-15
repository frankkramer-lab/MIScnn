#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
from keras.models import model_from_json
from keras.optimizers import Adam
import numpy
import math
#Internal libraries/scripts
import inputreader as CNNsolver_IR
from preprocessing import preprocessing_MRIs, data_generator
from utils.matrix_operations import concat_3Dmatrices
from models.unet_muellerdo import Unet
from models.metrics import dice_coefficient, dice_classwise, tversky_loss

#-----------------------------------------------------#
#                Neural Network - Class               #
#-----------------------------------------------------#
class NeuralNetwork:
    # Initialize class variables
    model = None
    config = None
    metrics = [dice_coefficient, dice_classwise,
              'categorical_accuracy', 'categorical_crossentropy']


    # Create a Convolutional Neural Network with Keras
    def __init__(self, config):
        model = Unet(input_shape=config["input_shape"],
                     n_labels=config["classes"],
                     activation="softmax")
        model.compile(optimizer=Adam(lr=config["learninig_rate"]),
                      loss=tversky_loss,
                      metrics=self.metrics)
        self.model = model
        self.config = config

    # Train the Neural Network model on the provided case ids
    def train(self, cases):
        # Preprocess Magnetc Resonance Images
        casePointer = preprocessing_MRIs(cases, self.config, training=True,
                                         skip_blanks=self.config["skip_blanks"])
        # Run training process with the Keras fit_generator
        self.model.fit_generator(data_generator(casePointer,
                                                self.config["data_path"],
                                                training=True),
                                 steps_per_epoch=len(casePointer),
                                 epochs=self.config["epochs"],
                                 max_queue_size=self.config["max_queue_size"])
        # Clean up temporary MRI pickles for training
        CNNsolver_IR.mri_pickle_cleanup()

    # Predict with the Neural Network model on the provided case ids
    def predict(self, cases):
        # Create a Input Reader instance
        reader = CNNsolver_IR.InputReader(self.config["data_path"])
        # Iterate over each case
        for id in cases:
            # Preprocess Magnetc Resonance Images
            casePointer = preprocessing_MRIs([id], self.config, training=False)
            # Run prediction process with the Keras predict_generator
            pred_seg = self.model.predict_generator(
                                data_generator(casePointer,
                                               self.config["data_path"],
                                               training=False),
                                steps=len(casePointer),
                                max_queue_size=self.config["max_queue_size"])
            # Reload pickled MRI object from disk to cache
            mri = reader.case_loader(id, load_seg=False, pickle=True)
            # Concatenate patches into a single 3D matrix back
            pred_seg = concat_3Dmatrices(patches=pred_seg,
                                         image_size=mri.vol_data.shape,
                                         window=self.config["patch_size"],
                                         overlap=self.config["overlap"])
            # Transform probabilities to classes
            pred_seg = numpy.argmax(pred_seg, axis=-1)
            # Add segmentation prediction to the MRI case object
            mri.add_segmentation(pred_seg, truth=False)
            # Backup MRI to pickle
            reader.mri_pickle_backup(id, mri)

    # Evaluate the Neural Network model on the provided case ids
    def evaluate(self, cases):
        # Create a Input Reader instance
        reader = CNNsolver_IR.InputReader(self.config["data_path"])
        # Iterate over each case
        for id in cases:
            # Preprocess Magnetc Resonance Images
            casePointer = preprocessing_MRIs([id], self.config, training=True)
            # Run Evaluation process with the Keras predict_generator
            result = self.model.evaluate_generator(
                                data_generator(casePointer,
                                               self.config["data_path"],
                                               training=True),
                                steps=len(casePointer),
                                max_queue_size=self.config["max_queue_size"])
            # Output evaluation result
            print(str(id) + "\t" + str(result))
            # Run prediction process with the Keras predict_generator
            pred_seg = self.model.predict_generator(
                                data_generator(casePointer,
                                               self.config["data_path"],
                                               training=False),
                                steps=len(casePointer),
                                max_queue_size=self.config["max_queue_size"])
            # Reload pickled MRI object from disk to cache
            mri = reader.case_loader(id, load_seg=True, pickle=True)
            # Concatenate patches into a single 3D matrix back
            pred_seg = concat_3Dmatrices(patches=pred_seg,
                                         image_size=mri.vol_data.shape,
                                         window=self.config["patch_size"],
                                         overlap=self.config["overlap"])
            # Transform probabilities to classes
            pred_seg = numpy.argmax(pred_seg, axis=-1)
            # Add segmentation prediction to the MRI case object
            mri.add_segmentation(pred_seg, truth=False)
            # Backup MRI to pickle
            reader.mri_pickle_backup(id, mri)

    # Dump model to file
    def dump(self, path):
        # Serialize model to JSON
        model_json = self.model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        # Serialize weights to HDF5
        self.model.save_weights("model/weights.h5")

    # Load model from file
    def load(self, path):
        # Load json and create model
        json_file = open('model/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # Load weights into new model
        self.model.load_weights("model/weights.h5")
        # Compile model
        self.model.compile(optimizer=Adam(lr=self.config["learninig_rate"]),
                           loss=tversky_loss,
                           metrics=self.metrics)
