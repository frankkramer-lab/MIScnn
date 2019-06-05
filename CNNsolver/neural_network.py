#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
from keras.models import model_from_json
import numpy
import math
#Internal libraries/scripts
import inputreader as CNNsolver_IR
from preprocessing import preprocessing_MRIs, data_generator
from utils.matrix_operations import concat_3Dmatrices
from models.unet import Unet

#-----------------------------------------------------#
#                Neural Network - Class               #
#-----------------------------------------------------#
class NeuralNetwork:
    # Initialize class variables
    model = None
    config = None

    # Create a Convolutional Neural Network with Keras
    def __init__(self, config):
        model = Unet(input_shape=config["input_shape"],
                     n_labels=config["classes"],
                     activation_name="sigmoid")
                     #activation="softmax"
        #TODO: compile extract from model
        self.model = model
        self.config = config

    # Train the Neural Network model on the provided case ids
    def train(self, cases):
        # Preprocess Magnetc Resonance Images
        casePointer = preprocessing_MRIs(cases, self.config, training=True)
        # Run training process with the Keras fit_generator
        self.model.fit_generator(data_generator(casePointer,
                                                self.config["data_path"],
                                                self.config["classes"],
                                                training=True),
                                 steps_per_epoch=len(casePointer),
                                 epochs=self.config["epochs"],
                                 max_queue_size=self.config["max_queue_size"])
        # Clean up temporary MRI pickles for training
        CNNsolver_IR.mri_pickle_cleanup()


# WORK IN PROGRESS!!!
    # # Predict with the Neural Network model on the provided case ids
    # def predict(self, cases):
    #     # Iterate over each case
    #     for id in cases:
    #         # Preprocess Magnetc Resonance Images
    #         casePointer = preprocessing_MRIs([id], self.config, training=False)
    #         # Run prediction process with the Keras predict_generator
    #         pred = self.model.predict_generator(
    #                             data_generator(casePointer,
    #                                            self.config["data_path"],
    #                                            self.config["classes"],
    #                                            training=False),
    #                             steps=len(casePointer),
    #                             max_queue_size=self.config["max_queue_size"])
    #         # Transform probabilities to classes
    #         pred_seg = numpy.argmax(pred, axis=-1)
    #         # Concatenate patches into a single 3D matrix back
    #         pred_seg = concat_3Dmatrices(pred_seg,
    #
    #                                      self.config["
    #                                      image_size, window, overlap)
    #
    #                                      patches, image_size, window, overlap
    #         # Add segmentation prediction to the MRI case object
    #         mri.add_segmentation(pred_seg, truth=False)
    #         # Add the MRI to the results list
    #
    #     # Create a Input Reader instance
    #     reader = CNNsolver_IR.InputReader(data_path)
    #     # Iterate over each case
    #     for i in ids:
    #         # Load the MRI of the case
    #         mri = reader.case_loader(i, load_seg=False, pickle=True)
    #         # Slice volume into patches
    #         mri.create_patches("vol", config["patch_size"], config["overlap"])
    #         # Calculate the number of steps for the prediction
    #         steps = math.ceil(len(mri.patches_vol) / config["batch_size"])
    #         # # Fit current MRI to the CNN model
    #         pred = self.model.predict_generator(
    #                                 mri.data_generator(
    #                                             config["batch_size"],
    #                                             steps,
    #                                             training=False),
    #                                 steps=steps,
    #                                 max_queue_size=config["max_queue_size"])
    #         # Transform probabilities to classes
    #         pred_seg = numpy.argmax(pred, axis=-1)
    #         # Concatenate patches into a single 3D matrix back
    #         # Add segmentation prediction to the MRI case object
    #         mri.add_segmentation(pred_seg, truth=False)
    #         # Add the MRI to the results list
    #         results.append(mri)
    #     # Return final results
    #     return results

    # # Evaluate the Neural Network model on the provided case ids
    # def evaluate(self, ids, data_path):
    #     # Create a Input Reader
    #     reader = CNNsolver_IR.InputReader(data_path)
    #     # Iterate over each case
    #     for i in ids:
    #         # Load the MRI of the case
    #         mri = reader.case_loader(i)
    #         # Calculate the number of steps for the fitting
    #         steps = math.ceil(mri.size / batch_size)
    #         # Fit current MRI to the CNN model
    #         score1, score2 = self.model.evaluate_generator(
    #             mri.generator_train(batch_size, steps),
    #             steps=steps, max_queue_size=max_queue_size)
    #         print(str(i) + "\t" + str(score1) + "\t" + str(score2))

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
        #self.model.compile('Adam', 'categorical_crossentropy', ['accuracy'])
