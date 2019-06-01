#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
from keras.models import model_from_json
import numpy
import math
#Internal libraries/scripts
import inputreader as CNNsolver_IR
from models.unet import Unet


#-----------------------------------------------------#
#                    Fixed Parameter                  #
#-----------------------------------------------------#
config = dict()
#config["image_shape"] = (None, 512, 512, 1)
config["input_shape"] = (None, 128, 128, 1)
config["classes"] = 3
config["batch_size"] = 10
config["patch_size"] = (32, 128, 128)
config["overlap"] = 4
config["max_queue_size"] = 3
config["epochs"] = 1


#-----------------------------------------------------#
#                Neural Network - Class               #
#-----------------------------------------------------#
class NeuralNetwork:
    # Initialize class variables
    model = None

    # Create a Convolutional Neural Network with Keras
    def __init__(self):
        model = Unet(input_shape=config["input_shape"],
                     n_labels=config["classes"],
                     activation_name="sigmoid")
                     #activation="softmax"
        #TODO: compile extract from model
        self.model = model

    # Train the Neural Network model on the provided case ids
    def train(self, ids, data_path):
        # Create a Input Reader instance
        reader = CNNsolver_IR.InputReader(data_path)
        # Iterate over each case
        for i in ids:
            # Load the MRI of the case
            mri = reader.case_loader(i, pickle=True)
            # Slice volume and segmentation into patches
            mri.create_patches("vol", config["patch_size"], config["overlap"])
            mri.create_patches("seg", config["patch_size"], config["overlap"])
            # Backup MRI to pickle for faster access in later epochs
            reader.mri_pickle_backup(i, mri)
            # Calculate the number of steps for the fitting
            steps = math.ceil(len(mri.patches_vol) / config["batch_size"])
            # Fit current MRI to the CNN model
            self.model.fit_generator(mri.data_generator(config["batch_size"],
                                                        steps=steps,
                                                        training=True),
                                     steps_per_epoch=steps,
                                     epochs=config["epochs"],
                                     max_queue_size=config["max_queue_size"])
        # Clean up temporary MRI pickles for training
        reader.mri_pickle_cleanup()

    # Predict with the Neural Network model on the provided case ids
    def predict(self, ids, data_path):
        results = []
        # Create a Input Reader instance
        reader = CNNsolver_IR.InputReader(data_path)
        # Iterate over each case
        for i in ids:
            # Load the MRI of the case
            mri = reader.case_loader(i, True)
            # Slice volume into patches
            mri.create_patches("vol", config["patch_size"], config["overlap"])
            # Calculate the number of steps for the prediction
            steps = math.ceil(len(mri.patches_vol) / config["batch_size"])
            # # Fit current MRI to the CNN model
            pred = self.model.predict_generator(
                                    mri.data_generator(
                                                config["batch_size"],
                                                steps,
                                                training=False),
                                    steps=steps,
                                    max_queue_size=config["max_queue_size"])
            # Transform probabilities to classes
            pred_seg = numpy.argmax(pred, axis=-1)
            # Add segmentation prediction to the MRI case object
            mri.add_segmentation(pred_seg, False)
            # Add the MRI to the results list
            results.append(mri)
        # Return final results
        return results

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
