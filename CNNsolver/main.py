#!/usr/bin/env python
# -*- coding: utf-8 -*-

#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import sys
import argparse
# Internal libraries/scripts
import neural_network as CNNsolver_NN
import evaluation as CNNsolver_EV

#-----------------------------------------------------#
#                  Parse command line                 #
#-----------------------------------------------------#
# Implement a modified ArgumentParser from the argparse package
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message + "\n")
        self.print_help()
        sys.exit(2)
# Initialize the modifed argument parser
parser = MyParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                add_help=False, description=
    """
Description...

Author: Dominink Müller
Email: dominik.mueller@informatik.uni-augsburg.de
Chair: IT-Infrastructure for Translational Medical Research- University Augsburg (Germany)
""")
# Add arguments for mutally exclusive required group
required_group = parser.add_argument_group(title='Required arguments')
required_group.add_argument('-i', '--input', type=str, action='store',
                required=True, dest='args_input', help='Path to data directory')
# Add arguments for optional group
optional_group = parser.add_argument_group(title='Optional arguments')
optional_group.add_argument('-h', '--help', action="help",
                help="Show this help message and exit")
# Parse arguments
args = parser.parse_args()

#-----------------------------------------------------#
#                   Configurations                    #
#-----------------------------------------------------#
# Define range of cases for the model training
cases = list(range(2,3))
# Configurations
config = dict()
config["data_path"] = args.args_input           # Path to the kits19 data dir
config["input_shape"] = (None, 128, 128, 1)     # Neural Network input shape
config["classes"] = 3                           # Number of output classes
config["batch_size"] = 10                       # Number of patches in on step
config["patch_size"] = (32, 128, 128)           # Patch shape/size
config["overlap"] = 4                           # Overlap in x-axis
config["max_queue_size"] = 3                    # Number of preprocessed batches
config["epochs"] = 1                            # Number of epochs for training

#-----------------------------------------------------#
#                    Runner code                      #
#-----------------------------------------------------#
# Create the Convolutional Neural Network
cnn_model = CNNsolver_NN.NeuralNetwork(config)

# Train the model
cnn_model.train(cases)
# Dump model
#cnn_model.dump("model")

# Load model
#cnn_model.load("model")
# Predict segmentation with CNN model
#res = cnn_model.predict(list(range(2,3)), path_data)

# Evaluate model
#res = []
#CNNsolver_EV.visual_evaluation(res)
#cnn_model.evaluate(list(range(3,4)), path_data)


#print(cnn_model.model.metrics_names)
