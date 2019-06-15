#!/usr/bin/env python
# -*- coding: utf-8 -*-

#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import sys
import argparse
import os
import tensorflow as tf
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
optional_group.add_argument('-v', '--verbose', action='store_true',
                default=False, dest='args_verbose',
                help="Print all informations and warnings")
optional_group.add_argument('-h', '--help', action="help",
                help="Show this help message and exit")
# Parse arguments
args = parser.parse_args()

#-----------------------------------------------------#
#                   Configurations                    #
#-----------------------------------------------------#
config = dict()
# Dataset
config["cases"] = list(range(2,3))
config["data_path"] = args.args_input           # Path to the kits19 data dir
# Neural Network Architecture
config["input_shape"] = (None, 16, 16, 1)       # Neural Network input shape
config["patch_size"] = (16, 16, 16)             # Patch shape/size
config["classes"] = 3                           # Number of output classes
# Training
config["batch_size"] = 3                        # Number of patches in on step
config["epochs"] = 1                            # Number of epochs for training
config["max_queue_size"] = 3                    # Number of preprocessed batches
config["learninig_rate"] = 0.00001              # Learninig rate for the training
# Data Augmentation
config["overlap"] = (0,0,0)                     # Overlap in (x,y,z)-axis
config["skip_blanks"] = True                    # Skip patches with only background
config["scale_input_values"] = False            # Scale volume values to [0,1]

#-----------------------------------------------------#
#                    Runner code                      #
#-----------------------------------------------------#
# Create the Convolutional Neural Network
cnn_model = CNNsolver_NN.NeuralNetwork(config)

# Train the model
cnn_model.train(config["cases"])
# Dump model
#cnn_model.dump("model")

# Load model
#cnn_model.load("model")
# Predict segmentation with CNN model
#cnn_model.evaluate(list(range(3,4)))

# Evaluate model
#CNNsolver_EV.visual_evaluation(list(range(2,3)), config["data_path"])
#cnn_model.evaluate(list(range(3,4)), path_data)


#print(cnn_model.model.metrics_names)
