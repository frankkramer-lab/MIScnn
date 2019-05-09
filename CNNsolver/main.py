#!/usr/bin/env python
# -*- coding: utf-8 -*-

#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import sys
import argparse
# Internal libraries/scripts
import inputreader as CNNsolver_IR
import neural_network as CNNsolver_NN

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
#                     Parameters                      #
#-----------------------------------------------------#
# Path to the kits19 data
path_data = args.args_input

#-----------------------------------------------------#
#                    Runner code                      #
#-----------------------------------------------------#
# Create the Convolutional Neural Network
cnn_model = CNNsolver_NN.NeuralNetwork()

# Train the model
cnn_model.train(list(range(0,3)), path_data)
# Dump model
cnn_model.dump("model")

# Load model
#cnn_model.load("model")

# Evaluate model
#res = cnn_model.predict(list(range(4,5)), path_data)
#print(res)
