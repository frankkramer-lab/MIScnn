#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2019 IT-Infrastructure for Translational Medical Research,    #
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
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import os
from re import match
import numpy as np
import random
import shutil
# Internal libraries/scripts
import miscnn.data_processing.sample as MIScnn_sample

#-----------------------------------------------------#
#                    Data IO class                    #
#-----------------------------------------------------#
# Class to handle all input and output functionality
class Data_IO:
    # Class variables
    interface = None                    # Data I/O interface
    input_path = None                   # Path to input data directory
    output_path = None                  # Path to MIScnn prediction directory
    batch_path = None                   # Path to temporary batch storage directory
    evaluation_path = None              # Path to evaluation results directory
    indices_list = None                 # List of sample indices after data set initialization
    delete_batchDir = None              # Boolean for deletion of complete tmp batches directory
                                        # or just the batch data for the current seed
    seed = random.randint(0,99999999)   # Random seed if running multiple MIScnn instances

    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    """ Initialization function for creating an object of the Data IO class.
    This class provides functionality for handling all input and output processes
    of the imaging data, as well as the temporary backup of batches to the disk.

    The user is only required to create an instance of the Data IO class with the desired specifications
    and IO interface for the correct format. It is possible to create a custom IO interface for handling
    special data structures or formats.

    Args:
        interface (io_interface):   A data IO interface which inherits the abstract_io class with the following methods:
                                    initialize, load_image, load_segmentation, load_prediction, save_prediction
        input_path (string):        Path to the input data directory, in which all imaging data have to be accessible.
        output_path (string):       Path to the output data directory, in which computed predictions will be stored. This directory
                                    will be created.
        batch_path (string):        Path to the batch data directory. This directory will be created and used for temporary files.
        evaluation_path (string):   Path to the evaluation data directory. This directory will be created and used for storing
                                    all kinds of evaluation results during validation processes.
    """
    def __init__(self, interface, input_path, output_path="predictions",
                 batch_path="batches", evaluation_path="evaluation",
                 delete_batchDir=True):
        # Parse parameter
        self.interface = interface
        self.input_path = input_path
        self.output_path = output_path
        self.batch_path = batch_path
        self.evaluation_path = evaluation_path
        self.delete_batchDir = delete_batchDir
        # Initialize Data I/O interface
        self.indices_list = interface.initialize(input_path)

    #---------------------------------------------#
    #                Sample Loader                #
    #---------------------------------------------#
    # Load a sample from the data set
    def sample_loader(self, index, load_seg=True, load_pred=False):
        # Load the image with the I/O interface
        image = self.interface.load_image(index)
        # Create a Sample object
        sample = MIScnn_sample.Sample(index, image, self.interface.channels,
                                      self.interface.classes)
        # IF needed read the provided segmentation for current sample
        if load_seg:
            segmentation = self.interface.load_segmentation(index)
            sample.add_segmentation(segmentation)
        # IF needed read the provided prediction for current sample
        if load_pred:
            prediction = self.interface.load_prediction(index, self.output_path)
            sample.add_prediction(prediction)
        # Return sample object
        return sample

    #---------------------------------------------#
    #              Prediction Backup              #
    #---------------------------------------------#
    # Save a segmentation prediction in NIFTI format
    def save_prediction(self, pred, index):
        # Create the output directory if not existent
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        # Backup the prediction
        self.interface.save_prediction(pred, index, self.output_path)

    #---------------------------------------------#
    #               Batch Management              #
    #---------------------------------------------#
    # Backup batches to npz files for fast access later
    def backup_batches(self, batch_img, batch_seg, pointer):
        # Create batch directory of not existent
        if not os.path.exists(self.batch_path):
            os.mkdir(self.batch_path)
        # Backup image batch
        if batch_img is not None:
            batch_img_path = os.path.join(self.batch_path, "batch_img." + \
                                          str(self.seed) + "." + str(pointer))
            np.savez(batch_img_path, data=batch_img)
        # Backup segmentation batch
        if batch_seg is not None:
            batch_seg_path = os.path.join(self.batch_path, "batch_seg." + \
                                          str(self.seed) + "." + str(pointer))
            np.savez_compressed(batch_seg_path, data=batch_seg)

    # Load a batch from a npz file for fast access
    def batch_load(self, pointer, img=True):
        # Identify batch type (image or segmentation)
        if img:
            batch_type = "batch_img"
        else:
            batch_type = "batch_seg"
        # Set up file path
        in_path = os.path.join(self.batch_path, batch_type + "." + \
                               str(self.seed) + "." + str(pointer) + ".npz")
        # Load numpy array from file
        batch = np.load(in_path)["data"]
        # Return loaded batch
        return batch

    # Clean up all temporary npz files
    def batch_npz_cleanup(self, pointer=None):
        # If a specific pointer is provided -> only delete this batch
        if pointer != None:
            # Define path to image file
            img_file = os.path.join(self.batch_path, "batch_img." + \
                                    str(self.seed) + "." + str(pointer) + \
                                    ".npz")
            # Delete image file
            if os.path.exists(img_file):
                os.remove(img_file)
            # Define path to segmentation file
            seg_file = os.path.join(self.batch_path, "batch_seg." + \
                                    str(self.seed) + "." + str(pointer) + \
                                    ".npz")
            # Delete segmentation file
            if os.path.exists(seg_file):
                os.remove(seg_file)
        # If no pointer is provided, delete all batches of this MIScnn instance
        elif pointer == None:
            # Iterate over each file in the batch directory
            directory = os.listdir(self.batch_path)
            for file in directory:
                # IF file matches seed pattern -> delete it
                if pointer == None and match("batch_[a-z]+\." + \
                    str(self.seed) + "\.*", file) is not None:
                    os.remove(os.path.join(self.batch_path, file))
            # Delete complete batch directory
            if self.delete_batchDir:
                shutil.rmtree(self.batch_path)

    #---------------------------------------------#
    #               Variable Access               #
    #---------------------------------------------#
    def get_indiceslist(self):
        return self.indices_list.copy()

# #-----------------------------------------------------#
# #               Evaluation Data Backup                #
# #-----------------------------------------------------#
# # Backup evaluation as TSV (Tab Separated File)
# def save_evaluation(data, directory, file, start=False):
#     # Set up the evaluation directory
#     if start and not os.path.exists(directory):
#         os.mkdir(directory)
#     # Define the writing type
#     if start:
#         writer_type = "w"
#     else:
#         writer_type = "a"
#     # Opening file writer
#     output_path = os.path.join(directory, file)
#     with open(output_path, writer_type) as fw:
#         # Join the data together to a row
#         line = "\t".join(map(str, data)) + "\n"
#         fw.write(line)
#
# # Create an evaluation subdirectory and change path
# def update_evalpath(updated_path, eval_path):
#     # Create evaluation directory if necessary
#     if not os.path.exists(eval_path):
#         os.mkdir(eval_path)
#     # Concatenate evaluation subdirectory path
#     subdir = os.path.join(eval_path, updated_path)
#     # Set up the evaluation subdirectory
#     if not os.path.exists(subdir):
#         os.mkdir(subdir)
#     # Return updated path to evaluation subdirectory
#     return subdir
