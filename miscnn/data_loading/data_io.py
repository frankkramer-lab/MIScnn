#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2020 IT-Infrastructure for Translational Medical Research,    #
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
import pickle
# Internal libraries/scripts
import miscnn.data_loading.sample as MIScnn_sample

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
        delete_batchDir (boolean):  Boolean if the whole temporary batch directory for prepared batches should be deleted after
                                    model utilization. If false only the batches with the associated seed will be deleted.
                                    This parameter is important when running multiple instances of MIScnn.
    """
    def __init__(self, interface, input_path, output_path="predictions",
                 batch_path="batches", delete_batchDir=True):
        # Parse parameter
        self.interface = interface
        self.input_path = input_path
        self.output_path = output_path
        self.batch_path = batch_path
        self.delete_batchDir = delete_batchDir
        # Initialize Data I/O interface
        self.indices_list = interface.initialize(input_path)

    #---------------------------------------------#
    #                Sample Loader                #
    #---------------------------------------------#
    # Load a sample from the data set
    def sample_loader(self, index, load_seg=True, load_pred=False, backup=False):
        # If sample is a backup -> load it from pickle
        if backup : return self.load_sample_pickle(index)
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
        # Add optional details to the sample object
        sample.add_details(self.interface.load_details(index))
        # Return sample object
        return sample

    #---------------------------------------------#
    #              Prediction Backup              #
    #---------------------------------------------#
    # Save a segmentation prediction
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
            batch_img_path = os.path.join(self.batch_path, str(self.seed) + \
                                          ".batch_img" + "." + str(pointer))
            np.savez(batch_img_path, data=batch_img)
        # Backup segmentation batch
        if batch_seg is not None:
            batch_seg_path = os.path.join(self.batch_path, str(self.seed) + \
                                          ".batch_seg" + "." + str(pointer))
            np.savez_compressed(batch_seg_path, data=batch_seg)

    # Load a batch from a npz file for fast access
    def batch_load(self, pointer, img=True):
        # Identify batch type (image or segmentation)
        if img:
            batch_type = "batch_img"
        else:
            batch_type = "batch_seg"
        # Set up file path
        in_path = os.path.join(self.batch_path, str(self.seed) + "." + \
                               batch_type + "." + str(pointer) + ".npz")
        # Load numpy array from file
        batch = np.load(in_path)["data"]
        # Return loaded batch
        return batch

    # Clean up all temporary npz files
    def batch_cleanup(self, pointer=None):
        # If a specific pointer is provided -> only delete this batch
        if pointer != None:
            # Define path to image file
            img_file = os.path.join(self.batch_path, str(self.seed) + \
                                    ".batch_img" + "." + str(pointer) + \
                                    ".npz")
            # Delete image file
            if os.path.exists(img_file):
                os.remove(img_file)
            # Define path to segmentation file
            seg_file = os.path.join(self.batch_path, str(self.seed) + \
                                    ".batch_seg" + "." + str(pointer) + \
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
                if pointer == None and match(str(self.seed) + "\.*", file) is not None:
                    os.remove(os.path.join(self.batch_path, file))
            # Delete complete batch directory
            if self.delete_batchDir:
                shutil.rmtree(self.batch_path)

    #---------------------------------------------#
    #                Sample Backup                #
    #---------------------------------------------#
    # Backup samples for later access
    def backup_sample(self, sample):
        if not os.path.exists(self.batch_path) : os.mkdir(self.batch_path)
        sample_path = os.path.join(self.batch_path, str(self.seed) + "." + \
                                   sample.index + ".pickle")
        if not os.path.exists(sample_path):
            with open(sample_path, 'wb') as handle:
                pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Load a backup sample from pickle
    def load_sample_pickle(self, index):
        sample_path = os.path.join(self.batch_path, str(self.seed) + "." + \
                                   index + ".pickle")
        with open(sample_path,'rb') as reader:
            sample = pickle.load(reader)
        return sample

    #---------------------------------------------#
    #               Variable Access               #
    #---------------------------------------------#
    def get_indiceslist(self):
        return self.indices_list.copy()

#-----------------------------------------------------#
#               Evaluation Data Backup                #
#-----------------------------------------------------#
# Backup history evaluation as TSV (Tab Separated File) on disk
def backup_history(history, evaluation_path):
    # Opening file writer
    output_path = os.path.join(evaluation_path, "history.tsv")
    with open(output_path, "w") as fw:
        # Write the header
        header = "epoch" + "\t" + "\t".join(history.keys()) + "\n"
        fw.write(header)
        # Write data rows
        zipped_data = list(zip(*history.values()))
        for i in range(0, len(history["loss"])):
            line = str(i+1) + "\t" + "\t".join(map(str, zipped_data[i])) + "\n"
            fw.write(line)

# Backup evaluation as TSV (Tab Separated File)
def backup_evaluation(data, evaluation_path, start=False):
    # Set up the evaluation directory
    if start and not os.path.exists(evaluation_path):
        os.mkdir(evaluation_path)
    # Define the writing type
    if start : writer_type = "w"
    else : writer_type = "a"
    # Opening file writer
    output_path = os.path.join(evaluation_path, "detailed_validation.tsv")
    with open(output_path, writer_type) as fw:
        # Join the data together to a row
        line = "\t".join(map(str, data)) + "\n"
        fw.write(line)

# Create an evaluation subdirectory and change path
def create_directories(eval_path, subeval_path=None):
    # Create evaluation directory if necessary
    if not os.path.exists(eval_path):
        os.mkdir(eval_path)
    # Create evaluation subdirectory if necessary
    if subeval_path is not None:
        # Concatenate evaluation subdirectory path if present
        subdir = os.path.join(eval_path, subeval_path)
        # Set up the evaluation subdirectory
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        # Return path to evaluation subdirectory
        return subdir
    # Return path to evaluation directory
    else : return eval_path
