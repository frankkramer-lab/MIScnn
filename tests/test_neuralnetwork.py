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
#External libraries
import unittest
import tempfile
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
#Internal libraries
from miscnn import Data_IO, Preprocessor
from miscnn.data_loading.interfaces import Dictionary_interface

#-----------------------------------------------------#
#              Unittest: Neural Network               #
#-----------------------------------------------------#
class PreprocessorTEST(unittest.TestCase):
    # Create random imaging and segmentation data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Create 2D imgaging and segmentation data set
        self.dataset2D = dict()
        for i in range(0, 10):
            img = np.random.rand(16, 16) * 255
            self.img = img.astype(int)
            seg = np.random.rand(16, 16) * 2
            self.seg = seg.astype(int)
            self.dataset2D["TEST.sample_" + str(i)] = (self.img, self.seg)
        # Initialize Dictionary IO Interface
        io_interface2D = Dictionary_interface(self.dataset2D, classes=3,
                                              three_dim=False)
        # Initialize temporary directory
        self.tmp_dir2D = tempfile.TemporaryDirectory(prefix="tmp.miscnn.")
        tmp_batches = os.path.join(self.tmp_dir2D.name, "batches")
        # Initialize Data IO
        self.data_io2D = Data_IO(io_interface2D, input_path="", output_path="",
                              batch_path=tmp_batches, delete_batchDir=False)
        # Create 3D imgaging and segmentation data set
        self.dataset3D = dict()
        for i in range(0, 10):
            img = np.random.rand(16, 16, 16) * 255
            self.img = img.astype(int)
            seg = np.random.rand(16, 16, 16) * 3
            self.seg = seg.astype(int)
            if i in range(8,10): sample = (self.img, None)
            else : sample = (self.img, self.seg)
            self.dataset3D["TEST.sample_" + str(i)] = sample
        # Initialize Dictionary IO Interface
        io_interface3D = Dictionary_interface(self.dataset3D, classes=3,
                                              three_dim=True)
        # Initialize temporary directory
        self.tmp_dir3D = tempfile.TemporaryDirectory(prefix="tmp.miscnn.")
        tmp_batches = os.path.join(self.tmp_dir3D.name, "batches")
        # Initialize Data IO
        self.data_io3D = Data_IO(io_interface3D, input_path="", output_path="",
                              batch_path=tmp_batches, delete_batchDir=False)


    # Delete all temporary files
    @classmethod
    def tearDownClass(self):
        self.tmp_dir2D.cleanup()
        self.tmp_dir3D.cleanup()

    #-------------------------------------------------#
    #                Base Functionality               #
    #-------------------------------------------------#
# create

# different parameter like architecture, loss, metric

    #-------------------------------------------------#
    #                     Training                    #
    #-------------------------------------------------#

    #-------------------------------------------------#
    #                    Prediction                   #
    #-------------------------------------------------#

    #-------------------------------------------------#
    #                    Validation                   #
    #-------------------------------------------------#

    #-------------------------------------------------#
    #                  Model Storage and Loading                 #
    #-------------------------------------------------#

# # Initialize Data IO Interface for NIfTI data
# interface = NIFTI_interface(channels=1, classes=3)
#
# # Create Data IO object to load and write samples in the file structure
# data_io = Data_IO(interface, path_data, delete_batchDir=True)
#
# # Access all available samples in our file structure
# sample_list = data_io.get_indiceslist()
# sample_list.sort()
#
# # Print out the sample list
# print("Sample list:", sample_list)
#
# # Now let's load each sample and obtain collect diverse information from them
# sample_data = {}
# for index in tqdm(sample_list):
#     # Sample loading
#     sample = data_io.sample_loader(index, load_seg=True)
#     # Create an empty list for the current asmple in our data dictionary
#     sample_data[index] = []
#     # Store the volume shape
#     sample_data[index].append(sample.img_data.shape)
#     # Identify minimum and maximum volume intensity
#     sample_data[index].append(sample.img_data.min())
#     sample_data[index].append(sample.img_data.max())
#     # Store voxel spacing
#     sample_data[index].append(sample.details["spacing"])
#     # Identify and store class distribution
#     unique_data, unique_counts = np.unique(sample.seg_data, return_counts=True)
#     class_freq = unique_counts / np.sum(unique_counts)
#     class_freq = np.around(class_freq, decimals=6)
#     sample_data[index].append(tuple(class_freq))
