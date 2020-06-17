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
#Internal libraries
from miscnn import Data_IO
from miscnn.data_loading.interfaces import Dictionary_interface
from miscnn.utils.patch_operations import *

#-----------------------------------------------------#
#              Unittest: Patch Operations             #
#-----------------------------------------------------#
class PatchOperationsTEST(unittest.TestCase):
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
            self.dataset3D["TEST.sample_" + str(i)] = (self.img, self.seg)
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
    #                   Slice Matrix                  #
    #-------------------------------------------------#
    def test_PATCHOPERATIONS_slicing(self):
        sample_list = self.data_io2D.get_indiceslist()
        for index in sample_list:
            sample = self.data_io2D.sample_loader(index)
            patches = slice_matrix(sample.img_data, window=(5,5),
                                   overlap=(2,2), three_dim=False)
            self.assertEqual(len(patches), 25)
            self.assertEqual(patches[0].shape, (5,5,1))
        sample_list = self.data_io3D.get_indiceslist()
        for index in sample_list:
            sample = self.data_io3D.sample_loader(index)
            patches = slice_matrix(sample.img_data, window=(5,5,5),
                                   overlap=(2,2,2), three_dim=True)
            self.assertEqual(len(patches), 125)
            self.assertEqual(patches[0].shape, (5,5,5,1))

    #-------------------------------------------------#
    #               Concatenate Matrices              #
    #-------------------------------------------------#
    def test_PATCHOPERATIONS_concatenate(self):
        sample_list = self.data_io2D.get_indiceslist()
        for index in sample_list:
            sample = self.data_io2D.sample_loader(index)
            patches = slice_matrix(sample.img_data, window=(5,5),
                                   overlap=(2,2), three_dim=False)
            concat = concat_matrices(patches=patches,
                                     image_size=(16,16),
                                     window=(5,5),
                                     overlap=(2,2),
                                     three_dim=False)
            self.assertEqual(concat.shape, (16,16,1))
        sample_list = self.data_io3D.get_indiceslist()
        for index in sample_list:
            sample = self.data_io3D.sample_loader(index)
            patches = slice_matrix(sample.img_data, window=(5,5,5),
                                   overlap=(2,2,2), three_dim=True)
            concat = concat_matrices(patches=patches,
                                     image_size=(16,16,16),
                                     window=(5,5,5),
                                     overlap=(2,2,2),
                                     three_dim=True)
            self.assertEqual(concat.shape, (16,16,16,1))

    #-------------------------------------------------#
    #                  Patch Padding                  #
    #-------------------------------------------------#
    def test_PATCHOPERATIONS_padding(self):
        sample_list = self.data_io2D.get_indiceslist()
        for index in sample_list:
            sample = self.data_io2D.sample_loader(index)
            img_padded = pad_patch(np.expand_dims(sample.img_data, axis=0),
                                   patch_shape=(8,20),
                                   return_slicer=False)
            self.assertEqual(img_padded.shape, (1,16,20,1))
        sample_list = self.data_io3D.get_indiceslist()
        for index in sample_list:
            sample = self.data_io3D.sample_loader(index)
            img_padded = pad_patch(np.expand_dims(sample.img_data, axis=0),
                                   patch_shape=(8,16,32),
                                   return_slicer=False)
            self.assertEqual(img_padded.shape, (1,16,16,32,1))

    #-------------------------------------------------#
    #                  Patch Cropping                 #
    #-------------------------------------------------#
    def test_PATCHOPERATIONS_cropping(self):
        sample_list = self.data_io2D.get_indiceslist()
        for index in sample_list:
            sample = self.data_io2D.sample_loader(index)
            img_padded, slicer = pad_patch(
                                    np.expand_dims(sample.img_data, axis=0),
                                    patch_shape=(8,20),
                                    return_slicer=True)
            img_processed = crop_patch(img_padded, slicer)
            self.assertEqual(img_processed.shape, (1,16,16,1))
        sample_list = self.data_io3D.get_indiceslist()
        for index in sample_list:
            sample = self.data_io3D.sample_loader(index)
            img_padded, slicer = pad_patch(
                                    np.expand_dims(sample.img_data, axis=0),
                                    patch_shape=(8,16,32),
                                    return_slicer=True)
            img_processed = crop_patch(img_padded, slicer)
            self.assertEqual(img_processed.shape, (1,16,16,16,1))
