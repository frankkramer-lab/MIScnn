#==============================================================================#
#  Author:       Philip Meyer                                                  #
#  Copyright:    2022 IT-Infrastructure for Translational Medical Research,    #
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
from miscnn.utils.visualizer import *
from miscnn import Data_IO
from miscnn.data_loading.interfaces import Dictionary_interface

#-----------------------------------------------------#
#              Unittest: Patch Operations             #
#-----------------------------------------------------#
class VisualizerTEST(unittest.TestCase):
    # Create random imaging and segmentation data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Create 2D imgaging and segmentation data set
        self.dataset2D = dict()
        for i in range(0, 10):
            img = np.random.rand(16, 16) * 255
            img = img.astype(int)
            seg = np.random.rand(16, 16) * 3
            seg = seg.astype(int)
            self.dataset2D["TEST.sample_" + str(i)] = (img, seg)
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
            img = img.astype(int)
            seg = np.random.rand(16, 16, 16) * 3
            seg = seg.astype(int)
            sample = (img, seg)
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

    def test_VISUALIZER_dimensionality(self):
        sample_list = self.data_io2D.get_indiceslist()
        for s in sample_list:
            sample = self.data_io2D.sample_loader(s, load_seg=False)
            res = detect_dimensionality(sample.img_data)
            self.assertEqual(res, 2)
        sample_list = self.data_io3D.get_indiceslist()
        for s in sample_list:
            sample = self.data_io3D.sample_loader(s, load_seg=False)
            res = detect_dimensionality(sample.img_data)
            self.assertEqual(res, 3)
    #-------------------------------------------------#
    #                    Luminosity                   #
    #-------------------------------------------------#
    def test_VISUALIZER_normalize(self):
        sample_list = self.data_io2D.get_indiceslist()
        for s in sample_list:
            sample = self.data_io2D.sample_loader(s, load_seg=True)
            res = normalize(sample.img_data, to_greyscale = True, normalize = True)
            self.assertEqual(res, (16,16))
            res = normalize(sample.seg_data, to_greyscale = True, normalize = True)
            self.assertEqual(res, (16,16))
        sample_list = self.data_io3D.get_indiceslist()
        for s in sample_list:
            sample = self.data_io3D.sample_loader(s, load_seg=True)
            res = normalize(sample.img_data, to_greyscale = True, normalize = True)
            self.assertEqual(res, (16, 16, 16))
            res = normalize(sample.seg_data, to_greyscale = True, normalize = True)
            self.assertEqual(res, (16, 16, 16))