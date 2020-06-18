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
from miscnn import Data_IO, Preprocessor, Neural_Network
from miscnn.data_loading.interfaces import Dictionary_interface
from miscnn.neural_network.architecture.unet import *

#-----------------------------------------------------#
#               Unittest: Architectures               #
#-----------------------------------------------------#
class architectureTEST(unittest.TestCase):
    # Create random imaging and segmentation data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Create 2D imgaging and segmentation data set
        self.dataset2D = dict()
        for i in range(0, 1):
            img = np.random.rand(32, 32) * 255
            self.img = img.astype(int)
            seg = np.random.rand(32, 32) * 2
            self.seg = seg.astype(int)
            self.dataset2D["TEST.sample_" + str(i)] = (self.img, self.seg)
        # Initialize Dictionary IO Interface
        io_interface2D = Dictionary_interface(self.dataset2D, classes=3,
                                              three_dim=False)
        # Initialize temporary directory
        self.tmp_dir2D = tempfile.TemporaryDirectory(prefix="tmp.miscnn.")
        tmp_batches = os.path.join(self.tmp_dir2D.name, "batches")
        # Initialize Data IO
        self.data_io2D = Data_IO(io_interface2D,
                                 input_path=os.path.join(self.tmp_dir2D.name),
                                 output_path=os.path.join(self.tmp_dir2D.name),
                                 batch_path=tmp_batches, delete_batchDir=False)
        # Initialize Preprocessor
        self.pp2D = Preprocessor(self.data_io2D, batch_size=1,
                                 data_aug=None, analysis="fullimage")
        # Get sample list
        self.sample_list2D = self.data_io2D.get_indiceslist()
        # Create 3D imgaging and segmentation data set
        self.dataset3D = dict()
        for i in range(0, 1):
            img = np.random.rand(32, 32, 32) * 255
            self.img = img.astype(int)
            seg = np.random.rand(32, 32, 32) * 3
            self.seg = seg.astype(int)
            self.dataset3D["TEST.sample_" + str(i)] = (self.img, self.seg)
        # Initialize Dictionary IO Interface
        io_interface3D = Dictionary_interface(self.dataset3D, classes=3,
                                              three_dim=True)
        # Initialize temporary directory
        self.tmp_dir3D = tempfile.TemporaryDirectory(prefix="tmp.miscnn.")
        tmp_batches = os.path.join(self.tmp_dir3D.name, "batches")
        # Initialize Data IO
        self.data_io3D = Data_IO(io_interface3D,
                                 input_path=os.path.join(self.tmp_dir3D.name),
                                 output_path=os.path.join(self.tmp_dir3D.name),
                                 batch_path=tmp_batches, delete_batchDir=False)
        # Initialize Preprocessor
        self.pp3D = Preprocessor(self.data_io3D, batch_size=1,
                                 data_aug=None, analysis="fullimage")
        # Get sample list
        self.sample_list3D = self.data_io3D.get_indiceslist()

    # Delete all temporary files
    @classmethod
    def tearDownClass(self):
        self.tmp_dir2D.cleanup()
        self.tmp_dir3D.cleanup()

    #-------------------------------------------------#
    #                  U-Net Standard                 #
    #-------------------------------------------------#
    def test_ARCHITECTURES_UNET_standard(self):
        model2D = Neural_Network(self.pp2D, architecture=UNet_standard())
        model2D.predict(self.sample_list2D)
        model3D = Neural_Network(self.pp3D, architecture=UNet_standard())
        model3D.predict(self.sample_list3D)

    #-------------------------------------------------#
    #                   U-Net Plain                   #
    #-------------------------------------------------#
    def test_ARCHITECTURES_UNET_plain(self):
        model2D = Neural_Network(self.pp2D, architecture=UNet_plain())
        model2D.predict(self.sample_list2D)
        model3D = Neural_Network(self.pp3D, architecture=UNet_plain())
        model3D.predict(self.sample_list3D)

    #-------------------------------------------------#
    #                  U-Net Residual                 #
    #-------------------------------------------------#
    def test_ARCHITECTURES_UNET_residual(self):
        model2D = Neural_Network(self.pp2D, architecture=UNet_residual())
        model2D.predict(self.sample_list2D)
        model3D = Neural_Network(self.pp3D, architecture=UNet_residual())
        model3D.predict(self.sample_list3D)

    #-------------------------------------------------#
    #                  U-Net MultiRes                 #
    #-------------------------------------------------#
    def test_ARCHITECTURES_UNET_multires(self):
        model2D = Neural_Network(self.pp2D, architecture=UNet_multiRes())
        model2D.predict(self.sample_list2D)
        model3D = Neural_Network(self.pp3D, architecture=UNet_multiRes())
        model3D.predict(self.sample_list3D)

    #-------------------------------------------------#
    #                   U-Net Dense                   #
    #-------------------------------------------------#
    def test_ARCHITECTURES_UNET_dense(self):
        model2D = Neural_Network(self.pp2D, architecture=UNet_dense())
        model2D.predict(self.sample_list2D)
        model3D = Neural_Network(self.pp3D, architecture=UNet_dense())
        model3D.predict(self.sample_list3D)

    #-------------------------------------------------#
    #                  U-Net Compact                  #
    #-------------------------------------------------#
    def test_ARCHITECTURES_UNET_compact(self):
        model2D = Neural_Network(self.pp2D, architecture=UNet_compact())
        model2D.predict(self.sample_list2D)
        model3D = Neural_Network(self.pp3D, architecture=UNet_compact())
        model3D.predict(self.sample_list3D)
