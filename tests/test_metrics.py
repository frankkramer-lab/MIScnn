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
from miscnn.neural_network.metrics import *

#-----------------------------------------------------#
#                  Unittest: Metrics                  #
#-----------------------------------------------------#
class metricTEST(unittest.TestCase):
    # Create random imaging and segmentation data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Create 2D imgaging and segmentation data set
        self.dataset = dict()
        for i in range(0, 2):
            img = np.random.rand(16, 16) * 255
            self.img = img.astype(int)
            seg = np.random.rand(16, 16) * 2
            self.seg = seg.astype(int)
            self.dataset["TEST.sample_" + str(i)] = (self.img, self.seg)
        # Initialize Dictionary IO Interface
        io_interface = Dictionary_interface(self.dataset, classes=3,
                                              three_dim=False)
        # Initialize temporary directory
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="tmp.miscnn.")
        tmp_batches = os.path.join(self.tmp_dir.name, "batches")
        # Initialize Data IO
        self.data_io = Data_IO(io_interface,
                               input_path=os.path.join(self.tmp_dir.name),
                               output_path=os.path.join(self.tmp_dir.name),
                               batch_path=tmp_batches, delete_batchDir=False)
        # Initialize Preprocessor
        self.pp = Preprocessor(self.data_io, batch_size=1,
                               data_aug=None, analysis="fullimage")
        # Initialize Neural Network
        self.model = Neural_Network(self.pp)
        # Get sample list
        self.sample_list = self.data_io.get_indiceslist()

    # Delete all temporary files
    @classmethod
    def tearDownClass(self):
        self.tmp_dir.cleanup()

    #-------------------------------------------------#
    #               Standard DSC Metric               #
    #-------------------------------------------------#
    def test_METRICS_DSC_standard(self):
        self.model.loss = dice_coefficient
        self.model.metrics = [dice_coefficient]
        self.model.train(self.sample_list, epochs=1)

    #-------------------------------------------------#
    #                Standard DSC Loss               #
    #-------------------------------------------------#
    def test_METRICS_DSC_standardLOSS(self):
        self.model.loss = dice_coefficient_loss
        self.model.metrics = [dice_coefficient_loss]
        self.model.train(self.sample_list, epochs=1)

    #-------------------------------------------------#
    #                 Soft DSC Metric                 #
    #-------------------------------------------------#
    def test_METRICS_DSC_soft(self):
        self.model.loss = dice_soft
        self.model.metrics = [dice_soft]
        self.model.train(self.sample_list, epochs=1)

    #-------------------------------------------------#
    #                  Soft DSC Loss                  #
    #-------------------------------------------------#
    def test_METRICS_DSC_softLOSS(self):
        self.model.loss = dice_soft_loss
        self.model.metrics = [dice_soft_loss]
        self.model.train(self.sample_list, epochs=1)

    #-------------------------------------------------#
    #                   Weighted DSC                  #
    #-------------------------------------------------#
    def test_METRICS_DSC_weighted(self):
        self.model.loss = dice_weighted([1,1,4])
        self.model.metrics = [dice_weighted([1,1,4])]
        self.model.train(self.sample_list, epochs=1)

    #-------------------------------------------------#
    #             Dice & Crossentropy loss            #
    #-------------------------------------------------#
    def test_METRICS_DSC_CrossEntropy(self):
        self.model.loss = dice_crossentropy
        self.model.metrics = [dice_crossentropy]
        self.model.train(self.sample_list, epochs=1)

    #-------------------------------------------------#
    #                   Tversky loss                  #
    #-------------------------------------------------#
    def test_METRICS_Tversky(self):
        self.model.loss = tversky_loss
        self.model.metrics = [tversky_loss]
        self.model.train(self.sample_list, epochs=1)

    #-------------------------------------------------#
    #           Tversky & Crossentropy loss           #
    #-------------------------------------------------#
    def test_METRICS_Tversky_CrossEntropy(self):
        self.model.loss = tversky_crossentropy
        self.model.metrics = [tversky_crossentropy]
        self.model.train(self.sample_list, epochs=1)
