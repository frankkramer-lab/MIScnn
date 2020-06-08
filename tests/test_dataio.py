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

#-----------------------------------------------------#
#                  Unittest: Data IO                  #
#-----------------------------------------------------#
class Data_IO(unittest.TestCase):
    # Create random imaging and segmentation data
    @classmethod
    def setUpClass(self):
        # Create imgaging and segmentation data set
        np.random.seed(1234)
        dataset = dict()
        for i in range(0, 10):
            img = np.random.rand(16, 16, 16) * 255
            self.img = img.astype(int)
            seg = np.random.rand(16, 16, 16) * 3
            self.seg = seg.astype(int)
            dataset["TEST.sample_" + str(i)] = (img, seg)
        # Initialize Dictionary IO Interface
        self.io_interface = Dictionary_interface(dataset)
        # Initialize temporary directory
        self.tmp_batches = tempfile.TemporaryDirectory(prefix="tmp.miscnn.",
                                                       suffix=".batches")

    # Delete all temporary files
    @classmethod
    def tearDownClass(self):
        pass

    #-------------------------------------------------#
    #                Base Functionality               #
    #-------------------------------------------------#
    # Class Creation
    def test_BASE_create(self):
        interface = Image_interface()

    # #-------------------------------------------------#
    # #                 Batch Management                #
    # #-------------------------------------------------#
    # # Class Creation
    # def test_BASE_create(self):
    #     interface = Image_interface()
    #
    # #-------------------------------------------------#
    # #                Evaluation Backup                #
    # #-------------------------------------------------#
    # # Class Creation
    # def test_EVALUATION_create(self):
    #     interface = Image_interface()

#-----------------------------------------------------#
#               Unittest: Main Function               #
#-----------------------------------------------------#
if __name__ == '__main__':
    unittest.main()
