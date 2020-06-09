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
from miscnn import Data_IO as DataIO, Preprocessor
from miscnn.data_loading.interfaces import Dictionary_interface

#-----------------------------------------------------#
#                Unittest: Preprocessor               #
#-----------------------------------------------------#
class Data_IO(unittest.TestCase):
    # Create random imaging and segmentation data
    @classmethod
    def setUpClass(self):
        # Create imgaging and segmentation data set
        np.random.seed(1234)
        self.dataset = dict()
        for i in range(0, 10):
            img = np.random.rand(16, 16, 16) * 255
            self.img = img.astype(int)
            seg = np.random.rand(16, 16, 16) * 3
            self.seg = seg.astype(int)
            if i in range(8,10): sample = (self.img, None)
            else : sample = (self.img, self.seg)
            self.dataset["TEST.sample_" + str(i)] = sample
        # Initialize Dictionary IO Interface
        io_interface = Dictionary_interface(self.dataset, classes=3)
        # Initialize temporary directory
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="tmp.miscnn.")
        tmp_batches = os.path.join(self.tmp_dir.name, "batches")
        # Initialize Data IO
        self.data_io = DataIO(io_interface, input_path="", output_path="",
                              batch_path=tmp_batches, delete_batchDir=False)

    # Delete all temporary files
    @classmethod
    def tearDownClass(self):
        self.tmp_dir.cleanup()

    #-------------------------------------------------#
    #                Base Functionality               #
    #-------------------------------------------------#
    # Class Creation
    def test_PREPROCESSOR_BASE_create(self):
        with self.assertRaises(Exception):
            Preprocessor()
        Preprocessor(self.data_io, batch_size=1, analysis="fullimage")
        Preprocessor(self.data_io, batch_size=1, analysis="patchwise-crop",
                     patch_shape=(16,16,16))
        Preprocessor(self.data_io, batch_size=1, analysis="patchwise-grid",
                     patch_shape=(16,16,16), data_aug=None)

    # Simple Prepreossor run
    def test_PREPROCESSOR_BASE_run(self):
        sample_list = self.data_io.get_indiceslist()
        pp = Preprocessor(self.data_io, batch_size=1, analysis="fullimage")
        batches = pp.run(sample_list[8:10], training=False, validation=False)
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0][0].shape, (1,16,16,16,1))
        self.assertIsNone(batches[0][1])
        batches = pp.run(sample_list[0:8], training=True, validation=False)
        self.assertEqual(batches[0][0].shape, (1,16,16,16,1))
        self.assertEqual(batches[0][1].shape, (1,16,16,16,3))
        batches = pp.run(sample_list[0:8], training=True, validation=True)
        self.assertEqual(batches[0][0].shape, (1,16,16,16,1))
        self.assertEqual(batches[0][1].shape, (1,16,16,16,3))

    # Different batchsizes run
    def test_PREPROCESSOR_BASE_batchsizes(self):
        sample_list = self.data_io.get_indiceslist()
        pp = Preprocessor(self.data_io, batch_size=1, analysis="fullimage")
        batches = pp.run(sample_list[0:8], training=True, validation=False)
        self.assertEqual(len(batches), 8)
        self.assertEqual(batches[0][0].shape, (1,16,16,16,1))
        pp = Preprocessor(self.data_io, batch_size=2, analysis="fullimage")
        batches = pp.run(sample_list[0:8], training=True, validation=False)
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[0][0].shape, (2,16,16,16,1))
        pp = Preprocessor(self.data_io, batch_size=3, analysis="fullimage")
        batches = pp.run(sample_list[0:8], training=True, validation=False)
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0][0].shape, (3,16,16,16,1))
        self.assertEqual(batches[-1][0].shape, (2,16,16,16,1))
        pp = Preprocessor(self.data_io, batch_size=8, analysis="fullimage")
        batches = pp.run(sample_list[0:8], training=True, validation=False)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0].shape, (8,16,16,16,1))
        pp = Preprocessor(self.data_io, batch_size=100, analysis="fullimage")
        batches = pp.run(sample_list[0:8], training=True, validation=False)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0].shape, (8,16,16,16,1))

    #-------------------------------------------------#
    #                  Postprocessing                 #
    #-------------------------------------------------#
    def test_PREPROCESSOR_postprocessing(self):
        sample_list = self.data_io.get_indiceslist()
        pp = Preprocessor(self.data_io, batch_size=1, analysis="fullimage")
        batches = pp.run(sample_list[0:8], training=True, validation=False)

    #-------------------------------------------------#
    #            Analysis: Patchwise-crop             #
    #-------------------------------------------------#

    #-------------------------------------------------#
    #            Analysis: Patchwise-grid             #
    #-------------------------------------------------#

    #-------------------------------------------------#
    #               Analysis: Fullimage               #
    #-------------------------------------------------#
