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
#                Unittest: Preprocessor               #
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
    # Class Creation
    def test_PREPROCESSOR_BASE_create(self):
        with self.assertRaises(Exception):
            Preprocessor()
        Preprocessor(self.data_io3D, batch_size=1, analysis="fullimage")
        Preprocessor(self.data_io3D, batch_size=1, analysis="patchwise-crop",
                     patch_shape=(16,16,16))
        Preprocessor(self.data_io3D, batch_size=1, analysis="patchwise-grid",
                     patch_shape=(16,16,16), data_aug=None)

    # Simple Prepreossor run
    def test_PREPROCESSOR_BASE_run(self):
        sample_list = self.data_io3D.get_indiceslist()
        pp = Preprocessor(self.data_io3D, data_aug=None, batch_size=1,
                          analysis="fullimage")
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

    # Prepreossor run with data augmentation
    def test_PREPROCESSOR_BASE_dataaugmentation(self):
        sample_list = self.data_io3D.get_indiceslist()
        pp = Preprocessor(self.data_io3D, batch_size=1,  analysis="fullimage")
        batches = pp.run(sample_list[8:10], training=False, validation=False)
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0][0].shape, (1,16,16,16,1))
        self.assertIsNone(batches[0][1])
        sample = self.data_io3D.sample_loader(sample_list[8], load_seg=False)
        self.assertFalse(np.array_equal(batches[0][0], sample.img_data))

    # Different batchsizes run
    def test_PREPROCESSOR_BASE_batchsizes(self):
        sample_list = self.data_io3D.get_indiceslist()
        pp = Preprocessor(self.data_io3D, batch_size=1, analysis="fullimage")
        batches = pp.run(sample_list[0:8], training=True, validation=False)
        self.assertEqual(len(batches), 8)
        self.assertEqual(batches[0][0].shape, (1,16,16,16,1))
        pp = Preprocessor(self.data_io3D, batch_size=2, analysis="fullimage")
        batches = pp.run(sample_list[0:8], training=True, validation=False)
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[0][0].shape, (2,16,16,16,1))
        pp = Preprocessor(self.data_io3D, batch_size=3, analysis="fullimage")
        batches = pp.run(sample_list[0:8], training=True, validation=False)
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0][0].shape, (3,16,16,16,1))
        self.assertEqual(batches[-1][0].shape, (2,16,16,16,1))
        pp = Preprocessor(self.data_io3D, batch_size=8, analysis="fullimage")
        batches = pp.run(sample_list[0:8], training=True, validation=False)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0].shape, (8,16,16,16,1))
        pp = Preprocessor(self.data_io3D, batch_size=100, analysis="fullimage")
        batches = pp.run(sample_list[0:8], training=True, validation=False)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0].shape, (8,16,16,16,1))

    # Batch preparation check
    def test_PREPROCESSOR_BASE_prepareBatches(self):
        sample_list = self.data_io3D.get_indiceslist()
        pp = Preprocessor(self.data_io3D, batch_size=1, analysis="fullimage",
                          prepare_batches=True)
        batch_pointer = pp.run(sample_list[0:8], training=True, validation=False)
        self.assertEqual(batch_pointer, 7)
        tmp_batches = os.path.join(self.tmp_dir3D.name, "batches")
        batch_list = []
        for batch_file in os.listdir(tmp_batches):
            if batch_file.startswith(str(pp.data_io.seed)):
                batch_list.append(batch_file)
        self.assertEqual(len(batch_list), 16)

    #-------------------------------------------------#
    #                  Postprocessing                 #
    #-------------------------------------------------#
    def test_PREPROCESSOR_postprocessing_(self):
        sample_list = self.data_io3D.get_indiceslist()
        pp = Preprocessor(self.data_io3D, batch_size=1, analysis="fullimage",
                          data_aug=None)
        batches = pp.run(sample_list[0:3], training=True, validation=False)
        for i in range(0, 3):
            pred_postprec = pp.postprocessing(sample_list[i], batches[i][1])
            self.assertEqual(pred_postprec.shape, (16,16,16))
            sam = self.data_io3D.sample_loader(sample_list[i], load_seg=True)
            self.assertTrue(np.array_equal(pred_postprec,
                            np.reshape(sam.seg_data, (16,16,16))))

    #-------------------------------------------------#
    #            Analysis: Patchwise-crop             #
    #-------------------------------------------------#
    def test_PREPROCESSOR_patchwisecrop_2D(self):
        sample_list = self.data_io2D.get_indiceslist()
        pp = Preprocessor(self.data_io2D, data_aug=None, batch_size=1,
                          analysis="patchwise-crop", patch_shape=(4,4))
        batches = pp.run(sample_list[0:3], training=True, validation=False)
        self.assertEqual(len(batches), 3)
        batches = pp.run(sample_list[0:1], training=False, validation=False)
        self.assertEqual(len(batches), 16)
        sample = self.data_io2D.sample_loader(sample_list[0], load_seg=True)
        sample.seg_data = to_categorical(sample.seg_data,
                                         num_classes=sample.classes)
        ready_data = pp.analysis_patchwise_crop(sample, data_aug=False)
        self.assertEqual(len(ready_data), 1)
        self.assertEqual(ready_data[0][0].shape, (4,4,1))
        self.assertEqual(ready_data[0][1].shape, (4,4,3))

    def test_PREPROCESSOR_patchwisecrop_3D(self):
        sample_list = self.data_io3D.get_indiceslist()
        pp = Preprocessor(self.data_io3D, data_aug=None, batch_size=1,
                          analysis="patchwise-crop", patch_shape=(4,4,4))
        batches = pp.run(sample_list[0:3], training=True, validation=False)
        self.assertEqual(len(batches), 3)
        batches = pp.run(sample_list[0:1], training=False, validation=False)
        self.assertEqual(len(batches), 64)
        sample = self.data_io3D.sample_loader(sample_list[0], load_seg=True)
        sample.seg_data = to_categorical(sample.seg_data,
                                         num_classes=sample.classes)
        ready_data = pp.analysis_patchwise_crop(sample, data_aug=False)
        self.assertEqual(len(ready_data), 1)
        self.assertEqual(ready_data[0][0].shape, (4,4,4,1))
        self.assertEqual(ready_data[0][1].shape, (4,4,4,3))

    def test_PREPROCESSOR_patchwisecrop_skipBlanks(self):
        sample_list = self.data_io3D.get_indiceslist()
        pp = Preprocessor(self.data_io3D, data_aug=None, batch_size=1,
                          analysis="patchwise-crop", patch_shape=(4,4,4))
        pp.patchwise_skip_blanks = True
        batches = pp.run(sample_list[0:3], training=True, validation=False)
        sample = self.data_io3D.sample_loader(sample_list[0], load_seg=True)
        sample.seg_data = to_categorical(sample.seg_data,
                                         num_classes=sample.classes)
        ready_data = pp.analysis_patchwise_crop(sample, data_aug=False)
        self.assertEqual(len(ready_data), 1)
        self.assertEqual(ready_data[0][0].shape, (4,4,4,1))
        self.assertEqual(ready_data[0][1].shape, (4,4,4,3))

    #-------------------------------------------------#
    #            Analysis: Patchwise-grid             #
    #-------------------------------------------------#
    def test_PREPROCESSOR_patchwisegrid_2D(self):
        sample_list = self.data_io2D.get_indiceslist()
        pp = Preprocessor(self.data_io2D, data_aug=None, batch_size=1,
                          analysis="patchwise-grid", patch_shape=(4,4))
        batches = pp.run(sample_list[0:1], training=False, validation=False)
        self.assertEqual(len(batches), 16)
        sample = self.data_io2D.sample_loader(sample_list[0], load_seg=True)
        sample.seg_data = to_categorical(sample.seg_data,
                                         num_classes=sample.classes)
        pp = Preprocessor(self.data_io2D, data_aug=None, batch_size=1,
                          analysis="patchwise-grid", patch_shape=(5,5))
        ready_data = pp.analysis_patchwise_grid(sample, data_aug=False,
                                                training=True)
        self.assertEqual(len(ready_data), 16)
        self.assertEqual(ready_data[0][0].shape, (5,5,1))
        self.assertEqual(ready_data[0][1].shape, (5,5,3))

    def test_PREPROCESSOR_patchwisegrid_3D(self):
        sample_list = self.data_io3D.get_indiceslist()
        pp = Preprocessor(self.data_io3D, data_aug=None, batch_size=1,
                          analysis="patchwise-grid", patch_shape=(4,4,4))
        batches = pp.run(sample_list[0:1], training=False, validation=False)
        self.assertEqual(len(batches), 64)
        sample = self.data_io3D.sample_loader(sample_list[0], load_seg=True)
        sample.seg_data = to_categorical(sample.seg_data,
                                         num_classes=sample.classes)
        pp = Preprocessor(self.data_io3D, data_aug=None, batch_size=1,
                          analysis="patchwise-grid", patch_shape=(5,5,5))
        ready_data = pp.analysis_patchwise_grid(sample, data_aug=False,
                                                training=True)
        self.assertEqual(len(ready_data), 64)
        self.assertEqual(ready_data[0][0].shape, (5,5,5,1))
        self.assertEqual(ready_data[0][1].shape, (5,5,5,3))

    def test_PREPROCESSOR_patchwisegrid_skipBlanks(self):
        sample_list = self.data_io3D.get_indiceslist()
        pp = Preprocessor(self.data_io3D, data_aug=None, batch_size=1,
                          analysis="patchwise-grid", patch_shape=(4,4,4))
        pp.patchwise_skip_blanks = True
        batches = pp.run(sample_list[0:3], training=True, validation=False)
        sample = self.data_io3D.sample_loader(sample_list[0], load_seg=True)
        sample.seg_data = to_categorical(sample.seg_data,
                                         num_classes=sample.classes)
        ready_data = pp.analysis_patchwise_grid(sample, data_aug=False,
                                                training=True)
        self.assertEqual(len(ready_data), 64)
        self.assertEqual(ready_data[0][0].shape, (4,4,4,1))
        self.assertEqual(ready_data[0][1].shape, (4,4,4,3))

    #-------------------------------------------------#
    #               Analysis: Fullimage               #
    #-------------------------------------------------#
    def test_PREPROCESSOR_fullimage_2D(self):
        sample_list = self.data_io2D.get_indiceslist()
        pp = Preprocessor(self.data_io2D, data_aug=None, batch_size=2,
                          analysis="fullimage")
        batches = pp.run(sample_list[0:3], training=True, validation=False)
        self.assertEqual(len(batches), 2)
        batches = pp.run(sample_list[0:1], training=False, validation=False)
        self.assertEqual(len(batches), 1)
        sample = self.data_io2D.sample_loader(sample_list[0], load_seg=True)
        sample.seg_data = to_categorical(sample.seg_data,
                                         num_classes=sample.classes)
        ready_data = pp.analysis_fullimage(sample, data_aug=False,
                                           training=True)
        self.assertEqual(len(ready_data), 1)
        self.assertEqual(ready_data[0][0].shape, (16,16,1))
        self.assertEqual(ready_data[0][1].shape, (16,16,3))

    def test_PREPROCESSOR_fullimage_3D(self):
        sample_list = self.data_io3D.get_indiceslist()
        pp = Preprocessor(self.data_io3D, data_aug=None, batch_size=2,
                          analysis="fullimage")
        batches = pp.run(sample_list[0:3], training=True, validation=False)
        self.assertEqual(len(batches), 2)
        batches = pp.run(sample_list[0:1], training=False, validation=False)
        self.assertEqual(len(batches), 1)
        sample = self.data_io3D.sample_loader(sample_list[0], load_seg=True)
        sample.seg_data = to_categorical(sample.seg_data,
                                         num_classes=sample.classes)
        ready_data = pp.analysis_fullimage(sample, data_aug=False,
                                           training=True)
        self.assertEqual(len(ready_data), 1)
        self.assertEqual(ready_data[0][0].shape, (16,16,16,1))
        self.assertEqual(ready_data[0][1].shape, (16,16,16,3))
