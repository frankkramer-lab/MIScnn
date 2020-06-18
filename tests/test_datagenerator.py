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
from miscnn import Data_IO, Preprocessor, Data_Augmentation
from miscnn.data_loading.interfaces import Dictionary_interface
from miscnn.neural_network.data_generator import DataGenerator

#-----------------------------------------------------#
#                Unittest: Preprocessor               #
#-----------------------------------------------------#
class DataGeneratorTEST(unittest.TestCase):
    # Create random imaging and segmentation data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Create imgaging and segmentation data set
        self.dataset = dict()
        for i in range(0, 10):
            img = np.random.rand(16, 16, 16) * 255
            self.img = img.astype(int)
            seg = np.random.rand(16, 16, 16) * 3
            self.seg = seg.astype(int)
            sample = (self.img, self.seg)
            self.dataset["TEST.sample_" + str(i)] = sample
        # Initialize Dictionary IO Interface
        io_interface = Dictionary_interface(self.dataset, classes=3,
                                            three_dim=True)
        # Initialize temporary directory
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="tmp.miscnn.")
        tmp_batches = os.path.join(self.tmp_dir.name, "batches")
        # Initialize Data IO
        self.data_io = Data_IO(io_interface, input_path="", output_path="",
                               batch_path=tmp_batches, delete_batchDir=False)
        # Initialize Data Augmentation
        self.data_aug = Data_Augmentation()
        # Get sample list
        self.sample_list = self.data_io.get_indiceslist()

    # Delete all temporary files
    @classmethod
    def tearDownClass(self):
        self.tmp_dir.cleanup()

    #-------------------------------------------------#
    #                Base Functionality               #
    #-------------------------------------------------#
    # Class Creation
    def test_DATAGENERATOR_create(self):
        pp_fi = Preprocessor(self.data_io, batch_size=4, data_aug=self.data_aug,
                             prepare_subfunctions=False, prepare_batches=False,
                             analysis="fullimage")
        data_gen = DataGenerator(self.sample_list, pp_fi, training=False,
                                 validation=False, shuffle=False,
                                 iterations=None)
        self.assertIsInstance(data_gen, DataGenerator)

    # Run data generation for training
    def test_DATAGENERATOR_runTraining(self):
        pp_fi = Preprocessor(self.data_io, batch_size=4, data_aug=self.data_aug,
                             prepare_subfunctions=False, prepare_batches=False,
                             analysis="fullimage")
        data_gen = DataGenerator(self.sample_list, pp_fi, training=True,
                                 shuffle=False, iterations=None)
        self.assertEqual(len(data_gen), 3)
        for batch in data_gen:
            self.assertIsInstance(batch, tuple)
            self.assertEqual(batch[0].shape, (4,16,16,16,1))
            self.assertEqual(batch[1].shape, (4,16,16,16,3))
        pp_pc = Preprocessor(self.data_io, batch_size=3, data_aug=self.data_aug,
                             prepare_subfunctions=False, prepare_batches=False,
                             patch_shape=(5,5,5), analysis="patchwise-crop")
        data_gen = DataGenerator(self.sample_list, pp_pc, training=True,
                                 shuffle=False, iterations=None)
        self.assertEqual(len(data_gen), 4)
        for batch in data_gen:
            self.assertIsInstance(batch, tuple)
            self.assertEqual(batch[0].shape, (3,5,5,5,1))
            self.assertEqual(batch[1].shape, (3,5,5,5,3))

    # Run data generation for prediction
    def test_DATAGENERATOR_runPrediction(self):
        pp_fi = Preprocessor(self.data_io, batch_size=4, data_aug=self.data_aug,
                             prepare_subfunctions=False, prepare_batches=False,
                             analysis="fullimage")
        data_gen = DataGenerator(self.sample_list, pp_fi, training=False,
                                 shuffle=False, iterations=None)
        self.assertEqual(len(data_gen), 10)
        for batch in data_gen:
            self.assertNotIsInstance(batch, tuple)
            self.assertEqual(batch.shape, (1,16,16,16,1))
        pp_pc = Preprocessor(self.data_io, batch_size=3, data_aug=self.data_aug,
                             prepare_subfunctions=False, prepare_batches=False,
                             patch_shape=(5,5,5), analysis="patchwise-crop")
        data_gen = DataGenerator(self.sample_list, pp_pc, training=False,
                                 shuffle=False, iterations=None)
        self.assertEqual(len(data_gen), 220)
        for batch in data_gen:
            self.assertNotIsInstance(batch, tuple)
            self.assertIn(batch.shape, [(3,5,5,5,1), (1,5,5,5,1)])

    # Check if full images without data augmentation are consistent
    def test_DATAGENERATOR_consistency(self):
        pp_fi = Preprocessor(self.data_io, batch_size=1, data_aug=None,
                             prepare_subfunctions=False, prepare_batches=False,
                             analysis="fullimage")
        data_gen = DataGenerator(self.sample_list, pp_fi,
                                 training=True, shuffle=False, iterations=None)
        i = 0
        for batch in data_gen:
            sample = self.data_io.sample_loader(self.sample_list[i],
                                                load_seg=True)
            self.assertTrue(np.array_equal(batch[0][0], sample.img_data))
            seg = to_categorical(sample.seg_data, num_classes=3)
            self.assertTrue(np.array_equal(batch[1][0], seg))
            i += 1

    # Iteration fixation test
    def test_DATAGENERATOR_iterations(self):
        pp_fi = Preprocessor(self.data_io, batch_size=1, data_aug=None,
                             prepare_subfunctions=False, prepare_batches=False,
                             analysis="fullimage")
        data_gen = DataGenerator(self.sample_list, pp_fi,
                                 training=True, shuffle=False, iterations=None)
        self.assertEqual(10, len(data_gen))
        data_gen = DataGenerator(self.sample_list, pp_fi,
                                 training=True, shuffle=False, iterations=5)
        self.assertEqual(5, len(data_gen))
        data_gen = DataGenerator(self.sample_list, pp_fi,
                                 training=True, shuffle=False, iterations=50)
        self.assertEqual(50, len(data_gen))
        data_gen = DataGenerator(self.sample_list, pp_fi,
                                 training=True, shuffle=False, iterations=100)
        self.assertEqual(100, len(data_gen))

    # Iteration fixation test
    def test_DATAGENERATOR_augcyling(self):
        data_aug = Data_Augmentation(cycles=20)
        pp_fi = Preprocessor(self.data_io, batch_size=4, data_aug=data_aug,
                             prepare_subfunctions=False, prepare_batches=False,
                             analysis="fullimage")
        data_gen = DataGenerator(self.sample_list, pp_fi,
                                 training=True, shuffle=False, iterations=None)
        self.assertEqual(50, len(data_gen))

    # Check if shuffling is functional
    def test_DATAGENERATOR_shuffle(self):
        pp_fi = Preprocessor(self.data_io, batch_size=1, data_aug=None,
                             prepare_subfunctions=False, prepare_batches=False,
                             analysis="fullimage")
        data_gen = DataGenerator(self.sample_list, pp_fi,
                                 training=True, shuffle=False, iterations=None)
        list_ordered = []
        for batch in data_gen : list_ordered.append(batch)
        for batch in data_gen : list_ordered.append(batch)
        data_gen = DataGenerator(self.sample_list, pp_fi,
                                 training=True, shuffle=True, iterations=None)
        list_shuffled = []
        for batch in data_gen : list_shuffled.append(batch)
        data_gen.on_epoch_end()
        for batch in data_gen : list_shuffled.append(batch)
        size = len(data_gen)
        o_counter = 0
        s_counter = 0
        for i in range(0, size):
            oa_img = list_ordered[i][0]
            oa_seg = list_ordered[i][1]
            ob_img = list_ordered[i+size][0]
            ob_seg = list_ordered[i+size][1]
            sa_img = list_shuffled[i][0]
            sa_seg = list_shuffled[i][1]
            sb_img = list_shuffled[i+size][0]
            sb_seg = list_shuffled[i+size][1]
            if np.array_equal(oa_img, ob_img) and \
                np.array_equal(oa_seg, ob_seg):
                o_counter += 1
            if not np.array_equal(sa_img, sb_img) and \
                not np.array_equal(sa_seg, sb_seg):
                s_counter += 1
        o_ratio = o_counter / size
        self.assertTrue(o_ratio == 1.0)
        s_ratio = s_counter / size
        self.assertTrue(1.0 >= s_ratio and s_ratio >= 0.5)

    # Run data generation with preparation of subfunctions and batches
    def test_DATAGENERATOR_prepareData(self):
        pp_fi = Preprocessor(self.data_io, batch_size=4, data_aug=None,
                             prepare_subfunctions=True, prepare_batches=True,
                             analysis="fullimage")
        data_gen = DataGenerator(self.sample_list, pp_fi, training=True,
                                 shuffle=True, iterations=None)
        self.assertEqual(len(data_gen), 3)
        for batch in data_gen:
            self.assertIsInstance(batch, tuple)
            self.assertEqual(batch[0].shape[1:], (16,16,16,1))
            self.assertEqual(batch[1].shape[1:], (16,16,16,3))
            self.assertIn(batch[0].shape[0], [2,4])
