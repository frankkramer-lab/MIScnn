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
from miscnn.evaluation import *
from miscnn.evaluation.cross_validation import split_folds, run_fold

#-----------------------------------------------------#
#                 Unittest: Evaluation                #
#-----------------------------------------------------#
class evaluationTEST(unittest.TestCase):
    # Create random imaging and segmentation data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Create 2D imgaging and segmentation data set
        self.dataset = dict()
        for i in range(0, 6):
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
        self.pp = Preprocessor(self.data_io, batch_size=2,
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
    #                 Cross-Validation                #
    #-------------------------------------------------#
    def test_EVALUATION_crossValidation(self):
        eval_path = os.path.join(self.tmp_dir.name, "evaluation")
        cross_validation(self.sample_list, self.model, k_fold=3, epochs=3,
                         iterations=None,
                         evaluation_path=eval_path,
                         run_detailed_evaluation=False,
                         draw_figures=False,
                         callbacks=[],
                         save_models=False,
                         return_output=False)
        self.assertTrue(os.path.exists(eval_path))
        self.assertTrue(os.path.exists(os.path.join(eval_path, "fold_0")))
        self.assertTrue(os.path.exists(os.path.join(eval_path, "fold_1")))
        self.assertTrue(os.path.exists(os.path.join(eval_path, "fold_2")))

    def test_EVALUATION_crossValidation_splitRun(self):
        eval_path = os.path.join(self.tmp_dir.name, "evaluation")
        split_folds(self.sample_list, k_fold=3, evaluation_path=eval_path)
        self.assertTrue(os.path.exists(eval_path))
        self.assertTrue(os.path.exists(os.path.join(eval_path, "fold_0")))
        self.assertTrue(os.path.exists(os.path.join(eval_path, "fold_1")))
        self.assertTrue(os.path.exists(os.path.join(eval_path, "fold_2")))
        for fold in range(0, 3):
            run_fold(fold, self.model, epochs=1, iterations=None,
                     evaluation_path=eval_path, draw_figures=False,
                     callbacks=[], save_models=True)
            fold_dir =os.path.join(eval_path, "fold_0")
            self.assertTrue(os.path.exists(os.path.join(fold_dir,
                                                        "history.tsv")))
            self.assertTrue(os.path.exists(os.path.join(fold_dir,
                                                        "sample_list.json")))
            self.assertTrue(os.path.exists(os.path.join(fold_dir,
                                                        "model.hdf5")))

    #-------------------------------------------------#
    #                 Split Validation                #
    #-------------------------------------------------#
    def test_EVALUATION_splitValidation(self):
        eval_path = os.path.join(self.tmp_dir.name, "evaluation")
        split_validation(self.sample_list, self.model, percentage=0.3, epochs=3,
                         iterations=None,
                         evaluation_path=eval_path,
                         run_detailed_evaluation=False,
                         draw_figures=False,
                         callbacks=[],
                         return_output=False)
        self.assertTrue(os.path.exists(eval_path))

    #-------------------------------------------------#
    #                  Leave One Out                  #
    #-------------------------------------------------#
    def test_EVALUATION_leaveOneOut(self):
        # Create 3D imgaging and segmentation data set
        self.dataset3D = dict()
        for i in range(0, 6):
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
        self.data_io3D = Data_IO(io_interface3D,
                                 input_path=os.path.join(self.tmp_dir3D.name),
                                 output_path=os.path.join(self.tmp_dir3D.name),
                                 batch_path=tmp_batches, delete_batchDir=False)
        # Initialize Preprocessor
        self.pp3D = Preprocessor(self.data_io3D, batch_size=2,
                                 data_aug=None, analysis="fullimage")
        # Initialize Neural Network
        model = Neural_Network(self.pp3D)
        # Get sample list
        self.sample_list3D = self.data_io3D.get_indiceslist()

        eval_path = os.path.join(self.tmp_dir3D.name, "evaluation")
        leave_one_out(self.sample_list3D, model, epochs=3, iterations=None,
                      evaluation_path=eval_path, callbacks=[])
        self.assertTrue(os.path.exists(eval_path))
        # Cleanup stuff
        self.tmp_dir3D.cleanup()
