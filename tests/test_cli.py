#==============================================================================#
#  Author:       Philip Meyer                                                  #
#  Copyright:    2021 IT-Infrastructure for Translational Medical Research,    #
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

import miscnn.cli.data_exploration as data_exp
import unittest
import tempfile
import os
import nibabel as nib
import numpy as np
import pandas as pd
from miscnn import Data_IO
from miscnn.data_loading.interfaces import NIFTI_interface


class MockArgParser():
    def __init__(self):
        self.default = ""
        self.shortName = []
        self.name = []
    
    def add_argument(self, shortname, name, *args, **kwargs):
        self.shortName.append(shortname)
        self.name.append(name)
    
    def set_defaults(self, *args, **kwargs):
        self.default = kwargs["which"]

class ArgData():
    def __init__(self, data_dir, imagetype):
        self.data_dir = data_dir
        self.imagetype = imagetype

def write_sample(sample_data, path, name):
        if not os.path.exists(path):
            raise IOError(
                "Data path, {}, could not be resolved".format(path)
            )
        # Save segmentation to disk
        sample_path = os.path.join(path, name)
        os.mkdir(sample_path)
        nifti = nib.Nifti1Image(sample_data[0], None)
        nib.save(nifti, os.path.join(sample_path, "imaging.nii.gz"))
        nifti = nib.Nifti1Image(sample_data[1], None)
        nib.save(nifti, os.path.join(sample_path, "segmentation.nii.gz"))


class cliTEST(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Create imgaging and segmentation data set
        np.random.seed(1234)
        self.dataset = dict()
        for i in range(0, 10):
            img = np.random.rand(16, 16, 16) * 256
            self.img = img.astype(int)
            seg = np.random.rand(16, 16, 16) * 3
            self.seg = seg.astype(int)
            sample = (self.img, self.seg)
            self.dataset["TEST.sample_" + str(i)] = sample
        # Initialize Dictionary IO Interface
        # Initialize temporary directory
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="tmp.miscnn.")
        self.tmp_data = os.path.join(self.tmp_dir.name, "data")
        os.mkdir(self.tmp_data)
        
        for key, value in self.dataset.items():
            write_sample(value, self.tmp_data, key)
        
        self.dataio = Data_IO(NIFTI_interface(), self.tmp_data)
        
        #perhaps this should be a test but the other things kind of depend on this data
        
        #generate sample dir

    # Delete all temporary files
    @classmethod
    def tearDownClass(self):
        self.tmp_dir.cleanup()

    #-------------------------------------------------#
    #                Base Functionality               #
    #-------------------------------------------------#
    # Class Creation
    def test_checkRegistration(self):
        mockParser = MockArgParser()
        data_exp.register_commands(mockParser)
        
        assert mockParser.default == "data_exp"
        assert len(mockParser.shortName) == len(mockParser.name)
        assert len(mockParser.shortName) > 0
        
    
    # Class Creation
    def test_CLI_setup(self):
        args = ArgData(self.tmp_data, "Unknown")
        
        self.context_data = data_exp.setup_execution(args)
        
        assert isinstance(self.context_data["dataio"], Data_IO)
        
        assert self.context_data["cnt"] == 10
        #verifies the scans work
        assert self.context_data["cnt"] == len(self.context_data["indices"])
        assert self.context_data["cnt"] == len(self.context_data["images"])
        assert self.context_data["cnt"] == len(self.context_data["segmentations"])
        
        self.assertCountEqual(self.context_data["indices"], self.dataset.keys())
        self.assertCountEqual(self.context_data["images"], self.dataset.keys())
        self.assertCountEqual(self.context_data["segmentations"], self.dataset.keys())
        
    
    def test_CLIclassAnalysis(self):
        result = data_exp.execute_class_analysis({"data_dir": self.tmp_data, "dataio":self.dataio, "segmentations":self.dataset.keys()})
        assert len(result) <= 3
        assert len(result) > 0
        
    def test_CLIstructureAnalysis(self):
        dataframe = pd.DataFrame({"name":[]})
        dataframe = data_exp.execute_structure_analysis({"data_dir": self.tmp_data, "dataio":self.dataio, "indices":self.dataset.keys()}, dataframe)
        
        expectation = pd.DataFrame.from_dict({id: [name, (16, 16, 16, 1), [1, 1, 1]] for id, name in enumerate(self.dataset.keys())}, orient="index",columns=["name", "shape", "voxel_spacing"])
        
        pd.testing.assert_frame_equal(dataframe,expectation,check_names=False)
        
    def test_CLIminmaxAnalysis(self):
        df, minmax_data = data_exp.execute_minmax_analysis({"data_dir": self.tmp_data, "dataio":self.dataio, "indices":self.dataset.keys()}, pd.DataFrame({"name":[]}))
        
        assert (df["minimum"] < df["maximum"]).all()
        assert (df["minimum"] >= minmax_data["glob_min"]).all()
        
        assert (df["maximum"] <= minmax_data["glob_max"]).all()
        
    def test_CLIminaxSegAnalysis(self):
        df, minmax_data = data_exp.execute_minmax_seg_analysis({"data_dir": self.tmp_data, "dataio":self.dataio, "segmentations":self.dataset.keys()}, pd.DataFrame({"name":[]}))
        
        for cl, minmax in minmax_data.items():
            assert (df["minimum_c" + str(cl)] < df["maximum_c" + str(cl)]).all()
            assert (df["minimum_c" + str(cl)] >= minmax["glob_min"]).all()
            
            assert (df["maximum_c" + str(cl)] <= minmax["glob_max"]).all()
        
    def test_CLIratioAnalysis(self):
        df = data_exp.execute_ratio_analysis({"data_dir": self.tmp_data, "dataio":self.dataio, "indices":self.dataset.keys()}, pd.DataFrame({"name":[]}))
        
        assert df["class_frequency"].apply(lambda x: len(x) == 3).all
    
    def test_CLIbinningAnalysis(self):
        bins = data_exp.execute_binning({"data_dir": self.tmp_data, "dataio":self.dataio, "indices":self.dataset.keys(), "images":self.dataset.keys(), "segmentations":self.dataset.keys()}, pd.DataFrame({"name":[]}), 5)
        
    def test_CLIbinningSegAnalysis(self):
        bins = data_exp.execute_binning_seg({"data_dir": self.tmp_data, "dataio":self.dataio, "segmentations":self.dataset.keys()}, pd.DataFrame({"name":[]}), 5)
    