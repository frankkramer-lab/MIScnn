#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
import os
from re import match
import nibabel as nib
import pickle
#Internal libraries/scripts
import mri_sample as CNNsolver_MRI

#-----------------------------------------------------#
#                 Input Reader - class                #
#-----------------------------------------------------#
class InputReader:
    # Initialize class variables
    data_path = None

    # Create an Input Reader object
    def __init__(self, dp):
        self.data_path = dp

    #-----------------------------------------#
    #               NIFTI Reader              #
    #-----------------------------------------#
    ## Code source from provided starter_code.utils on kits19 github
    ## Slightly modified for easier usage

    # Read a volumne NIFTI file from the kits19 data directory
    def load_volume_nii(self, cid):
        data_path = self.data_path
        # Resolve location where data should be living
        if not os.path.exists(data_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(str(data_path))
            )
        # Get case_id from provided cid)
        try:
            cid = int(cid)
            case_id = "case_{:05d}".format(cid)
        except ValueError:
            case_id = cid
        # Make sure that case_id exists under the data_path
        case_path = os.path.join(data_path, case_id)
        if not os.path.exists(case_path):
            raise ValueError(
                "Case could not be found \"{}\"".format(case_path.name)
            )
        # Load volume from NIFTI file
        vol = nib.load(os.path.join(case_path, "imaging.nii.gz"))
        return vol

    # Read a segmentation NIFTI file from the kits19 data directory
    def load_segmentation_nii(self, cid):
        data_path = self.data_path
        # Resolve location where data should be living
        if not os.path.exists(data_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(str(data_path))
            )
        # Get case_id from provided cid)
        try:
            cid = int(cid)
            case_id = "case_{:05d}".format(cid)
        except ValueError:
            case_id = cid
        # Make sure that case_id exists under the data_path
        case_path = os.path.join(data_path, case_id)
        if not os.path.exists(case_path):
            raise ValueError(
                "Case could not be found \"{}\"".format(case_path.name)
            )
        # Load segmentation from NIFTI file
        seg = nib.load(os.path.join(case_path, "segmentation.nii.gz"))
        return seg

    #-----------------------------------#
    #            Case Loader            #
    #-----------------------------------#
    # Load a MRI in NIFTI format and creates a MRI sample object
    def case_loader(self, case_id, load_seg=True, pickle=False):
        # IF pickle modus is True and MRI pickle file exist
        if pickle and os.path.exists("model/mri_tmp." + str(case_id) + \
                                     ".pickle"):
            # Load MRI object from pickle and return MRI
            mri = self.mri_pickle_load(case_id)
            return mri
        # Read volume NIFTI file
        volume = self.load_volume_nii(case_id)
        # Create and return a MRI_Sample object
        mri = CNNsolver_MRI.MRI(volume)
        # IF needed read the provided segmentation for current MRI sample
        if load_seg:
            segmentation = self.load_segmentation_nii(case_id)
            mri.add_segmentation(segmentation, True)
        # Return MRI sample object
        return mri

    #-----------------------------------#
    #          MRI Fast Access          #
    #-----------------------------------#
    # Backup a MRI object to a pickle for fast access later
    def mri_pickle_backup(self, case_id, mri):
        pickle_out = open("model/MRI.case" + str(case_id) + ".pickle","wb")
        pickle.dump(mri, pickle_out)
        pickle_out.close()

    # Load a MRI object from a pickle for fast access
    def mri_pickle_load(self, case_id):
        pickle_in = open("model/MRI.case" + str(case_id) + ".pickle","rb")
        mri = pickle.load(pickle_in)
        return mri

    # Clean up all temporary pickles
    def mri_pickle_cleanup(self):
        # Iterate over each file in the model directory
        directory = os.listdir("model")
        for file in directory:
            # IF file matches temporary MRI pickle name pattern -> delete it
            if match("MRI\.case[0-9]+\.pickle", file) is not None:
                os.remove(os.path.join("model", file))
