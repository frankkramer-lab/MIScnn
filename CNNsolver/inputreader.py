#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
import os.path
import nibabel as nib
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
    def case_loader(self, case_id, load_seg=True):
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
