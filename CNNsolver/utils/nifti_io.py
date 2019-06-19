#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
import os
import nibabel as nib

#-----------------------------------------------------#
#                    NIFTI Reader                     #
#-----------------------------------------------------#
## Code source from provided starter_code.utils on kits19 github
## Slightly modified for easier usage

# Read a volumne NIFTI file from the kits19 data directory
def load_volume_nii(cid, data_path):
    # Resolve location where data should be living
    if not os.path.exists(data_path):
        raise IOError(
            "Data path, {}, could not be resolved".format(str(data_path))
        )
    # Get case_id from provided cid
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
def load_segmentation_nii(cid, data_path):
    # Resolve location where data should be living
    if not os.path.exists(data_path):
        raise IOError(
            "Data path, {}, could not be resolved".format(str(data_path))
        )
    # Get case_id from provided cid
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


# Read a prediction NIFTI file from the prediction directory
def load_prediction_nii(pid, data_path):
    # Resolve location where data should be living
    if not os.path.exists(data_path):
        raise IOError(
            "Data path, {}, could not be resolved".format(str(data_path))
        )
    # Parse the prediction name for the provided pid
    pid = int(pid)
    pred_file = "prediction_{:05d}".format(pid) + ".nii.gz"
    # Make sure that pred_id exists under the data_path
    pred_path = os.path.join(data_path, pred_file)
    if not os.path.exists(pred_path):
        raise ValueError(
            "Prediction could not be found \"{}\"".format(pred_path.name)
        )
    # Load prediction from NIFTI file
    pred = nib.load(pred_path)
    return pred

#-----------------------------------------------------#
#                    NIFTI Writer                     #
#-----------------------------------------------------#
# Write a segmentation prediction into in the NIFTI file format
def save_segmentation(seg, cid, output_path):
    # Resolve location where data should be written
    if not os.path.exists(output_path):
        raise IOError(
            "Data path, {}, could not be resolved".format(str(output_path))
        )
    # Convert numpy array to NIFTI
    nifti = nib.Nifti1Image(seg, None)
    nifti.get_data_dtype() == seg.dtype
    # Save segmentation to disk
    nib.save(nifti, os.path.join(output_path,
                                 "prediction_" + str(cid).zfill(5) + ".nii.gz"))
