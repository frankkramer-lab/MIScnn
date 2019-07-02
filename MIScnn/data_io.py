#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
import os
from re import match
import pickle
import shutil
#Internal libraries/scripts
import utils.mri_sample as CNNsolver_MRI
from utils.nifti_io import load_volume_nii, load_segmentation_nii, save_segmentation

#-----------------------------------------------------#
#                     Case Loader                     #
#-----------------------------------------------------#
# Load a MRI in NIFTI format and creates a MRI sample object
def case_loader(case_id, data_path, load_seg=True):
    # Read volume NIFTI file
    volume = load_volume_nii(case_id, data_path)
    # Create and return a MRI_Sample object
    mri = CNNsolver_MRI.MRI(volume)
    # IF needed read the provided segmentation for current MRI sample
    if load_seg:
        segmentation = load_segmentation_nii(case_id, data_path)
        mri.add_segmentation(segmentation)
    # Return MRI sample object
    return mri

#-----------------------------------------------------#
#                  Prediction backup                  #
#-----------------------------------------------------#
# Save a segmentation prediction in NIFTI format
def save_prediction(pred, case_id, out_path):
    # Create the output directory if not existent
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    # Backup the prediction
    save_segmentation(pred, case_id, out_path)

#-----------------------------------------------------#
#                   MRI Fast Access                   #
#-----------------------------------------------------#
# Backup a MRI object to a pickle for fast access later
def backup_batches(batches_vol, batches_seg, path, case_id):
    # Create model directory of not existent
    if not os.path.exists(path):
        os.mkdir(path)
    # Create subdirectory for the case if not existent
    case_dir = os.path.join(path, "tmp.case_" + str(case_id).zfill(5))
    if not os.path.exists(case_dir):
        os.mkdir(case_dir)
    # Backup volume batches
    if batches_vol is not None:
        for i, batch in enumerate(batches_vol):
            out_path = os.path.join(case_dir,
                                    "batch_vol." + str(i) + ".pickle")
            with open(out_path, 'wb') as pickle_out:
                pickle.dump(batch, pickle_out)
    # Backup segmentation batches
    if batches_seg is not None:
        for i, batch in enumerate(batches_seg):
            out_path = os.path.join(case_dir,
                                    "batch_seg." + str(i) + ".pickle")
            with open(out_path, 'wb') as pickle_out:
                pickle.dump(batch, pickle_out)

# Load a MRI object from a pickle for fast access
def batch_load(id_tuple, path, vol=True):
    # Parse ids
    case_id = id_tuple[0]
    batch_id = id_tuple[1]
    # Identify batch type (volume or segmentation)
    if vol:
        batch_type = "batch_vol"
    else:
        batch_type = "batch_seg"
    # Set up file path
    in_path = os.path.join(path, "tmp.case_" + str(case_id).zfill(5),
                           batch_type + "." + str(batch_id) + ".pickle")
    with open(in_path, 'rb') as pickle_in:
        batch = pickle.load(pickle_in)
    # Return loaded batch
    return batch

# Clean up all temporary pickles
def batch_pickle_cleanup():
    # Iterate over each file in the model directory
    directory = os.listdir("model")
    for file in directory:
        # IF file matches temporary subdirectory name pattern -> delete it
        if match("tmp\.case\_[0-9]+", file) is not None:
            shutil.rmtree(os.path.join("model", file))

#-----------------------------------------------------#
#               Evaluation Data Backup                #
#-----------------------------------------------------#
# Backup evaluation as TSV (Tab Separated File)
def save_evaluation(data, directory, file, start=False):
    # Set up the evaluation directory
    if start and not os.path.exists(directory):
        os.mkdir(directory)
    # Define the writing type
    if start:
        writer_type = "w"
    else:
        writer_type = "a"
    # Opening file writer
    output_path = os.path.join(directory, file)
    with open(output_path, writer_type) as fw:
        # Join the data together to a row
        line = "\t".join(map(str, data)) + "\n"
        fw.write(line)
