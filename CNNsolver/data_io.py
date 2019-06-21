#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
import os
from re import match
import pickle
#Internal libraries/scripts
import mri_sample as CNNsolver_MRI
from utils.nifti_io import load_volume_nii, load_segmentation_nii, save_segmentation

#-----------------------------------------------------#
#                     Case Loader                     #
#-----------------------------------------------------#
# Load a MRI in NIFTI format and creates a MRI sample object
def case_loader(case_id, data_path, load_seg=True, pickle=False):
    # IF pickle modus is True and MRI pickle file exist
    if pickle and os.path.exists("model/MRI.case" + str(case_id) + \
                                 ".pickle"):
        # Load MRI object from pickle and return MRI
        mri = mri_pickle_load(case_id)
        return mri
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
def mri_pickle_backup(case_id, mri):
    pickle_out = open("model/MRI.case" + str(case_id) + ".pickle","wb")
    pickle.dump(mri, pickle_out)
    pickle_out.close()

# Load a MRI object from a pickle for fast access
def mri_pickle_load(case_id):
    pickle_in = open("model/MRI.case" + str(case_id) + ".pickle","rb")
    mri = pickle.load(pickle_in)
    return mri

# Clean up all temporary pickles
def mri_pickle_cleanup():
    # Iterate over each file in the model directory
    directory = os.listdir("model")
    for file in directory:
        # IF file matches temporary MRI pickle name pattern -> delete it
        if match("MRI\.case[0-9]+\.pickle", file) is not None:
            os.remove(os.path.join("model", file))

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
