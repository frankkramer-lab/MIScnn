#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
import numpy as np
import math
from keras.utils import to_categorical
#Internal libraries/scripts
import inputreader as CNNsolver_IR
from utils.matrix_operations import slice_3Dmatrix

#-----------------------------------------------------#
#          Patches and Batches Preprocessing          #
#-----------------------------------------------------#
# Load and preprocess all MRI's to batches for later training or prediction
def preprocessing_MRIs(cases, config, training=False):
    print("Preprocessing of the Magnetic Resonance Images")
    # Parameter initialization
    casePointer = []
    # Create a Input Reader instance
    reader = CNNsolver_IR.InputReader(config["data_path"])
    # Iterate over each case
    for i in cases:
        # Load the MRI of the current case
        mri = reader.case_loader(i, load_seg=training, pickle=False)
        # Slice volume into patches
        patches_vol = slice_3Dmatrix(mri.vol_data,
                                     config["patch_size"],
                                     config["overlap"])
        # Calculate the number of batches for this MRI
        steps = math.ceil(len(patches_vol) / config["batch_size"])
        # Create batches from the volume patches
        batches_vol = create_batches(patches_vol,
                                     config["batch_size"],
                                     steps)
        mri.add_batches(batches_vol, vol=True)
        # IF training: Slice segmentation into patches and create batches
        if training:
            patches_seg = slice_3Dmatrix(mri.seg_data,
                                         config["patch_size"],
                                         config["overlap"])
            batches_seg = create_batches(patches_seg,
                                         config["batch_size"],
                                         steps)
            mri.add_batches(batches_seg, vol=False)
        # Backup MRI to pickle for faster access in later usages
        reader.mri_pickle_backup(i, mri)
        # Save the number of steps in the casePointer list
        casePointer.extend([i] * steps)
    # Return the casePointer list for the data generator usage later
    return casePointer

# Create batches from a list of patches
def create_batches(patches, batch_size, steps):
    # Initialize result list
    batches = []
    # Create a batch in each step
    for i in range(0, steps):
        # Assign patches to the next batch
        start = i * batch_size
        end = start + batch_size
        if end > len(patches):
            end = len(patches)
        # Concatenate volume patches into the batch
        batch = np.concatenate(patches[start:end], axis=0)
        # Append batch to result batches list
        batches.append(batch)
    # Return resulting batches list
    return batches

#-----------------------------------------------------#
#              MRI Data Generator (Keras)             #
#-----------------------------------------------------#
# MRI Data Generator for training and predicting (WITH-/OUT segmentation)
## Returns a batch containing multiple patches for each call
def data_generator(casePointer, data_path, classes, training=False):
    # Initialize a counter for MRI internal batch pointer and current Case MRI
    batch_pointer = None
    current_case = -1
    current_mri = None
    # Create a Input Reader instance
    reader = CNNsolver_IR.InputReader(data_path)
    # Start while loop (Keras specific requirement)
    while True:
        for i in range(0, len(casePointer)):
            # Load the next pickled MRI object if necessary
            if current_case != casePointer[i]:
                batch_pointer = 0
                current_case = casePointer[i]
                current_mri = reader.case_loader(current_case,
                                                 load_seg=training,
                                                 pickle=True)
            # Load next volume batch
            batch_vol = current_mri.batches_vol[batch_pointer]
            # IF batch is for training -> return next vol & seg batch
            if training:
                # Load next segmentation batch
                batch_seg = current_mri.batches_seg[batch_pointer]
                # Transform digit segmentation classes into categorical
                batch_seg = to_categorical(batch_seg, num_classes=classes)
                # Update batch_pointer
                batch_pointer += 1
                # Return volume and segmentation batch
                yield(batch_vol, batch_seg)
            # IF batch is for predicting -> return next vol batch
            else:
                # Update batch_pointer
                batch_pointer += 1
                # Return volume batch
                yield(batch_vol)

#-----------------------------------------------------#
#            Other preprocessing functions            #
#-----------------------------------------------------#
#NOT USED
def convert_to_grayscale(volume):
    hu_min = -512
    hu_max = 512
    # Clip at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = np.clip(volume, hu_min, hu_max)

    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255*im_volume
    return np.stack((im_volume, im_volume, im_volume), axis=-1)
