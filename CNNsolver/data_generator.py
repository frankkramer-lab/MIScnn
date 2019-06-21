#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
import keras
import numpy as np
#Internal libraries/scripts
from data_io import case_loader

#-----------------------------------------------------#
#              MRI Data Generator (Keras)             #
#-----------------------------------------------------#
# MRI Data Generator for training and predicting (WITH-/OUT segmentation)
## Returns a batch containing multiple patches for each call
def DataGenerator(casePointer, data_path, training=False, shuffle=False):
    # Initialize a counter for MRI internal batch pointer and current Case MRI
    batch_pointer = None
    current_case = -1
    current_mri = None
    # Start while loop (Keras specific requirement)
    while True:
        for i in range(0, len(casePointer)):
            # At the beginning of an epoch: Reset batchPointer & shuffle batches
            if i == 0:
                batch_pointer = 0
                if shuffle:
                    casePointer = shuffle_batches(casePointer)
            # Load the next pickled MRI object if necessary
            if current_case != casePointer[i]:
                batch_pointer = 0
                current_case = casePointer[i]
                current_mri = case_loader(current_case,
                                          data_path,
                                          load_seg=training,
                                          pickle=True)
            # Load next volume batch
            batch_vol = current_mri.batches_vol[batch_pointer]
            # IF batch is for training -> return next vol & seg batch
            if training:
                # Load next segmentation batch
                batch_seg = current_mri.batches_seg[batch_pointer]
                # Update batch_pointer
                batch_pointer += 1
                # Return volume and segmentation batch
                yield batch_vol, batch_seg
            # IF batch is for predicting -> return next vol batch
            else:
                # Update batch_pointer
                batch_pointer += 1
                # Return volume batch
                yield batch_vol

# At every epoch end: Shuffle batches
def shuffle_batches(casePointer):
    # Create a unique indices map
    cp_unique, cp_counts = np.unique(casePointer,
                                     return_counts=True)
    cp_indices = np.arange(len(cp_unique))
    # Shuffle the indices map
    np.random.shuffle(cp_indices)
    # Add the casePointers accordingly to the shuffled indices map
    cp_shuffled = []
    for i in cp_indices:
        cp_shuffled.extend([cp_unique[i]] * cp_counts[i])
    # Return the shuffled casePointer array
    return cp_shuffled
