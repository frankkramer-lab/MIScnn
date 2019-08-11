#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
import numpy as np
import math
import itertools
from tqdm import tqdm
from keras.utils import to_categorical
#Internal libraries/scripts
from miscnn.data_io import case_loader, backup_batches
from miscnn.utils.matrix_operations import slice_3Dmatrix

#-----------------------------------------------------#
#          Patches and Batches Preprocessing          #
#-----------------------------------------------------#
# Load and preprocess all MRI's to batches for later training or prediction
def preprocessing_MRIs(cases, config, training=False, validation=False):
    # Parameter initialization
    batchPointer = []
    # Iterate over each case
    for i in tqdm(cases):
        # Initialize batches
        batches_vol = None
        batches_seg = None
        # Load the MRI of the current case
        mri = case_loader(i, config["data_path"], load_seg=training)
        # IF scaling: Scale each volume value to [0,1]
        if config["scale_input_values"]:
            mri.vol_data = scale_volume_values(mri.vol_data)
        # Slice volume into patches
        patches_vol = slice_3Dmatrix(mri.vol_data,
                                     config["patch_size"],
                                     config["overlap"])
        # IF training: Slice segmentation into patches
        if training:
            patches_seg = slice_3Dmatrix(mri.seg_data,
                                         config["patch_size"],
                                         config["overlap"])
            # IF skip blank patches: remove all blank patches from the list
            if config["skip_blanks"] and not validation:
                patches_vol, patches_seg = remove_blanks(patches_vol,
                                                         patches_seg)
            # IF rotation: Rotate patches for data augmentation
            if config["rotation"] and not validation:
                patches_vol, patches_seg = rotate_patches(patches_vol,
                                                          patches_seg)
            # IF flipping: Reflect/Flip patches for data augmentation
            if config["flipping"] and not validation:
                patches_vol, patches_seg = flip_patches(patches_vol,
                                                        patches_seg,
                                                        config["flip_axis"])
        # Calculate the number of batches for this MRI
        steps = math.ceil(len(patches_vol) / config["batch_size"])
        # Create batches from the volume patches
        batches_vol = create_batches(patches_vol,
                                     config["batch_size"],
                                     steps,
                                     train=training)
        # IF training: Create batches from the segmentation batches
        if training:
            batches_seg = create_batches(patches_seg,
                                         config["batch_size"],
                                         steps)
            # Transform digit segmentation classes into categorical
            for b, batch in enumerate(batches_seg):
                batches_seg[b] = to_categorical(batch,
                                                num_classes=config["classes"])
        # Backup volume & segmentation batches as temporary files
        # for later usage
        backup_batches(batches_vol, batches_seg,
                       path=config["model_path"],
                       case_id=i)
        # Extend the current batches to the batchPointer list
        current_batchPointer = itertools.product([i], range(len(batches_vol)))
        batchPointer.extend(list(current_batchPointer))
    # Return the batchPointer list for the data generator usage later
    return batchPointer

# Create batches from a list of patches
def create_batches(patches, batch_size, steps, train=True):
    # Initialize result list
    batches = []
    # Create a batch in each step
    for i in range(0, steps):
        # Assign patches to the next batch
        start = i * batch_size
        end = start + batch_size
        if end > len(patches):
            end = len(patches)
            if (end - batch_size) >= 0 and train:
                start = end - batch_size
        # Concatenate volume patches into the batch
        batch = np.concatenate(patches[start:end], axis=0)
        # Append batch to result batches list
        batches.append(batch)
    # Return resulting batches list
    return batches

#-----------------------------------------------------#
#            Other preprocessing functions            #
#-----------------------------------------------------#
# Remove all blank patches (with only background)
def remove_blanks(patches_vol, patches_seg, background_class=0):
    # Iterate over each patch
    for i in reversed(range(0, len(patches_seg))):
        # IF patch DON'T contain any non background class -> remove it
        if not np.any(patches_seg[i] != background_class):
            del patches_vol[i]
            del patches_seg[i]
    # Return all non blank patches
    return patches_vol, patches_seg

# Scale the input volume voxel values between [0,1] or with Z-Score
def scale_volume_values(volume, z_score=True):
    if z_score:
        # Compute mean and standard deviation
        mean = np.mean(volume)
        std = np.std(volume)
        # Scaling
        volume_normalized = (volume - mean) / std
    else:
        # Identify minimum and maximum
        max_value = np.max(volume)
        min_value = np.min(volume)
        # Scaling
        volume_normalized = (volume - min_value) / (max_value - min_value)
    # Return scaled volume
    return volume_normalized

# Rotate patches at the y-z axis
def rotate_patches(patches_vol, patches_seg):
    # Initialize lists for rotated patches
    rotated_vol = []
    rotated_seg = []
    # Iterate over 90,180,270 degree (1*90, 2*90, 3*90)
    for times in range(1,4):
        # Iterate over each patch
        for i in range(len(patches_vol)):
            # Rotate volume & segmentation and cache rotated patches
            patch_vol_rotated = np.rot90(patches_vol[i], k=times, axes=(2,3))
            rotated_vol.append(patch_vol_rotated)
            patch_seg_rotated = np.rot90(patches_seg[i], k=times, axes=(2,3))
            rotated_seg.append(patch_seg_rotated)
    # Add rotated patches to the original patches lists
    patches_vol.extend(rotated_vol)
    patches_seg.extend(rotated_seg)
    # Return processed patches lists
    return patches_vol, patches_seg

# Flip patches at the provided axes
def flip_patches(patches_vol, patches_seg, axis=(-2)):
    # Initialize list of flipped patches
    flipped_vol = []
    flipped_seg = []
    # Iterate over each patch
    for i in range(len(patches_vol)):
        # Flip volume & segmentation and cache flipped patches
        patch_vol_flipped = np.flip(patches_vol[i], axis=axis)
        flipped_vol.append(patch_vol_flipped)
        patch_seg_flipped = np.flip(patches_seg[i], axis=axis)
        flipped_seg.append(patch_seg_flipped)
    # Add flipped patches to the original patches lists
    patches_vol.extend(flipped_vol)
    patches_seg.extend(flipped_seg)
    # Return processed patches lists
    return patches_vol, patches_seg
