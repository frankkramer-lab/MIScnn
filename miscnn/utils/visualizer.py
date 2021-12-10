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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import random
from miscnn.data_loading.sample import Sample

#-----------------------------------------------------#
#                  Helper Functions                   #
#-----------------------------------------------------#

def luminosity(r, g, b):
    return r * 0.2126 + g * 0.7152 + b * 0.0722
vec_luminosity = np.vectorize(luminosity)

def normalize_volume(sample, to_greyscale=False, normalize=True):
    if (len(sample.shape) > 4 or len(sample.shape) < 3):
        raise RuntimeError("Expected sample to be a 3 dimensional volume")
    if (to_greyscale and not len(sample.shape) == 4):
        raise RuntimeError("Sample is not RGB")

    img = sample

    if (to_greyscale):
        img[:, :, :] = vec_luminosity(img[:, :, :, 0], img[:, :, :, 1], img[:, :, :, 2])
    elif (len(sample.shape) == 4 and sample.shape[-1] == 1):
        img = np.squeeze(img, axis=-1)

    if (normalize):
        img = 255 * (img - np.min(img)) / np.ptp(img)
        img = img.astype(int)

    return img

def normalize_image(sample, to_greyscale=False, normalize=True):
    if (len(sample.shape) > 3 or len(sample.shape) < 2):
        raise RuntimeError("Expected sample to be a 2 dimensional image")
    if (to_greyscale and not len(sample.shape) == 3):
        raise RuntimeError("Sample is not RGB")

    img = sample

    if (to_greyscale):
        img[:, :] = vec_luminosity(img[:, :, 0], img[:, :, 1], img[:, :, 2])
    elif (len(sample.shape) == 3 and sample.shape[-1] == 1):
        img = np.squeeze(img, axis=-1)

    if (normalize):
        img = 255 * (img - np.min(img)) / np.ptp(img)
        img = img.astype(int)

    return img

def detect_dimensionality(sample):
    shape_size = len(sample.shape)

    if (shape_size == 1):
        raise RuntimeError("Only has one dimension") #this is just a graph tbh
    elif (shape_size == 2):
        return 2;
    elif (shape_size == 3):
        if (sample.shape[-1] in [1, 3]):
            return 2;
        else:
            return 3;
    elif (shape_size == 4):
        return 3;
    else:
        raise RuntimeError("Too many dimensions.")

def normalize(sample, to_greyscale=False, normalize=True):
    dimensionality = detect_dimensionality(sample)
    if dimensionality == 2:
        return normalize_image(sample, to_greyscale, normalize)
    elif dimensionality == 3:
        return normalize_volume(sample, to_greyscale, normalize)

def compute_preprocessed_sample(sample, subfunctions):
    for func in subfunctions:
        func.preprocessing(sample)
    return sample

def load_samples(sample_list, data_io, load_seg, load_pred):
    return [data_io.sample_loader(s, load_seg=load_seg, load_pred=load_pred) for s in sample_list]

def to_samples(sample_list, data_io = None, load_seg = None, load_pred = None):

    if load_seg is None:
        seg = True
    else:
        seg = load_seg

    if load_pred is None:
        pred = True
    else:
        pred = load_pred

    res = []

    for sample in sample_list:
        if isinstance(sample, Sample):
            if load_seg is None:
                seg &= sample.seg_data is not None
            if load_pred is None:
                pred &= sample.pred_data is not None
            res.append(sample)
            continue
        elif isinstance(sample, str):
            res.append(load_samples([sample], data_io, seg, pred))
        elif isinstance(sample, np.ndarray):
            sampleObj = Sample("ndarray_" + str(random.choice(range(999999999)), sample, 1, 0))
            sampleObj.img_data = sample
            res.append(sampleObj);
        else:
            raise ValueError("Cannot interpret an object of type " + str(type(sample)) + " as a sample")

    return res



def visualize_samples(sample_list, out_dir = "vis", mask_seg = False, mask_pred = True, data_io = None, preprocessing = None):
    #create a homogenous datastructure
    samples = to_samples(sample_list, data_io, mask_seg, mask_pred)

    #apply potential preprocessing
    if preprocessing is not None:
        samples = [compute_preprocessed_sample(s, preprocessing) for s in samples]

    #normalize images both in data and structure
    for sample in samples:
        sample.img_data = normalize(sample.img_data, to_greyscale=True, normalize=True)
        if sample.seg_data is not None:
            sample.seg_data = normalize(sample.seg_data, to_greyscale=False, normalize=False)
        elif mask_seg:
            raise RuntimeError("Sample " + sample.index + " lacks the segmentation needed for generating the mask")
        if sample.pred_data is not None:
            sample.pred_data = normalize(sample.pred_data, to_greyscale=False, normalize=False)
        elif mask_pred:
            raise RuntimeError("Sample " + sample.index + " lacks the prediction needed for generating the mask")

        ani = None
        if mask_seg and mask_pred:
            vol_truth = overlay_segmentation(sample.img_data, sample.seg_data)
            vol_pred = overlay_segmentation(sample.img_data, sample.pred_data)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            # Initialize the two subplots (axes) with an empty 512x512 image
            data = np.zeros(vol_truth.shape[1:3])
            ax1.set_title("Ground Truth")
            ax2.set_title("Prediction")
            img1 = ax1.imshow(data)
            img2 = ax2.imshow(data)
            # Update function for both images to show the slice for the current frame
            def update(i):
                plt.suptitle("Case ID: " + str(sample.index) + " - " + "Slice: " + str(i))
                img1.set_data(vol_truth[i])
                img2.set_data(vol_pred[i])
                return [img1, img2]
            # Compute the animation (gif)
            ani = animation.FuncAnimation(fig, update, frames=len(truth), interval=10,
                                          repeat_delay=0, blit=False)
        elif mask_seg:
            vol_truth = overlay_segmentation(sample.img_data, sample.seg_data)
            fig, ax = plt.subplots()
            # Initialize the two subplots (axes) with an empty 512x512 image
            data = np.zeros(vol_truth.shape[1:3])
            ax.set_title("Segmentation")
            img = ax.imshow(data)
            # Update function for both images to show the slice for the current frame
            def update(i):
                plt.suptitle("Case ID: " + str(sample.index) + " - " + "Slice: " + str(i))
                img.set_data(vol_truth[i])
                return [img]
            # Compute the animation (gif)
            ani = animation.FuncAnimation(fig, update, frames=len(vol_truth), interval=10,
                                          repeat_delay=0, blit=False)
        elif mask_pred:
            vol_pred = overlay_segmentation(sample.img_data, sample.pred_data)
            fig, ax = plt.subplots()
            # Initialize the two subplots (axes) with an empty 512x512 image
            data = np.zeros(vol_pred.shape[1:3])
            ax.set_title("Segmentation")
            img = ax.imshow(data)
            # Update function for both images to show the slice for the current frame
            def update(i):
                plt.suptitle("Case ID: " + str(sample.index) + " - " + "Slice: " + str(i))
                img.set_data(vol_pred[i])
                return [img]
            # Compute the animation (gif)
            ani = animation.FuncAnimation(fig, update, frames=len(vol_pred), interval=10,
                                          repeat_delay=0, blit=False)
        else:
            fig, ax = plt.subplots()
            # Initialize the two subplots (axes) with an empty 512x512 image
            data = np.zeros(sample.img_data.shape[1:3])
            ax.set_title("Segmentation")
            img = ax.imshow(data)
            # Update function for both images to show the slice for the current frame
            def update(i):
                plt.suptitle("Case ID: " + str(sample.index) + " - " + "Slice: " + str(i))
                img.set_data(sample.img_data[i])
                return [img]
            # Compute the animation (gif)
            ani = animation.FuncAnimation(fig, update, frames=len(sample.img_data), interval=10,
                                          repeat_delay=0, blit=False)


        # Set up the output path for the gif
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        file_name = "visualization.case_" + str(sample.index).zfill(5) + ".gif"
        out_path = os.path.join(out_dir, file_name)
        # Save the animation (gif)
        ani.save(out_path, writer='imagemagick', fps=30)
        # Close the matplot
        plt.close()

def visualize_prediction_overlap_3D(sample, classes=None, visualization_path = "visualization", alpha = 0.6):
    tp_color = [31,113,80]
    fp_color = [153,12,12]
    fn_color = [3,92,135]
    #true negative is blank as it would create confusion

    if sample.seg_data is None or sample.pred_data is None:
        raise ValueError("Sample needs to have both a segmentation and a prediction map.")

    sample = to_samples([sample])

    vol_greyscale = normalize(sample.img_data)

    if classes is None:
        classes = np.unique(sample.seg_data)
        sample.seg_data = np.squeeze(sample.seg_data, axis=-1)

    # Convert volume to RGB
    vol_rgb = np.stack([vol_greyscale, vol_greyscale, vol_greyscale], axis=-1)

    for c in classes:
        seg_rgb = np.zeros((sample.img_data.shape[0], sample.img_data.shape[1], sample.img_data.shape[2], 3), dtype=np.int)

        seg_broadcast_arr = np.equal(sample.seg_data, c)
        pred_broadcast_arr = np.equal(sample.pred_data, c)

        seg_rgb[ seg_broadcast_arr & pred_broadcast_arr ] = tp_color
        seg_rgb[np.logical_not(seg_broadcast_arr) & pred_broadcast_arr] = fp_color
        seg_rgb[seg_broadcast_arr & np.logical_not(pred_broadcast_arr)] = fn_color

        # Get binary array for places where an ROI lives
        segbin = np.greater(sample.seg_data, 0) | np.greater(sample.pred_data, 0)
        repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)

        # Weighted sum where there's a value to overlay
        vol_overlayed = np.where(
            repeated_segbin,
            np.round(alpha*seg_rgb+(1-alpha)*vol_rgb).astype(np.uint8),
            np.round(vol_rgb).astype(np.uint8)
        )


        fig, ax = plt.subplots()
        # Initialize the plot with an empty image
        data = np.zeros(vol_overlayed.shape[1:3])
        ax.set_title("Confulsion Overlap")
        axis_img = ax.imshow(data)
        # Update function for both images to show the slice for the current frame
        def update(i):
            plt.suptitle("Slice: " + str(i))
            axis_img.set_data(vol_overlayed[i])
            return [axis_img]
        # Compute the animation (gif)
        ani = animation.FuncAnimation(fig, update, frames=len(vol_overlayed), interval=15,
                                      repeat_delay=0, blit=False)
        # Set up the output path for the gif
        if not os.path.exists(visualization_path):
            os.mkdir(visualization_path)
        file_name = "visualization." + str(sample.index) + ".class_" + str(c) + ".gif"
        out_path = os.path.join(visualization_path, file_name)
        # Save the animation (gif)
        ani.save(out_path, writer='imagemagick', fps=30)
        # Close the matplot
        plt.close()

#-----------------------------------------------------#
#                     Subroutines                     #
#-----------------------------------------------------#
# Based on: https://github.com/neheller/kits19/blob/master/starter_code/visualize.py
def overlay_segmentation_greyscale(vol, seg, cm="hsv", alpha=0.3):
    # Convert volume to RGB
    vol_rgb = np.stack([vol, vol, vol], axis=-1)
    # Initialize segmentation in RGB
    shp = seg.shape
    seg_rgb = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.int)

    # Set class to appropriate color
    cmap = matplotlib.cm.get_cmap(cm)
    uniques = np.unique(seg)
    for u in uniques:
        seg_rgb[np.equal(seg, u)] = cmap(u / len(uniques))[0:3]
    # Get binary array for places where an ROI lives
    segbin = np.greater(seg, 0)
    repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)

    # Weighted sum where there's a value to overlay
    vol_overlayed = np.where(
        repeated_segbin,
        np.round(alpha*seg_rgb+(1-alpha)*vol_rgb).astype(np.uint8),
        np.round(vol_rgb).astype(np.uint8)
    )
    # Return final volume with segmentation overlay
    return vol_overlayed
