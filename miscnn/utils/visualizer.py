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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

def visualize_evaluation(case_id, vol, truth, pred, eva_path):
    # Squeeze image files to remove channel axis
    # THIS IS JUST A TEMPORARY SOLUTION
    # THIS JUST WORKS FOR GREYSCALE IMAGES!!
    vol = np.squeeze(vol, axis=-1)
    truth = np.squeeze(truth, axis=-1)
    pred = np.squeeze(pred, axis=-1)
    # Color volumes according to truth and pred segmentation
    vol_truth = overlay_segmentation(vol, truth)
    vol_pred = overlay_segmentation(vol, pred)
    # Create a figure and two axes objects from matplot
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Initialize the two subplots (axes) with an empty 512x512 image
    data = np.zeros(vol.shape[1:3])
    ax1.set_title("Ground Truth")
    ax2.set_title("Prediction")
    img1 = ax1.imshow(data)
    img2 = ax2.imshow(data)
    # Update function for both images to show the slice for the current frame
    def update(i):
        plt.suptitle("Case ID: " + str(case_id) + " - " + "Slice: " + str(i))
        img1.set_data(vol_truth[i])
        img2.set_data(vol_pred[i])
        return [img1, img2]
    # Compute the animation (gif)
    ani = animation.FuncAnimation(fig, update, frames=len(truth), interval=10,
                                  repeat_delay=0, blit=False)
    # Set up the output path for the gif
    if not os.path.exists(eva_path):
        os.mkdir(eva_path)
    file_name = "visualization.case_" + str(case_id).zfill(5) + ".gif"
    out_path = os.path.join(eva_path, file_name)
    # Save the animation (gif)
    ani.save(out_path, writer='imagemagick', fps=30)
    # Close the matplot
    plt.close()

def visualize_sample(img, seg, index, eva_path):
    # Squeeze image files to remove channel axis
    # THIS IS JUST A TEMPORARY SOLUTION
    # THIS JUST WORKS FOR GREYSCALE IMAGES!!
    img = np.squeeze(img, axis=-1)
    seg = np.squeeze(seg, axis=-1)
    # Color image with segmentation if present
    if seg is not None:
        img = overlay_segmentation(img, seg)
    # Create a figure and an axes object from matplot
    fig, ax = plt.subplots()
    # Initialize the plot with an empty image
    data = np.zeros(img.shape[1:3])
    ax.set_title("Visualization")
    axis_img = ax.imshow(data)
    # Update function for both images to show the slice for the current frame
    def update(i):
        plt.suptitle("Slice: " + str(i))
        axis_img.set_data(img[i])
        return [axis_img]
    # Compute the animation (gif)
    ani = animation.FuncAnimation(fig, update, frames=len(seg), interval=10,
                                  repeat_delay=0, blit=False)
    # Set up the output path for the gif
    if not os.path.exists(eva_path):
        os.mkdir(eva_path)
    file_name = "visualization." + str(index) + ".gif"
    out_path = os.path.join(eva_path, file_name)
    # Save the animation (gif)
    ani.save(out_path, writer='imagemagick', fps=30)
    # Close the matplot
    plt.close()


#-----------------------------------------------------#
#                     Subroutines                     #
#-----------------------------------------------------#
# Based on: https://github.com/neheller/kits19/blob/master/starter_code/visualize.py
def overlay_segmentation(vol, seg):
    # Scale volume to greyscale range
    vol_greyscale = (255*(vol - np.min(vol))/np.ptp(vol)).astype(int)
    # Convert volume to RGB
    vol_rgb = np.stack([vol_greyscale, vol_greyscale, vol_greyscale], axis=-1)
    # Initialize segmentation in RGB
    shp = seg.shape
    seg_rgb = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.int)
    # Set class to appropriate color
    seg_rgb[np.equal(seg, 1)] = [255, 0,   0]
    seg_rgb[np.equal(seg, 2)] = [0,   0, 255]
    # Get binary array for places where an ROI lives
    segbin = np.greater(seg, 0)
    repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)
    # Weighted sum where there's a value to overlay
    alpha = 0.3
    vol_overlayed = np.where(
        repeated_segbin,
        np.round(alpha*seg_rgb+(1-alpha)*vol_rgb).astype(np.uint8),
        np.round(vol_rgb).astype(np.uint8)
    )
    # Return final volume with segmentation overlay
    return vol_overlayed
