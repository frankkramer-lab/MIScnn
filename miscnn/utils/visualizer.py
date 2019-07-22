#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

#-----------------------------------------------------#
#             Functions for Visualization             #
#-----------------------------------------------------#
# Visualize loss and metric plot for training
def visualize_training(history, prefix, eva_path):
    # Set up the evaluation directory
    if not os.path.exists(eva_path):
        os.mkdir(eva_path)
    # Plot the generalized dice coefficient
    plt.plot(history.history['dice_classwise'])
    plt.plot(history.history['val_dice_classwise'])
    plt.title('Generalized Dice coefficient')
    plt.ylabel('Dice coefficient')
    plt.xlabel('Epoch')
    plt.legend(['Train Set', 'Test Set'], loc='upper left')
    out_path = os.path.join(eva_path,
                            "dice_classwise." + str(prefix) + ".png")
    plt.savefig(out_path)
    plt.close()
    # Plot the tversky loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Tvsersky Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Set', 'Test Set'], loc='upper left')
    out_path = os.path.join(eva_path,
                            "tversky_loss." + str(prefix) + ".png")
    plt.savefig(out_path)
    plt.close()

def visualize_evaluation(case_id, vol, truth, pred, eva_path):
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
