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

def visualize_evaluation(case_id, truth, pred, eva_path):
    # Create a figure and two axes objects from matplot
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Initialize the two subplots (axes) with an empty 512x512 image
    data = np.zeros((512, 512))
    ax1.set_title("Ground Truth")
    ax2.set_title("Prediction")
    img1 = ax1.imshow(data, cmap="gray", vmin=0, vmax=2)
    img2 = ax2.imshow(data, cmap="gray", vmin=0, vmax=2)
    # Update function for both images to show the slice for the current frame
    def update(i):
        plt.suptitle("Case ID: " + str(case_id) + " - " + "Frame: " + str(i))
        img1.set_data(truth[i])
        img2.set_data(pred[i])
        return [img1, img2]
    # Compute the animation (gif)
    ani = animation.FuncAnimation(fig, update, frames=len(truth), interval=5,
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
