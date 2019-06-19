#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
import matplotlib.pyplot as plt

#-----------------------------------------------------#
#             Functions for Visualization             #
#-----------------------------------------------------#
# Visualize loss and metric plot for training
def visualize_training(history, fold):
    plt.plot(history.history['dice_classwise'])
    plt.plot(history.history['val_dice_classwise'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("dice_classwise." + str(fold) + ".png")

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("tversky_loss." + str(fold) + ".png")

def visualize_evaluation(truth, pred):
    # Iterate over each slice
    for i in range(len(truth)):
        fig, axeslist = plt.subplots(ncols=2, nrows=1)
        axeslist.ravel()[0].imshow(truth[i], cmap='gray')
        axeslist.ravel()[0].set_title("Ground Truth")
        axeslist.ravel()[1].imshow(pred[i], cmap='gray')
        axeslist.ravel()[1].set_title("Prediction")
        plt.savefig("visualization/slice_" + str(i) + ".png")
        plt.close()
