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



# import numpy
# import scipy.misc
# import pickle
#
# def visualize(batch, prefix, highlight=False):
#     if highlight:
#         batch = numpy.argmax(batch, axis=-1)
#         batch = batch*100
#     else:
#         batch = numpy.squeeze(batch, axis=-1)
#     for i in range(0, len(batch)):
#         for j in range(0, len(batch[i])):
#             fpath = "visualization/" + str(prefix) + "." + str(i) + "-" + str(j) + ".png"
#             scipy.misc.imsave(str(fpath), batch[i][j])
