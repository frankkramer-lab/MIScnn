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
# External libraries
import matplotlib.pyplot as plt
import os

#-----------------------------------------------------#
#      Create evaluation figures of a validation      #
#-----------------------------------------------------#
""" Function for automatic drawing loss/metric vs epochs figures between training and
    testing data set. The plots will be saved in the provided evaluation directory.

Args:
    history (dictionary):                   A Keras history dictionary resulted from a validation.
    metrics (List of Metric Functions):     List of one or multiple Metric Functions, which will be shown during training.
                                            Any Metric Function defined in Keras, in miscnn.neural_network.metrics or any custom
                                            metric function, which follows the Keras metric guidelines, can be used.
    evaluation_path (string):               Path to the evaluation data directory. This directory will be created and
                                            used for storing all kinds of evaluation results during the validation processes.
"""
def plot_validation(history, metrics, evaluation_directory):
    # Plot the figure for the loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title("Validation: " + "Loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Set', 'Test Set'], loc='upper left')
    out_path = os.path.join(evaluation_directory,
                            "validation." + "loss" + ".png")
    plt.savefig(out_path)
    plt.close()
    # Plot figures for the other metrics
    for metric_function in metrics:
        # identify metric name
        if hasattr(metric_function, "__name__"):
            metric_name = metric_function.__name__
        else:
            metric_name = metric_function.name
        # Plot metric
        plt.plot(history[metric_name])
        plt.plot(history["val_" + metric_name])
        plt.title("Validation: " + metric_name)
        plt.ylabel(metric_name)
        plt.xlabel('Epoch')
        plt.legend(['Train Set', 'Test Set'], loc='upper left')
        out_path = os.path.join(evaluation_directory,
                                "validation." + metric_name + ".png")
        plt.savefig(out_path)
        plt.close()

def visualize_frequencies_1D(sample, out_path = "visualization", sample_rate = None):
    arr = np.flatten(sample.img_data)
    size = len(arr)
    
    transform = np.fft.rfft(arr)
    freq = np.fft.rfftfreq(1/sample_rate, sample_rate)
        
    plt.plot(freq, np.absolute(transform)) # fft_data is a complex number, so the magnitude is computed here
    
    plt.xlim(np.amin(freq), np.amax(freq))
    plt.savefig(sample.index + ".png")
    plt.clf()
        

def visualize_pixel_dist(sample_list, functor, name = "graph", min = 0, max = 2000, cutoff = True):
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")
    plt.xlim([min, max])
    
    for index in tqdm(sample_list):
        sample = functor(index)
        if (cutoff):
            masked = sample.img_data[np.where((sample.img_data < max) & (sample.img_data > min))]
        else:
            masked = sample.img_data
        histogram, bin_edges = np.histogram(masked, bins=(max - min), range=(min, max))
        histogram = histogram / masked.size
        plt.plot(bin_edges[0:-1], histogram)
    
    plt.savefig(name + ".png")
    plt.clf()

