import numpy
import scipy.misc
import pickle

def visualize(batch, prefix, highlight=False):
    if highlight:
        batch = numpy.argmax(batch, axis=-1)
        batch = batch*100
    else:
        batch = numpy.squeeze(batch, axis=-1)
    for i in range(0, len(batch)):
        for j in range(0, len(batch[i])):
            fpath = "visualization/" + str(prefix) + "." + str(i) + "-" + str(j) + ".png"
            scipy.misc.imsave(str(fpath), batch[i][j])
