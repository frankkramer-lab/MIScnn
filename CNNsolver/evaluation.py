import numpy
import scipy.misc
import pickle

def visual_evaluation(resList):
    # pickle_out = open("model/dict.pickle","wb")
    # pickle.dump(resList, pickle_out)
    # pickle_out.close()

    # pickle_in = open("model/dict.pickle","rb")
    # resList = pickle.load(pickle_in)

    res = numpy.asarray(resList[0].seg_data)
    res = res*100
    print(res.shape)
    for i in range(0, len(res)):
        fpath = "visualization/" + "real" + "." + "bild" + str(i) + ".png"
        conv = numpy.reshape(res[i], (512,512))
        scipy.misc.imsave(str(fpath), conv)

    res = numpy.asarray(resList[0].pred_data)
    res = res*100
    for i in range(0, len(res)):
        fpath = "visualization/" + "pred" + "." + "bild" + str(i) + ".png"
        conv = numpy.reshape(res[i], (512,512))
        scipy.misc.imsave(str(fpath), conv)
