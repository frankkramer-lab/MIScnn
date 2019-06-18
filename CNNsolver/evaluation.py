#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import numpy as np
# Internal libraries/scripts
from utils.visualizer import visualize_training

#-----------------------------------------------------#
#                   Cross-Validation                  #
#-----------------------------------------------------#
def cross_validation(model, config):
    # Parse variables
    n_folds = config["n_folds"]
    # Randomly permute the case list
    cases_permuted =  np.random.permutation(config["cases"])
    # Split case list into folds
    folds = np.array_split(cases_permuted, n_folds)
    # Start cross-validation
    for i in range(len(folds)):
        # Subset training and validation data set
        training = folds[i]
        validation = np.delete(folds, folds[i])
        # Run training & validation
        history = model.evaluate(training, validation)
        # Draw plots for the training & validation
        print(history)
        visualize_training(history)
#
# from data_io import case_loader
# import scipy.misc
# def visual_evaluation(case_list, data_path):
#     # Iterate over each case
#     for id in case_list:
#         mri = case_loader(id, data_path, load_seg=True, pickle=True)
#         res = mri.seg_data*100
#         for i in range(0, len(res)):
#             fpath = "visualization/" + "real" + "." + "bild" + str(i) + ".png"
#             conv = numpy.reshape(res[i], (512,512))
#             scipy.misc.imsave(str(fpath), conv)
#
#         res = mri.pred_data*100
#         for i in range(0, len(res)):
#             fpath = "visualization/" + "pred" + "." + "bild" + str(i) + ".png"
#             conv = numpy.reshape(res[i], (512,512))
#             scipy.misc.imsave(str(fpath), conv)
#
#         freqArray = mri.pred_data[135].flatten()
#         counter = dict()
#         for x in freqArray:
#             if x in counter:
#                 counter[x] += 1
#             else:
#                 counter[x] = 1
#
#         print(counter)
