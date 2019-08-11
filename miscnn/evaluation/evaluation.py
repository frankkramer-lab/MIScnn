#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import numpy as np
import math
# Internal libraries/scripts
import miscnn.neural_network as MIScnn_NN
from miscnn.utils.visualizer import visualize_training, visualize_evaluation
from miscnn.utils.nifti_io import load_segmentation_nii, load_prediction_nii, load_volume_nii
from miscnn.data_io import save_evaluation, update_evalpath

#-----------------------------------------------------#
#                   Cross-Validation                  #
#-----------------------------------------------------#
def cross_validation(cases, config):
    # Randomly permute the case list
    cases_permuted = np.random.permutation(cases)
    # Split case list into folds
    folds = np.array_split(cases_permuted, config["n_folds"])
    fold_indices = list(range(len(folds)))
    # Cache original evaluation path
    eval_path = config["evaluation_path"]
    # Start cross-validation
    for i in fold_indices:
        # Create a Convolutional Neural Network model
        model = MIScnn_NN.NeuralNetwork(config)
        # Subset training and validation data set
        training = np.concatenate([folds[x] for x in fold_indices if x!=i],
                                  axis=0)
        validation = folds[i]
        # Initialize evaluation subdirectory for current fold
        config["evaluation_path"] = update_evalpath("fold_" + str(i), eval_path)
        # Run training & validation
        history = model.evaluate(training, validation)
        # Draw plots for the training & validation
        visualize_training(history, "fold_" + str(i), config["evaluation_path"])
        # Save model to file
        model.dump("fold_" + str(i))
        # Make a detailed validation of the current cv-fold
        detailed_validation(model, validation, "fold_" + str(i), config)

#-----------------------------------------------------#
#                 %-Split Validation                  #
#-----------------------------------------------------#
def split_validation(cases, config):
    # Calculate the number of samples in the testing set
    test_size = int(math.ceil(float(len(cases) * config["per_split"])))
    # Randomly pick samples until %-split percentage
    testing = []
    for i in range(test_size):
        test_sample = cases.pop(np.random.choice(len(cases)))
        testing.append(test_sample)
    # Rename the remaining cases as training
    training = cases
    # Create a Convolutional Neural Network model
    model = MIScnn_NN.NeuralNetwork(config)
    # Run training & validation
    history = model.evaluate(training, testing)
    # Draw plots for the training & validation
    visualize_training(history, "split_validation", config["evaluation_path"])
    # Make a detailed validation of the current cv-fold
    detailed_validation(model, testing, "complete", config)

#-----------------------------------------------------#
#                    Leave-one-out                    #
#-----------------------------------------------------#
def leave_one_out(cases, config):
    # Start leave-one-out cycling
    for i in range(config["n_loo"]):
        # Create a Convolutional Neural Network model
        model = MIScnn_NN.NeuralNetwork(config)
        # Choose a random sample
        loo = cases.pop(np.random.choice(len(cases)))
        # Train the model with the remaining cases
        model.train(cases)
        # Make a detailed validation on the LOO sample
        detailed_validation(model, [loo], str(loo), config)

#-----------------------------------------------------#
#                 Detailed Validation                 #
#-----------------------------------------------------#
def detailed_validation(model, cases, suffix, config):
    # Initialize kits19 scoring file
    save_evaluation(["case_id", "score_KidneyTumor", "score_Tumor"],
                    config["evaluation_path"],
                    "kits19_scoring." + suffix + ".tsv",
                    start=True)
    # Predict the cases with the provided model
    model.predict(cases)
    # Iterate over each case
    for id in cases:
        # Load the truth segmentation
        truth = load_segmentation_nii(id, config["data_path"]).get_data()
        # Load the prediction segmentation
        pred = load_prediction_nii(id, config["output_path"]).get_data()
        # Calculate kits19 score
        score_kidney, score_tumor = kits19_score(truth, pred)
        # Save kits19 score to file
        save_evaluation([id, score_kidney, score_tumor],
                        config["evaluation_path"],
                        "kits19_scoring." + suffix + ".tsv")
        # Calculate class frequency per slice
        if config["class_freq"]:
            class_freq = calc_ClassFrequency(truth, pred)
            for i in range(len(class_freq)):
                print(str(id) + "\t" + str(i) + "\t" + str(class_freq[i]))
        # Visualize the truth and prediction segmentation
        if config["visualize"]:
            # Load the volume
            vol = load_volume_nii(id, config["data_path"]).get_data()
            # Run visualization
            visualize_evaluation(id, vol, truth, pred, config["evaluation_path"])

#-----------------------------------------------------#
#              Other evaluation functions             #
#-----------------------------------------------------#
# Compute the score for the Kidney Tumor Segmentation Challenge 2019
def kits19_score(truth, pred):
    # Compute tumor+kidney Dice
    try:
        tk_pd = np.greater(pred, 0)
        tk_gt = np.greater(truth, 0)
        tk_dice = 2*np.logical_and(tk_pd, tk_gt).sum()/(
            tk_pd.sum() + tk_gt.sum()
        )
    except ZeroDivisionError:
        return 0.0, 0.0
    # Compute tumor Dice
    try:
        tu_pd = np.greater(pred, 1)
        tu_gt = np.greater(truth, 1)
        tu_dice = 2*np.logical_and(tu_pd, tu_gt).sum()/(
            tu_pd.sum() + tu_gt.sum()
        )
    except ZeroDivisionError:
        return tk_dice, 0.0
    # Return both scores
    return tk_dice, tu_dice

# Calculate the Class Frequency for each slice in x-axis
def calc_ClassFrequency(truth, pred):
    finalList = []
    # Iterate over each slice
    for slice in range(len(truth)):
        # Compute the frequency tables
        t_unique, t_counts = np.unique(truth[slice], return_counts=True)
        p_unique, p_counts = np.unique(pred[slice], return_counts=True)
        slice_cache = dict()
        # Add the freqcuencies to a directory
        for i in range(len(t_unique)):
            slice_cache["truth-" + str(t_unique[i])] = t_counts[i]
            if i < len(p_unique):
                slice_cache["pred-" + str(p_unique[i])] = p_counts[i]
        # Backup frequency directory for this slice
        finalList.append(slice_cache)
    # Return list of frequency directories for each slice
    return finalList
