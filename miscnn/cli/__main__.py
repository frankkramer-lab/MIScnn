#==============================================================================#
#  Author:       Philip Meyer                                                  #
#  Copyright:    2021 IT-Infrastructure for Translational Medical Research,    #
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


import argparse
import pathlib
import os
import os.path
from miscnn import Data_IO
import pandas as pd
import numpy as np
from tqdm import tqdm
from miscnn.data_loading.interfaces import NIFTI_interface
from miscnn.data_loading.interfaces import Image_interface
from miscnn.data_loading.interfaces import DICOM_interface

parser = argparse.ArgumentParser(description='MIScnn CLI')
subparsers = parser.add_subparsers(help='Components')
parser.add_argument('-v', dest="verbose", action='store_true', default=False,
                    help='provide verbose output', required=False)
parser.add_argument('--data_dir', dest="data_dir", type=str, default="./data",
                    help='set path to data dir', required=False)

parser.set_defaults(which='null')
#verification
verification_parser = subparsers.add_parser("verify")
verification_parser.set_defaults(which='verify')
verification_parser.add_argument("-t", "--type", dest="imagetype", choices=["NIFTI", "DICOM", "IMG"], help="The method of medical image storage in the datapath", required=True)
#verification_parser.add_argument('-b', "--batches", dest="batches", action='store_true', default=False, help='check if loading of batches works. Provide list of existent seeds.', required=False)

#implement cleaning
cleanup_parser = subparsers.add_parser("cleanup")
cleanup_parser.set_defaults(which='cleanup')
cleanup_parser.add_argument("-b", "--batches", dest="batches", action='store_true', default=False, help="Cleanup batch directory", required=False)
cleanup_parser.add_argument("-e", "--evaluation", dest="eval", action='store_true', default=False, help="Cleanup evaluation directory", required=False)
cleanup_parser.add_argument("-p", "--prediction", dest="pred", action='store_true', default=False, help="Cleanup prediction directory", required=False)

#data exploration subparser
data_exp_parser = subparsers.add_parser("data_exp")
data_exp_parser.set_defaults(which='data_exp')
data_exp_parser.add_argument("-t", "--type", dest="imagetype", choices=["NIFTI", "DICOM", "IMG", "Unknown"], default="Unknown", help="The method of medical image storage in the datapath", required=False)
data_exp_parser.add_argument('-cn', "--counts", dest="counts", action='store_true', default=False, help='count data provided', required=False)
data_exp_parser.add_argument('-cl', "--classes", dest="classes", action='store_true', default=False, help='count data provided', required=False)
data_exp_parser.add_argument('-mem', "--memory", dest="memory", action='store_true', default=False, help='compute memory cost', required=False)
data_exp_parser.add_argument('-s', "--structure", dest="structure", action='store_true', default=False, help='scan data for its structure', required=False)
data_exp_parser.add_argument('-m', "--minmax", dest="minmax", action='store_true', default=False, help='compute range per data element and overall range.', required=False)
data_exp_parser.add_argument('-mS', "--minmax_seg", dest="minmax_seg", action='store_true', default=False, help='compute range per data class and overall range.', required=False)
data_exp_parser.add_argument('-r', "--ratio", dest="ratio", action='store_true', default=False, help='Ratios between the segmentation classes provided.', required=False)
data_exp_parser.add_argument("-e", "--export", dest="export", type=str, default="", help="If and there the data should be exported to", required=False)
data_exp_parser.add_argument("-b", "--binning", dest="binning", type=int, default=0, help="compute N bins over the dataset and get variation over the images.", required=False)
data_exp_parser.add_argument("-bS", "--binning_seg", dest="binning_seg", type=int, default=0, help="compute N bins over the dataset per class and get variation over the images.", required=False)

#TODO add config code
#TODO add conversion
#TODO add visualization

args = parser.parse_args()

def del_tree(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    

if (args.which == "verify"):
    interface = None
    if (args.imagetype == "NIFTI"):
        interface = NIFTI_interface()
        print("using NIFTI_interface")
    elif (args.imagetype == "DICOM"):
        interface = DICOM_interface()
        print("using DICOM_interface")
    elif (args.imagetype == "IMG"):
        interface = Image_interface()
        print("using Image_interface")
    dataio = Data_IO(interface, args.data_dir)
    indices = dataio.get_indiceslist()
    if len(indices) == 0: #or maybe lower than a threshold
        print("[WARNING] Datapath " + str(args.data_path) + " does not seem to contain any samples.")
    for index in indices:
        try:
            sample = dataio.sample_loader(index, load_seg=False)
        except:
            print("[ERROR] Sample image with index " + index + " failed to load using the " + args.imagetype + " interface.")
        try:
            sample = dataio.sample_loader(index, load_seg=True)
        except:
            print("[WARNING] Sample segmentation with index " + index + " failed to load using the " + args.imagetype + " interface.")
elif (args.which == "cleanup"):
    if (args.batches):
        del_tree(args.data_dir + "/batches")
    if (args.eval):
        del_tree(args.data_dir + "/evaluation")
    if (args.pred):
        del_tree(args.data_dir + "/prediction")
elif (args.which == "data_exp"):
    interface = None
    data_dir = str(args.data_dir)
    if (args.imagetype == "Unknown"):
        files = [f[f.find("."):] for dp, dn, filenames in os.walk(data_dir) for f in filenames if os.path.isfile(os.path.join(dp, f)) and ("imaging" in f or "segmentation" in f)]
        unique = list(np.unique(np.asarray(files)))
        unique = [u for u in unique if u in [".nii", ".nii.gz", ".dcm", ".png"]]
        if len(unique) > 1:
            print("Failed to infer image type")
            exit()
        if (unique[0] == ".png"):
            interface = Image_interface()
            print("Inferred Image_interface")
        elif (unique[0] == ".dcm"):
            interface = DICOM_interface()
            print("Inferred DICOM_interface")
        else:
            interface = NIFTI_interface()
            print("Inferred NIFTI_interface")
    elif (args.imagetype == "NIFTI"):
        interface = NIFTI_interface()
        print("using NIFTI_interface")
    elif (args.imagetype == "DICOM"):
        interface = DICOM_interface()
        print("using DICOM_interface")
    elif (args.imagetype == "IMG"):
        interface = Image_interface()
        print("using Image_interface")
    dataio = Data_IO(interface, args.data_dir)
    
    indices = dataio.get_indiceslist()
    cnt = len(indices)
    print("interface found " + str(cnt) + " indices in the data directory.")
    
    images = [index for index in indices if os.path.exists(data_dir + "/" + index + "/imaging.nii.gz") or os.path.exists(data_dir + "/" + index + "/imaging.dcm") or os.path.exists(data_dir + "/" + index + "/imaging.png")]
    segmentations = [index for index in indices if os.path.exists(data_dir + "/" + index + "/segmentation.nii.gz") or os.path.exists(data_dir + "/" + index + "/segmentation.dcm") or os.path.exists(data_dir + "/" + index + "/segmentation.png")]
    
    global_min = 999999
    global_max = -99999

    #this is the values that descibe the value space shared by all data objects.
    shared_min = 999999
    shared_max = -99999
    
    class_set = []
    
    class_minmax = {}
    
    if (args.counts):
        print("Found " + str(cnt) + " samples.")
        print("In Samples found " + str(len(images)) + " images.")
        print("In Samples found " + str(len(segmentations)) + " image segmentations.")
    df = pd.DataFrame()
    df["name"] = indices
    if (args.classes or args.minmax_seg or args.binning_seg > 0):
        print("computing classes of all samples. This is implied with segmentation class analysis.")
        show_warnings = False
        for index in tqdm(segmentations):
            sample = dataio.sample_loader(index, load_seg=True)
            classes = np.unique(sample.seg_data)
            
            if (len(class_set)):
                class_set = list(classes)
            for c in classes:
                if (not c in class_set):
                    if (show_warnings):
                        print("Warning: not all classes occur in every file.")
                    class_set.append(c)
            
            show_warnings = True
        print("The total count of classes over all segmentations is " + str(len(class_set)))
        if (len(class_set) < 2):
            print("Warning detected a class count smaller than 2. This dataset is likely damaged.")
        class_cnt = len(class_set)
    if (args.memory):
        print("collecting memory information of data directory.")
        imagesize = 0
        max_img = 0
        min_img = 99999999999
        segsize = 0
        max_seg = 0
        min_seg = 99999999999
        sample_data = {}
        for index in tqdm(indices):
            if (not os.path.isdir(data_dir + "/" + index)):
                continue
            sample_data[index] = [index]
            size = os.path.getsize(data_dir + "/" + index + "/imaging.nii.gz")
            max_img = max(size, max_img)
            min_img = min(size, min_img)
            sample_data[index].append(size)
            imagesize += size
            if (os.path.exists(data_dir + "/" + index + "/segmentation.nii.gz")):
                size = os.path.getsize(data_dir + "/" + index + "/segmentation.nii.gz")
                sample_data[index].append(size)
                max_seg = max(size, max_seg)
                min_seg = min(size, min_seg)
                segsize += size
            else:
                sample_data[index].append(0)
        print("total datasize of images is " + str(imagesize) + " which averages to " + str(imagesize / images) + " bytes of data per image.")
        print("total datasize of segmentations is " + str(segsize) + " which averages to " + str(segsize / segmentations) + " bytes of data per segmentation.")
        print("the minimum value in the data images is " + str(min_img))
        print("the maximum value in the data images is " + str(max_img))
        print("the minimum value in the data segmentations is " + str(min_seg))
        print("the maximum value in the data segmentations is " + str(max_seg))
        df = df.merge(pd.DataFrame.from_dict(sample_data, orient="index",columns=["name", "image_size", "segmentation_size"]), on="name", how="right")
        #TODO compute model size, evaluation and prediction memory cost. as well as batches
    if (args.structure):
        sample_data = {}
        print("collecting structure data.")
        for index in tqdm(indices):
            if (not os.path.isdir(data_dir + "/" + index)):
                continue
            # Sample loading
            sample = dataio.sample_loader(index, load_seg=False)
            # Create an empty list for the current asmple in our data dictionary
            sample_data[index] = [index]
            sample_data[index].append(sample.img_data.shape)
            if ("spacing" in sample.get_extended_data()):
                sample_data[index].append(sample.get_extended_data()["spacing"])
            elif ("affine" in sample.get_extended_data()):
            
                spacing_matrix = sample.get_extended_data()["affine"][:3,:3]
                # Identify correct spacing diagonal
                diagonal_negative = np.diag(spacing_matrix)
                diagonal_positive = np.diag(spacing_matrix[::-1,:])
                if np.count_nonzero(diagonal_negative) != 1:
                    spacing = diagonal_negative
                elif np.count_nonzero(diagonal_positive) != 1:
                    spacing = diagonal_positive
                else:
                    warnings.warn("Affinity matrix of NIfTI volume can not be parsed.")
                # Calculate absolute values for voxel spacing
                spacing = np.absolute(spacing)
                sample_data[index].append(spacing)
            else:
                sample_data[index].append(None)
            # Identify and store class distribution
        df = df.merge(pd.DataFrame.from_dict(sample_data, orient="index",columns=["name", "shape", "voxel_spacing"]), on="name", how="right")
    if (args.minmax or args.binning > 0):
        sample_data = {}
        print("finding minima and maxima of the data images")
        for index in tqdm(indices):
            if (not os.path.isdir(data_dir + "/" + index)):
                continue
            # Sample loading
            sample = dataio.sample_loader(index, load_seg=False)
            # Create an empty list for the current asmple in our data dictionary
            sample_data[index] = [index]
            # Identify minimum and maximum volume intensity
            min_val = sample.img_data.min()
            max_val = sample.img_data.max()
            global_min = min(global_min, min_val)
            global_max = max(global_max, max_val)
            if (shared_min == 999999):
                shared_min = min_val
            else:
                shared_min = max(shared_min, min_val)
            if (shared_max == -99999):
                shared_max = max_val
            else:
                shared_max = min(shared_max, min_val)
            sample_data[index].append(min_val)
            sample_data[index].append(max_val)
        print("the global value space is (", str(global_min) + ", " + str(global_max) + ")")
        print("the shared value space is (", str(shared_min) + ", " + str(shared_max) + ")")
        df = df.merge(pd.DataFrame.from_dict(sample_data, orient="index",columns=["name", "minimum", "maximum"]), on="name", how="right")
    if (args.minmax_seg or args.binning_seg > 0):
        sample_data = {}
        print("finding minima and maxima of each segmentation class the data images")
        
        for c in class_set:
            class_minmax[c] = [999999, -99999, 999999, -99999]
        
        for index in tqdm(segmentations):
            # Sample loading
            sample = dataio.sample_loader(index, load_seg=True)
            # Create an empty list for the current asmple in our data dictionary
            
            sample_data[index] = [index]
            for c in class_set:
                # Identify minimum and maximum volume intensity
                data = np.ma.MaskedArray(sample.img_data, sample.seg_data.astype(np.uint8) != np.uint8(c))
                min_val = data.min()
                max_val = data.max()
                class_minmax[c][0] = min(class_minmax[c][0], min_val)
                class_minmax[c][1] = max(class_minmax[c][1], max_val)
                if (class_minmax[c][2] == 999999):
                    class_minmax[c][2] = min_val
                else:
                    class_minmax[c][2] = max(class_minmax[c][2], min_val)
                if (class_minmax[c][3] == -99999):
                    class_minmax[c][3] = max_val
                else:
                    class_minmax[c][3] = min(class_minmax[c][3], max_val)
                sample_data[index].append(min_val)
                sample_data[index].append(max_val)
        for c in class_set:
            print("the global value space for class " + str(c) + " is (", str(class_minmax[c][0]) + ", " + str(class_minmax[c][1]) + ")")
            print("the shared value space for class " + str(c) + " is (", str(class_minmax[c][2]) + ", " + str(class_minmax[c][3]) + ")")
        
        columns = ["name"]
        for c in class_set:
            columns.append("minimum_c" + str(c))
            columns.append("maximum_c" + str(c))
        
        df = df.merge(pd.DataFrame.from_dict(sample_data, orient="index",columns=columns), on="name", how="right")
    if (args.ratio):
        sample_data = {}
        print("computing class ratios.")
        for index in tqdm(indices):
            if (not os.path.isdir(data_dir + "/" + index)):
                continue
            has_seg = os.path.exists(data_dir + "/" + index + "/segmentation.nii.gz")
            if (not has_seg):
                sample_data[index] = [index]
                continue
            # Sample loading
            sample = dataio.sample_loader(index, load_seg=True)
            # Create an empty list for the current asmple in our data dictionary
            sample_data[index] = [index]
            # Store voxel spacing
            unique_data, unique_counts = np.unique(sample.seg_data, return_counts=True)
            class_freq = unique_counts / np.sum(unique_counts)
            class_freq = np.around(class_freq, decimals=6)
            sample_data[index].append(tuple(class_freq))
        df = df.merge(pd.DataFrame.from_dict(sample_data, orient="index",columns=["name", "class_frequency"]), on="name", how="right")
    if (args.binning > 0):
        threshhold = []
        ratio = (shared_max - shared_min) / args.binning
        threshhold.append(shared_min)
        for i in range(args.binning):
            threshhold.append(shared_min + ratio * (1 + i))
        threshhold.append(global_max + 1)
        print(threshhold)
        sample_data = {}
        for index in tqdm(indices):
            if (not os.path.isdir(data_dir + "/" + index)):
                continue
            sample = dataio.sample_loader(index, load_seg=False)
            # Create an empty list for the current asmple in our data dictionary
            sample_data[index] = [index]
            # Identify minimum and maximum volume intensity
            sample_data[index].append((threshhold[0] < sample.img_data).sum())
            for i in range(args.binning + 1):
                sample_data[index].append(((threshhold[i] <= sample.img_data) & (threshhold[i + 1] > sample.img_data)).sum())
        
        df = df.merge(pd.DataFrame.from_dict(sample_data, orient="index",columns=["name"] + ["bin"+str(i) for i in range(len(threshhold))]), on="name", how="right")
    if (args.binning_seg > 0):
        print("fitting bins for segmentation classes.")
        threshholds = {}
        for c in class_set:
            threshhold = []
            ratio = (class_minmax[c][3] - class_minmax[c][2]) / args.binning_seg
            threshhold.append(class_minmax[c][2])
            for i in range(args.binning_seg):
                threshhold.append(class_minmax[c][2] + ratio * (1 + i))
            threshhold.append(class_minmax[c][1] + 1)
            
            threshholds[c] = threshhold
            print("The threshholds for class " + str(c) + " are: " + str(threshhold))
        
        print("Computing bins for segmentation classes.")
        sample_data = {}
        for index in tqdm(segmentations):
            sample = dataio.sample_loader(index, load_seg=True)
            # Create an empty list for the current asmple in our data dictionary
            sample_data[index] = [index]
            
            for c in class_set:
                # Identify minimum and maximum volume intensity
                sample_data[index].append(((threshholds[c][0] < sample.img_data) & (sample.seg_data.astype(np.uint8) == np.uint8(c))).sum())
                for i in range(args.binning_seg + 1):
                    sample_data[index].append(((threshholds[c][i] <= sample.img_data) & (threshholds[c][i + 1] > sample.img_data) & (sample.seg_data.astype(np.uint8) == np.uint8(c))).sum())
        
        indexes = ["name"]
        for c in class_set:
            for i in range(len(threshhold)):
                indexes.append("class" + str(c) + "_bin"+str(i))
        
        df = df.merge(pd.DataFrame.from_dict(sample_data, orient="index",columns=indexes), on="name", how="right")
    
    if (len(args.export) > 0):
        df.to_csv(args.export)
    else:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
        
#check if z-score observes normal distribution. calculate transformation to normal distribution