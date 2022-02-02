#==============================================================================#
#  Author:       Philip Meyer                                                #
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

import pathlib
import os
import os.path
import pandas as pd
import numpy as np
from tqdm import tqdm
from miscnn import Data_IO
from miscnn.data_loading.interfaces import miscnn_data_interfaces, get_data_interface_from_file_term

def register_commands(parser):
    parser.set_defaults(which='data_exp')
    parser.add_argument("-t", "--type", dest="imagetype", choices=["NIFTI", "DICOM", "IMG", "Unknown"], default="Unknown", help="The method of medical image storage in the datapath", required=False)
    parser.add_argument('-cn', "--counts", dest="counts", action='store_true', default=False, help='count data provided', required=False)
    parser.add_argument('-cl', "--classes", dest="classes", action='store_true', default=False, help='count data provided', required=False)
    parser.add_argument('-mem', "--memory", dest="memory", action='store_true', default=False, help='compute memory cost', required=False)
    parser.add_argument('-s', "--structure", dest="structure", action='store_true', default=False, help='scan data for its structure', required=False)
    parser.add_argument('-m', "--minmax", dest="minmax", action='store_true', default=False, help='compute range per data element and overall range.', required=False)
    parser.add_argument('-mS', "--minmax_seg", dest="minmax_seg", action='store_true', default=False, help='compute range per data class and overall range.', required=False)
    parser.add_argument('-r', "--ratio", dest="ratio", action='store_true', default=False, help='Ratios between the segmentation classes provided.', required=False)
    parser.add_argument("-e", "--export", dest="export", type=str, default="", help="If and there the data should be exported to", required=False)
    parser.add_argument("-b", "--binning", dest="binning", type=int, default=0, help="compute N bins over the dataset and get variation over the images.", required=False)
    parser.add_argument("-bS", "--binning_seg", dest="binning_seg", type=int, default=0, help="compute N bins over the dataset per class and get variation over the images.", required=False)

def setup_execution(args):
    data_dir = str(args.data_dir)
    interface = None
    if (args.imagetype in miscnn_data_interfaces.keys()):
        interface = miscnn_data_interfaces[args.imagetype]()
    else:
        files = [f[f.find("."):] for dp, dn, filenames in os.walk(data_dir) for f in filenames if os.path.isfile(os.path.join(dp, f)) and ("imaging" in f or "segmentation" in f)]
        unique = list(np.unique(np.asarray(files)))
        unique = [get_data_interface_from_file_term(u) for u in unique]
        if len(unique) != 1:
            raise RuntimeError("Failed to infer image type")
        if (None in unique):
            raise RuntimeError("Failed to infer image type")
        interface = unique[0]()
        
    
    dataio = Data_IO(interface, args.data_dir)
    
    indices = dataio.get_indiceslist()
    cnt = len(indices)
    print("interface found " + str(cnt) + " indices in the data directory.")

    images = [index for index in indices if os.path.exists(data_dir + "/" + index + "/imaging.nii.gz") or os.path.exists(data_dir + "/" + index + "/imaging.dcm") or os.path.exists(data_dir + "/" + index + "/imaging.png")]
    segmentations = [index for index in indices if os.path.exists(data_dir + "/" + index + "/segmentation.nii.gz") or os.path.exists(data_dir + "/" + index + "/segmentation.dcm") or os.path.exists(data_dir + "/" + index + "/segmentation.png")]
    
    return {"dataio": dataio, "indices": indices, "cnt": cnt, "images": images, "segmentations": segmentations, "data_dir":data_dir}


def execute_counts(context):
    print("Found " + str(context["cnt"]) + " samples.")
    print("In Samples found " + str(len(context["images"])) + " images.")
    print("In Samples found " + str(len(context["segmentations"])) + " image segmentations.")

def execute_class_analysis(context):
    print("computing classes of all samples. This is implied with segmentation class analysis.")
    class_set = []
    show_warnings = False
    for index in tqdm(context["segmentations"]):
        sample = context["dataio"].sample_loader(index, load_seg=True)
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
    return class_set

#this checks filesize not datasize
def execute_memory_analysis(context, df):
    print("collecting memory information of data directory.")
    imagesize = 0
    max_img = 0
    min_img = 99999999999
    segsize = 0
    max_seg = 0
    min_seg = 99999999999
    sample_data = {}
    for index in tqdm(context["indices"]):
        if (not os.path.isdir(context["data_dir"] + "/" + index)):
            continue
        sample_data[index] = [index]
        size = os.path.getsize(context["data_dir"] + "/" + index + "/imaging.nii.gz")
        max_img = max(size, max_img)
        min_img = min(size, min_img)
        sample_data[index].append(size)
        imagesize += size
        if (os.path.exists(context["data_dir"] + "/" + index + "/segmentation.nii.gz")):
            size = os.path.getsize(context["data_dir"] + "/" + index + "/segmentation.nii.gz")
            sample_data[index].append(size)
            max_seg = max(size, max_seg)
            min_seg = min(size, min_seg)
            segsize += size
        else:
            sample_data[index].append(0)
    print("total datasize of images is " + str(imagesize) + " which averages to " + str(imagesize / len(context["images"])) + " bytes of data per image.")
    print("total datasize of segmentations is " + str(segsize) + " which averages to " + str(segsize / len(context["segmentations"])) + " bytes of data per segmentation.")
    print("the minimum size for the data images is " + str(min_img))
    print("the maximum size for the data images is " + str(max_img))
    print("the minimum size for the data segmentations is " + str(min_seg))
    print("the maximum size for the data segmentations is " + str(max_seg))
    return df.merge(pd.DataFrame.from_dict(sample_data, orient="index",columns=["name", "image_size", "segmentation_size"]), on="name", how="right")
    #TODO compute model size, evaluation and prediction memory cost. as well as batches

def execute_structure_analysis(context, df):
    sample_data = {}
    print("collecting structure data.")
    
    for index in tqdm(context["indices"]):
        if (not os.path.isdir(context["data_dir"] + "/" + index)):
            continue
        # Sample loading
        sample = context["dataio"].sample_loader(index, load_seg=False)
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
    return df.merge(pd.DataFrame.from_dict(sample_data, orient="index",columns=["name", "shape", "voxel_spacing"]), on="name", how="right")

def execute_minmax_analysis(context, df):
    sample_data = {}
    print("finding minima and maxima of the data images")
    
    global_min = 999999
    global_max = -99999
    shared_min = 999999
    shared_max = -99999
    
    for index in tqdm(context["indices"]):
        if (not os.path.isdir(context["data_dir"] + "/" + index)):
            continue
        # Sample loading
        sample = context["dataio"].sample_loader(index, load_seg=False)
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
            shared_max = min(shared_max, max_val)
        sample_data[index].append(min_val)
        sample_data[index].append(max_val)
    print("the global value space is (", str(global_min) + ", " + str(global_max) + ")")
    print("the shared value space is (", str(shared_min) + ", " + str(shared_max) + ")")
    df = df.merge(pd.DataFrame.from_dict(sample_data, orient="index",columns=["name", "minimum", "maximum"]), on="name", how="right")
    
    return df, {"glob_min" : global_min, "glob_max" : global_max, "shr_min" : shared_min, "shr_max" : shared_max}

def execute_minmax_seg_analysis(context, df, class_data = None):
    if class_data is None:
        class_data = execute_class_analysis(context)    
    sample_data = {}
    print("finding minima and maxima of each segmentation class the data images")
    
    class_minmax = {}
    
    for c in class_data:
        class_minmax[c] = {"glob_min" : 999999, "glob_max" : -99999, "shr_min" : 999999, "shr_max" : -99999}
    
    for index in tqdm(context["segmentations"]):
        # Sample loading
        sample = context["dataio"].sample_loader(index, load_seg=True)
        # Create an empty list for the current asmple in our data dictionary
        
        sample_data[index] = [index]
        for c in class_data:
            # Identify minimum and maximum volume intensity
            data = np.ma.MaskedArray(sample.img_data, sample.seg_data.astype(np.uint8) != np.uint8(c))
            min_val = data.min()
            max_val = data.max()
            class_minmax[c]["glob_min"] = min(class_minmax[c]["glob_min"], min_val)
            class_minmax[c]["glob_max"] = max(class_minmax[c]["glob_max"], max_val)
            if (class_minmax[c]["shr_min"] == 999999):
                class_minmax[c]["shr_min"] = min_val
            else:
                class_minmax[c]["shr_min"] = max(class_minmax[c]["shr_min"], min_val)
            if (class_minmax[c]["shr_max"] == -99999):
                class_minmax[c]["shr_max"] = max_val
            else:
                class_minmax[c]["shr_max"] = min(class_minmax[c]["shr_max"], max_val)
            sample_data[index].append(min_val)
            sample_data[index].append(max_val)
    for c in class_data:
        print("the global value space for class " + str(c) + " is (", str(class_minmax[c]["glob_min"]) + ", " + str(class_minmax[c]["glob_max"]) + ")")
        print("the shared value space for class " + str(c) + " is (", str(class_minmax[c]["shr_min"]) + ", " + str(class_minmax[c]["shr_max"]) + ")")
    
    columns = ["name"]
    for c in class_data:
        columns.append("minimum_c" + str(c))
        columns.append("maximum_c" + str(c))
    
    df = df.merge(pd.DataFrame.from_dict(sample_data, orient="index",columns=columns), on="name", how="right")
    return df, class_minmax
    
def execute_ratio_analysis(context, df):
    sample_data = {}
    print("computing class ratios.")
    for index in tqdm(context["indices"]):
        if (not os.path.isdir(context["data_dir"] + "/" + index)):
            continue
        has_seg = os.path.exists(context["data_dir"] + "/" + index + "/segmentation.nii.gz")
        if (not has_seg):
            sample_data[index] = [index]
            continue
        # Sample loading
        sample = context["dataio"].sample_loader(index, load_seg=True)
        # Create an empty list for the current asmple in our data dictionary
        sample_data[index] = [index]
        # Store voxel spacing
        unique_data, unique_counts = np.unique(sample.seg_data, return_counts=True)
        class_freq = unique_counts / np.sum(unique_counts)
        class_freq = np.around(class_freq, decimals=6)
        sample_data[index].append(tuple(class_freq))
    df = df.merge(pd.DataFrame.from_dict(sample_data, orient="index",columns=["name", "class_frequency"]), on="name", how="right")
    return df

def execute_binning(context, df, binning, minmax_data = None):
    if minmax_data is None:
        df, minmax_data = execute_minmax_analysis(context, df)
    
    threshhold = []
    ratio = (minmax_data["shr_max"] - minmax_data["shr_min"]) / binning
    threshhold.append(minmax_data["shr_min"])
    for i in range(binning):
        threshhold.append(minmax_data["shr_min"] + ratio * (1 + i))
    threshhold.append(minmax_data["glob_max"] + 1)
    print("The threshholds are: " + str(threshhold))
    sample_data = {}
    for index in tqdm(context["indices"]):
        if (not os.path.isdir(context["data_dir"] + "/" + index)):
            continue
        sample = context["dataio"].sample_loader(index, load_seg=False)
        # Create an empty list for the current asmple in our data dictionary
        sample_data[index] = [index]
        # Identify minimum and maximum volume intensity
        sample_data[index].append((threshhold[0] < sample.img_data).sum())
        for i in range(binning + 1):
            sample_data[index].append(((threshhold[i] <= sample.img_data) & (threshhold[i + 1] > sample.img_data)).sum())
    
    df = df.merge(pd.DataFrame.from_dict(sample_data, orient="index",columns=["name"] + ["bin"+str(i) for i in range(len(threshhold))]), on="name", how="right")
    return df

def execute_binning_seg(context, df, binning_seg, class_data = None, minmax_data = None):
    if class_data is None:
        class_data = execute_class_analysis(context)
    
    if minmax_data is None:
        df, minmax_data = execute_minmax_seg_analysis(context, df, class_data)
    
    print("fitting bins for segmentation classes.")
    threshholds = {}
    for c in class_data:
        threshhold = []
        ratio = (minmax_data[c]["shr_max"] - minmax_data[c]["shr_min"]) / binning_seg
        threshhold.append(minmax_data[c]["shr_min"])
        for i in range(binning_seg):
            threshhold.append(minmax_data[c]["shr_min"] + ratio * (1 + i))
        threshhold.append(minmax_data[c]["glob_max"] + 1)
        
        threshholds[c] = threshhold
        print("The threshholds for class " + str(c) + " are: " + str(threshhold))
    
    print("Computing bins for segmentation classes.")
    sample_data = {}
    for index in tqdm(context["segmentations"]):
        sample = context["dataio"].sample_loader(index, load_seg=True)
        # Create an empty list for the current asmple in our data dictionary
        sample_data[index] = [index]
        
        for c in class_data:
            # Identify minimum and maximum volume intensity
            sample_data[index].append(((threshholds[c][0] < sample.img_data) & (sample.seg_data.astype(np.uint8) == np.uint8(c))).sum())
            for i in range(binning_seg + 1):
                sample_data[index].append(((threshholds[c][i] <= sample.img_data) & (threshholds[c][i + 1] > sample.img_data) & (sample.seg_data.astype(np.uint8) == np.uint8(c))).sum())
    
    indexes = ["name"]
    for c in class_data:
        for i in range(len(threshhold)):
            indexes.append("class" + str(c) + "_bin"+str(i))
    
    df = df.merge(pd.DataFrame.from_dict(sample_data, orient="index",columns=indexes), on="name", how="right")
    return df

def execute(args):
    
    context = setup_execution(args)
    
    if (args.counts):
        execute_counts(context)
    
    df = pd.DataFrame()
    df["name"] = context["indices"]
    
    class_data = None
    minmax_data = None
    minmax_seg_data = None
    
    if (args.classes):
        class_data = execute_class_analysis(context)
    if (args.memory):
        df = execute_memory_analysis(context, df)
    if (args.structure):
        df = execute_structure_analysis(context, df)
    if (args.minmax):
        df, minmax_data = execute_minmax_analysis(context, df)
    if (args.minmax_seg):
        df, minmax_seg_data = execute_minmax_seg_analysis(context, df, class_data)
    
    if (args.ratio):
        df = execute_ratio_analysis(context, df)
    if (args.binning > 0):
        df = execute_binning(context, df, args.binning, minmax_data)
    if (args.binning_seg > 0):
        df = execute_binning_seg(context, df, args.binning_seg, class_data, minmax_seg_data)
    
    if (len(args.export) > 0):
        df.to_csv(args.export)
    else:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
