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
import os
import glob
import nibabel as nib
import re
import numpy as np
import warnings
import pydicom
import SimpleITK as sitk
from skimage import draw
# Internal libraries/scripts
from miscnn.data_loading.interfaces.abstract_io import Abstract_IO

#-----------------------------------------------------#
#                 DICOM I/O Interface                 #
#-----------------------------------------------------#
""" Data I/O Interface for DICOM files."""

class DICOM_interface(Abstract_IO):
    # Class variable initialization
    def __init__(self, channels=1, classes=2, three_dim=True, mask_background = 0, structure_dict = {}):
        self.data_directory = None
        self.channels = channels
        self.classes = classes
        self.three_dim = three_dim
        self.cache = dict()
        self.mask_background = mask_background
        self.structure_dict = structure_dict

    #---------------------------------------------#
    #                  initialize                 #
    #---------------------------------------------#
    # Initialize the interface and return number of samples
    def initialize(self, input_path):
        # Resolve location where imaging data should be living
        if not os.path.exists(input_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(str(input_path))
            )
        # Cache data directory
        self.data_directory = input_path
        # Identify samples
        sample_list = os.listdir(input_path)
        # Return sample list
        return sample_list

    #---------------------------------------------#
    #                  load_image                 #
    #---------------------------------------------#
    # Read a volume NIFTI file from the data directory
    def load_image(self, index):

        dicom_images_path = None

        # Make sure that the image file exists in the data set directory
        img_path = os.path.join(self.data_directory, index)

        #search folders and find the first DICOM image of the whole scan to get the path of the DICOM images
        for root, _, files in os.walk(img_path):
            for _ in filter(lambda x: re.findall('1-001.dcm', x), files):
                dicom_images_path = root
        
        if not os.path.exists(dicom_images_path):
            raise ValueError(
                "No DICOM scans could not be found \"{}\"".format(img_path)
            )

        #create a ImageSeriesReader Object
        dicom_reader = sitk.ImageSeriesReader()

        #read all DICOM files contained in the provided folder
        dicom_file_names = dicom_reader.GetGDCMSeriesFileNames(str(dicom_images_path))

        #set file names
        dicom_reader.SetFileNames(dicom_file_names)

        #read dicom files
        dcm = dicom_reader.Execute()

        #Read one of the images to get DICOM Pixelspacing and SliceThickness paramters
        dcm_temp = pydicom.read_file(dicom_file_names[0])

        self.cache[index] = np.array([dcm_temp.SliceThickness, dcm_temp.PixelSpacing[0], dcm_temp.PixelSpacing[1]], dtype=np.float32) 


        #return both the sitk image and all images as a numpy array
        return sitk.GetArrayFromImage(dcm)

    
    def load_image_sitk(self, index):

        dicom_images_path = None

        # Make sure that the image file exists in the data set directory
        img_path = os.path.join(self.data_directory, index)

        #search folders and find the first DICOM image of the whole scan to get the path of the DICOM images
        for root, _, files in os.walk(img_path):
            for _ in filter(lambda x: re.findall('1-001.dcm', x), files):
                dicom_images_path = root

        if not os.path.exists(dicom_images_path):
            raise ValueError(
                "No DICOM scans could not be found \"{}\"".format(img_path)
            )

        #create a ImageSeriesReader Object
        dicom_reader = sitk.ImageSeriesReader()

        #read all DICOM files contained in the provided folder
        dicom_file_names = dicom_reader.GetGDCMSeriesFileNames(str(dicom_images_path))

        #set file names
        dicom_reader.SetFileNames(dicom_file_names)

        #read dicom files
        dcm = dicom_reader.Execute()

        #Read one of the images to get DICOM Pixelspacing and SliceThickness paramters
        dcm_temp = pydicom.read_file(dicom_file_names[0])

        self.cache[index] = np.array([dcm_temp.SliceThickness, dcm_temp.PixelSpacing[0], dcm_temp.PixelSpacing[1]], dtype=np.float32) 


        #return both the sitk image and all images as a numpy array
        return dcm


    #---------------------------------------------#
    #              load_segmentation              #
    #---------------------------------------------#
    # Read a segmentation NIFTI file from the data directory
    def load_segmentation(self, index):
        
        rtStruct_path = None

        # Make sure that the segmentation file exists in the data set directory
        seg_path = os.path.join(self.data_directory, index)

        #search folder structure for rt dcm file named 1-1.dcm
        for root, _, files in os.walk(seg_path):
            for file in filter(lambda x: re.match('1-1.dcm', x), files):
                rtStruct_path = os.path.join(root, file)

        if not os.path.exists(rtStruct_path):
            raise ValueError(
                "No structure file could not be found \"{}\"".format(rtStruct_path)
            )
        #get ROI data
        contours = self.get_ROI_data(rtStruct_path)

        images = self.load_image_sitk(index)

        segmentations = self.convert(contours, images)


        # Return DICOM structure file 
        return segmentations


    def get_ROI_data(self, rtstruct_file = None):

        ss_file = pydicom.read_file(rtstruct_file) 

        contours = []

        for sequence, metadata in zip(ss_file.ROIContourSequence, ss_file.StructureSetROISequence):
            
            contour_data = {}

            contour_data['ROI_ID'] = metadata.ROINumber
            contour_data['ROI_Name'] = metadata.ROIName
            contour_data['ROI_Sequence'] = []
            for contour in sequence.ContourSequence:
                            contour_data['ROI_Sequence'].append({
                                'type': (contour.ContourGeometricType if hasattr(contour, 'ContourGeometricType') else 'unknown'),
                                'points': {
                                    'x': ([contour.ContourData[index] for index in range(0, len(contour.ContourData), 3)] if hasattr(contour, 'ContourData') else None), 
                                    'y': ([contour.ContourData[index + 1] for index in range(0, len(contour.ContourData), 3)] if hasattr(contour, 'ContourData') else None),  
                                    'z': ([contour.ContourData[index + 2] for index in range(0, len(contour.ContourData), 3)] if hasattr(contour, 'ContourData') else None)  
                                }
                            })

            contours.append(contour_data)
        
        return contours

    #---------------------------------------------#
    #              load_ROI_names                 #
    #---------------------------------------------#
    #gets all ROIs contained in a DICOM structure file
    def get_ROI_names(self, index):

        rtStruct_path = None

        # Make sure that the segmentation file exists in the data set directory
        seg_path = os.path.join(self.data_directory, index)

        #search folder structure for rt dcm file named 1-1.dcm
        for root, _, files in os.walk(seg_path):
            for file in filter(lambda x: re.match('1-1.dcm', x), files):
                rtStruct_path = os.path.join(root, file)

        contours = self.get_ROI_data(rtStruct_path)

        structure_names = [[cont['ROI_ID'], cont['ROI_Name']] for cont in contours]

        return structure_names


    def convert(self, rtstruct_contours, dicom_image):

        shape = dicom_image.GetSize()

        mask = sitk.Image(shape, sitk.sitkUInt8)
        mask.CopyInformation(dicom_image)

        np_mask = sitk.GetArrayFromImage(mask)
        np_mask.fill(self.mask_background)      

        
        for rtstruct in rtstruct_contours:

            
            
            if rtstruct['ROI_Name'] in self.structure_dict:

                rtstruct_seq = rtstruct['ROI_Sequence']

                for seq in rtstruct_seq:

                    coordinates = seq['points']

                    pts = np.zeros([len(coordinates['x']), 3])

                    for index in range(0, len(coordinates['x'])):
                        # lets convert world coordinates to voxel coordinates
                        world_coords = dicom_image.TransformPhysicalPointToIndex((coordinates['x'][index], coordinates['y'][index], coordinates['z'][index]))
                        pts[index, 0] = world_coords[0]
                        pts[index, 1] = world_coords[1]
                        pts[index, 2] = world_coords[2]

                    z = int(pts[0, 2])

                    
                    filled_poly = self._poly2mask(pts[:, 0], pts[:, 1], [shape[0], shape[1]])
                    np_mask[z, filled_poly] = int(self.structure_dict[rtstruct['ROI_Name']]) # sitk is xyz, numpy is zyx
                    #mask = sitk.GetImageFromArray(np_mask)


        return np_mask


    def _poly2mask(self, coords_x, coords_y, shape):
        
        fill_coords_x, fill_coords_y = draw.polygon(coords_x, coords_y, shape)
        mask = np.zeros(shape, dtype=np.bool)
        mask[fill_coords_y, fill_coords_x] = True # sitk is xyz, numpy is zyx

        return mask


    #---------------------------------------------#
    #               load_prediction               #
    #---------------------------------------------#
    # Read a prediction NIFTI file from the MIScnn output directory
    def load_prediction(self, index, output_path):
        # Resolve location where data should be living
        if not os.path.exists(output_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(str(output_path))
            )
        # Parse the provided index to the prediction file name
        pred_file = str(index) + ".nii.gz"
        pred_path = os.path.join(output_path, pred_file)
        # Make sure that prediction file exists under the prediction directory
        if not os.path.exists(pred_path):
            raise ValueError(
                "Prediction could not be found \"{}\"".format(pred_path)
            )
        # Load prediction from NIFTI file
        pred = nib.load(pred_path)
        # Transform NIFTI object to numpy array
        pred_data = pred.get_fdata()
        # Return prediction
        return pred_data

    #---------------------------------------------#
    #                 load_details                #
    #---------------------------------------------#
    # Parse slice thickness
    def load_details(self, i):

        spacing = self.cache[i]
        # Delete cached spacing
        del self.cache[i]
        # Return detail dictionary
        return {"spacing":spacing}

    #---------------------------------------------#
    #               save_prediction               #
    #---------------------------------------------#
    # Write a segmentation prediction into in the NIFTI file format
    def save_prediction(self, pred, index, output_path):
        # Resolve location where data should be written
        if not os.path.exists(output_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(output_path)
            )
        # Convert numpy array to NIFTI
        nifti = nib.Nifti1Image(pred, None)
        #nifti.get_data_dtype() == pred.dtype
        # Save segmentation to disk
        pred_file = str(index) + ".nii.gz"
        nib.save(nifti, os.path.join(output_path, pred_file))




