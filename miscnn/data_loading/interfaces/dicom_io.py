#==============================================================================#
#  Author:       Michael Lempart                                               #
#  Copyright:    2020 Department of Radiation Physics,                         #
#                Lund University                                               #
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
import pydicom
import SimpleITK as sitk
from skimage import draw

# Internal libraries/scripts
from miscnn.data_loading.interfaces.abstract_io import Abstract_IO

#-----------------------------------------------------#
#                 DICOM I/O Interface                 #
#-----------------------------------------------------#
""" This class provides a Data I/O Interface that can be used to load images and structure sets
    that are provided in the Digital Imaging and Communications in Medicine (DICOM) format

    The DICOM Viewer was designed for the Lung CT Segmentation Challenge 2017.
    https://wiki.cancerimagingarchive.net/display/Public/Lung+CT+Segmentation+Challenge+2017#cb38430390714dbbad13f267f39a33eb
"""

class DICOM_interface(Abstract_IO):
    # Class variable initialization
    def __init__(self, channels=1, classes=2, three_dim=True, mask_background=0,
                 structure_dict={}, annotation_tag=None):

        """
        Args:
            channels (int): Number of channels of the input images.
            classes (int): Number of segmentation classes.
            three_dim (bool): 3D volume if True.
            mask_background (int): Determines the value of the background pixels.
            structure_dict (dict): Dictionary containg ROI names.
            annotation_tag (string): String to identify annotation series.
        Returns:
            None
        """

        self.data_directory = None
        self.channels = channels
        self.classes = classes
        self.three_dim = three_dim
        self.annotation_tag = annotation_tag
        self.cache = dict()
        assert isinstance(mask_background,int), "mask_background value should be an integer"
        self.mask_background = mask_background
        self.structure_dict = structure_dict

    #---------------------------------------------#
    #                  initialize                 #
    #---------------------------------------------#
    # Initialize the interface and return number of samples
    def initialize(self, input_path):

        """
        Args:
            input_path (str): Path to the dataset.
        Returns:
            sample_list (list): List of all cases found in the provided path.
        """

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
    # Read a volume DICOM file from the data directory
    def load_image(self, index):

        """
        Args:
            index (str): Description of the sample that should be opened.
        Returns:
            dcm (array): CT 3D volume.
        """

        dicom_images_path = None

        img_path = os.path.join(self.data_directory, index)

        # search folders and find a DICOM image of the whole scan to get
        # the path of the DICOM images
        dicom_images_path = None
        for root, _, files in os.walk(img_path):
            for _ in filter(lambda x: re.findall('.*001.dcm', x), files):
                dicom_images_path = root
                break

        # Make sure that the image file exists in the data set directory
        if not dicom_images_path or not os.path.exists(dicom_images_path):
            raise ValueError(
                "No DICOM scans could not be found \"{}\"".format(img_path)
            )

        #create a ImageSeriesReader Object
        dicom_reader = sitk.ImageSeriesReader()

        #read all DICOM files contained in the provided folder
        dicom_file_names = dicom_reader.GetGDCMSeriesFileNames(
                                                         str(dicom_images_path))

        #set file names
        dicom_reader.SetFileNames(dicom_file_names)

        #read dicom files
        dcm = dicom_reader.Execute()

        # Read one of the images to get DICOM Pixelspacing and SliceThickness
        # paramters
        dcm_temp = pydicom.read_file(dicom_file_names[0])

        self.cache[index] = np.array([dcm_temp.SliceThickness,
                                      dcm_temp.PixelSpacing[0],
                                      dcm_temp.PixelSpacing[1]],
                                      dtype=np.float32)


        #return both the sitk image and all images as a numpy array
        return sitk.GetArrayFromImage(dcm)


    def load_image_sitk(self, index):

        """
        Args:
            index (str): Description of the sample that should be opened.
        Returns:
            dcm (sitkImage Object): Image in Sitk format (needed for other subfunctions of this class)
        """

        dicom_images_path = None

        # Make sure that the image file exists in the data set directory
        img_path = os.path.join(self.data_directory, index)

        # search folders and find a DICOM image of the whole scan to get
        # the path of the DICOM images
        dicom_images_path = None
        for root, _, files in os.walk(img_path):
            for _ in filter(lambda x: re.findall('.*001.dcm', x), files):
                dicom_images_path = root
                break

        if not dicom_images_path or not os.path.exists(dicom_images_path):
            raise ValueError(
                "No DICOM scans could not be found \"{}\"".format(img_path)
            )

        #create a ImageSeriesReader Object
        dicom_reader = sitk.ImageSeriesReader()

        #read all DICOM files contained in the provided folder
        dicom_file_names = dicom_reader.GetGDCMSeriesFileNames(
                                                         str(dicom_images_path))

        #set file names
        dicom_reader.SetFileNames(dicom_file_names)

        #read dicom files
        dcm = dicom_reader.Execute()

        #Read one of the images to get DICOM Pixelspacing and SliceThickness
        #paramters
        dcm_temp = pydicom.read_file(dicom_file_names[0])

        self.cache[index] = np.array([dcm_temp.SliceThickness,
                                      dcm_temp.PixelSpacing[0],
                                      dcm_temp.PixelSpacing[1]],
                                      dtype=np.float32)


        #return both the sitk image and all images as a numpy array
        return dcm


    #---------------------------------------------#
    #              load_segmentation              #
    #---------------------------------------------#
    # Read a RT structure file from the data directory
    def load_segmentation(self, index):

        """
        Args:
            index (str): Description of the sample that should be opened.
        Returns:
            segmentations (array): 3D volume containing segmentations of all ROIs.
        """

        rtStruct_path = self.get_rtStruct_file(index)

        #get names, IDs and sequences of all ROIs contained in the provided structure file
        contours = self.get_ROI_data(rtStruct_path)

        #Load CT images in sitk format
        images = self.load_image_sitk(index)

        #get segmentations
        segmentations = self.convert(contours, images)

        # Return segmentations
        return segmentations


    #Gets Name, ID and sequence from a provided structure file
    def get_ROI_data(self, rtstruct_file = None):

        """
        Args:
            rtstruct_file (str): Path to an RT structure DICOM file.
        Returns:
            contour_data (dict): Dictionary containing ROI name, ID and Sequence.
        """

        ss_file = pydicom.read_file(rtstruct_file)

        contours = []

        for sequence, metadata in zip(ss_file.ROIContourSequence,
                                      ss_file.StructureSetROISequence):

            contour_data = {}

            contour_data['ROI_ID'] = metadata.ROINumber
            contour_data['ROI_Name'] = metadata.ROIName
            contour_data['ROI_Sequence'] = []
            for contour in sequence.ContourSequence:
                contour_data['ROI_Sequence'].append({
                    'type': (contour.ContourGeometricType if hasattr(contour, 'ContourGeometricType') else 'No_type'),
                    'points': {
                        'x': ([contour.ContourData[i] for i in range(0, len(contour.ContourData), 3)]),
                        'y': ([contour.ContourData[i + 1] for i in range(0, len(contour.ContourData), 3)]),
                        'z': ([contour.ContourData[i + 2] for i in range(0, len(contour.ContourData), 3)])
                    }
                })

            contours.append(contour_data)

        return contours

    #gets the path of a RT structure file
    def get_rtStruct_file(self, index):

        """
        Args:
            index (str): Description of the sample that should be opened.
        Returns:
            rtStruct_path (str): path to a RT structure file.
        """


        path = os.path.join(self.data_directory, index)

        # Check if annotation tag is available
        if not self.annotation_tag:
            raise ValueError("Please provide an annotation tag")

        # search folder structure for the segmentation by using the
        # annotation tag
        rtStruct_path = None
        pattern = ".*" + self.annotation_tag + ".*"
        for root, dirs, files in os.walk(path):
            annot_dir = list(filter(lambda x: re.match(pattern, x), dirs))
            if len(annot_dir) >= 1:
                annot_dir = os.path.join(root, annot_dir[0])
                files = os.listdir(annot_dir)
                if len(files) > 1 or len(files) == 0:
                    raise ValueError("More than one or no segmentation file found",
                                     index, annot_dir)
                rtStruct_path = os.path.join(annot_dir, files[0])
                break

        # Make sure that the segmentation file exists in the data set directory
        if not rtStruct_path or not os.path.exists(rtStruct_path):
            raise ValueError(
                "Structure file could not be found \"{}\"".format(rtStruct_path)
            )

        return rtStruct_path

    #---------------------------------------------#
    #         convert structure sequences         #
    #---------------------------------------------#
    #converts structure sequences to filled poly masks
    def convert(self, rtstruct_contours, dicom_image):

        """
        Args:
            rtstruct_contours (dict): dictionary containing structure name, ID and Sequence.
            dicom_image (SitkImage Object): SITK image object.
        Returns:
            np_mask (array): segmentations as a numpy array.
        """

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
                        # convert world coordinates to voxel coordinates
                        voxel_coords = dicom_image.TransformPhysicalPointToIndex((coordinates['x'][index], coordinates['y'][index], coordinates['z'][index]))
                        pts[index, 0] = voxel_coords[0]
                        pts[index, 1] = voxel_coords[1]
                        pts[index, 2] = voxel_coords[2]

                    z = int(pts[0, 2])


                    filled_poly = self._poly2mask(pts[:, 0], pts[:, 1], [shape[0], shape[1]])
                    np_mask[z, filled_poly] = int(self.structure_dict[rtstruct['ROI_Name']]) # sitk is xyz, numpy is zyx

        return np_mask

    #converts polygon to mask
    def _poly2mask(self, coords_x, coords_y, shape):

        """
        Args:
            coords_x (array): Array with x coordinates of a structure sequence.
            coords_y (array): Array with y coordinates of a structure sequence.
            shape (list): segmentation image shape.
        Returns:
            mask (array): filled 2D segmentation mask.
        """

        poly_coords_x, poly_coords_y = draw.polygon(coords_x, coords_y, shape)
        #create an empty mask
        mask = np.zeros(shape, dtype=np.bool)
        #Observe that sitk uses the xyz convention, wheras numpy use zyx
        mask[poly_coords_y, poly_coords_x] = True

        return mask


    #---------------------------------------------#
    #              load_ROI_names                 #
    #---------------------------------------------#
    #gets all ROIs contained in a DICOM structure file
    def get_ROI_names(self, index):

        """
        Args:
            index (str): Description of the sample that should be opened.
        Returns:
            structure_names (list): List containing all ROI names and IDs found in a structure file.
        """

        rtStruct_path = self.get_rtStruct_file(index)

        contours = self.get_ROI_data(rtStruct_path)

        structure_names = [[cont['ROI_ID'], cont['ROI_Name']] for cont in contours]

        return structure_names


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
