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
from miscnn.data_loading.interfaces.nifti_io import NIFTI_interface
from miscnn.data_loading.interfaces.dictionary_io import Dictionary_interface
from miscnn.data_loading.interfaces.nifti_slicer_io import NIFTIslicer_interface
from miscnn.data_loading.interfaces.image_io import Image_interface
from miscnn.data_loading.interfaces.dicom_io import DICOM_interface


miscnn_data_interfaces = {
    "NIFTI": NIFTI_interface,
    "DICT": Dictionary_interface,
    "NIFTI_S": NIFTIslicer_interface,
    "IMG": Image_interface,
    "DICOM": DICOM_interface,
}

def get_data_interface_from_file_term(file_term):
    for i in miscnn_data_interfaces.values():
        if (i.check_file_termination(file_term)):
            return i
    return None