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
import miscnn.cli.data_exploration as data_exp
import os
import os.path
from miscnn import Data_IO
from miscnn.data_loading.interfaces import miscnn_data_interfaces

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

data_exp.register_commands(subparsers.add_parser("data_exp"))
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
    if (not args.imagetype in miscnn_data_interfaces.keys()):
        raise RuntimeError("Unknown Format")
    
    interface = miscnn_data_interfaces[args.imagetype]()
    
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
    data_exp.execute(args)
    
        
#check if z-score observes normal distribution. calculate transformation to normal distribution