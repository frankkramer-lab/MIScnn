from setuptools import setup
from setuptools import find_packages


with open("docs/README.PyPI.md", "r") as fh:
    long_description = fh.read()

setup(
   name='miscnn',
   version='1.0.3',
   description='Framework for Medical Image Segmentation with Convolutional Neural Networks and Deep Learning',
   url='https://github.com/frankkramer-lab/MIScnn',
   author='Dominik Müller',
   author_email='dominik.mueller@informatik.uni-augsburg.de',
   license='GPLv3',
   long_description=long_description,
   long_description_content_type="text/markdown",
   packages=find_packages(),
   python_requires='>=3.6',
   install_requires=['tensorflow==2.3.0',
                     'numpy>=1.18.5',
                     'nibabel>=3.1.0',
                     'matplotlib==3.3.1',
                     'pillow==7.2.0',
                     'batchgenerators>=0.20.1',
                     'pydicom==2.0.0',
                     'SimpleITK==1.2.3',
                     'scikit-image==0.15.0'],
   classifiers=["Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.6",
                "Programming Language :: Python :: 3.7",
                "Programming Language :: Python :: 3.8",
                "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                "Operating System :: OS Independent",

                "Intended Audience :: Healthcare Industry",
                "Intended Audience :: Science/Research",

                "Topic :: Scientific/Engineering :: Artificial Intelligence",
                "Topic :: Scientific/Engineering :: Image Recognition",
                "Topic :: Scientific/Engineering :: Medical Science Apps."]
)
