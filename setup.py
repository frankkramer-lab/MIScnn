from setuptools import setup
from setuptools import find_packages


with open("docs/README.PyPI.md", "r") as fh:
    long_description = fh.read()

setup(
   name='miscnn',
   version='0.38',
   description='Framework for Medical Image Segmentation with Convolutional Neural Networks and Deep Learning',
   url='https://github.com/frankkramer-lab/MIScnn',
   author='Dominik Müller',
   author_email='dominik.mueller@informatik.uni-augsburg.de',
   license='GPLv3',
   long_description=long_description,
   long_description_content_type="text/markdown",
   packages=find_packages(),
   python_requires='>=3.6',
   install_requires=['tensorflow==2.2.0',
                     'numpy>=1.18.2',
                     'nibabel>=3.1.0',
                     'matplotlib==3.0.3',
                     'Pillow==6.2.2',
                     'batchgenerators>=0.19.3'],
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
