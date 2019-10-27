from setuptools import setup
from setuptools import find_packages


with open("docs/README.PyPI.md", "r") as fh:
    long_description = fh.read()

setup(
   name='miscnn',
   version='0.24',
   description='Framework for Medical Image Segmentation with Convolutional Neural Networks and Deep Learning',
   url='https://github.com/frankkramer-lab/MIScnn',
   author='Dominik Müller',
   author_email='dominik.mueller@informatik.uni-augsburg.de',
   license='GPLv3',
   long_description=long_description,
   long_description_content_type="text/markdown",
   packages=find_packages(),
   install_requires=['numpy==1.16.4',
                     'tensorflow-gpu==1.13.1',
                     'Keras>=2.2.4',
                     'nibabel>=2.4.0',
                     'matplotlib>=3.0.3',
                     'batchgenerators>=0.19.3'],
   classifiers=["Programming Language :: Python :: 3",
                "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                "Operating System :: OS Independent",

                "Intended Audience :: Healthcare Industry",
                "Intended Audience :: Science/Research",

                "Topic :: Scientific/Engineering :: Artificial Intelligence",
                "Topic :: Scientific/Engineering :: Image Recognition",
                "Topic :: Scientific/Engineering :: Medical Science Apps."]
)
