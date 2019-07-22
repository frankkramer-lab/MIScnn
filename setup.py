from setuptools import setup
from setuptools import find_packages

setup(
   name='miscnn',
   version='0.1',
   description='Framework for Medical Image Segmentation with Convolutional Neural Networks and Deep Learning',
   url='https://github.com/muellerdo/MIScnn',
   author='Dominik Müller',
   author_email='dominik.mueller@informatik.uni-augsburg.de',
   license='MIT',
   packages=find_packages(),
   install_requires=['numpy>=1.16.4',
                     'tqdm>=4.32.2',
                     'Keras>=2.2.4',
                     'nibabel>=2.4.0',
                     'matplotlib>=3.0.3']
)
