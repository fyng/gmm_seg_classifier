from setuptools import setup

setup(
    name='gmm_seg_classifier',
    version='0.0.1',
    description='Image segmentation and classification using Gaussian Mixture Model',
    author='Feiyang Huang',
    author_email='feh4005@med.cornell.edu',
    url='https://github.com/fyng/gmm_seg_classifier',
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'pytest',
        'scikit-image',
        'scikit-learn',
        'setuptools',
        'matplotlib',
        'opencv-python',
    ],
    packages=['gmm_seg_classifier'],
    license='Apache License, Version 2.0',
)