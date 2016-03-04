from distutils.core import setup

setup(
    name='bat-country',
    packages=['batcountry'],
    version='0.2-crowsonkb',
    description='A lightweight, extendible, easy to use Python package for deep dreaming and image generation with Caffe and CNNs',
    author='Adrian Rosebrock',
    author_email='adrian@pyimagesearch.com',
    url='https://github.com/crowsonkb/bat-country',
    license='MIT',
    install_requires=[
        'Pillow>=2.9.0',
        'decorator>=3.4.2',
        'imutils>=0.2.2',
        'matplotlib>=1.4.3',
        'mock>=1.0.1',
        'networkx>=1.9.1',
        'nose>=1.3.7',
        'numpy>=1.10.4',
        'protobuf>=3.0.0b2',
        'pyparsing>=2.0.3',
        'python-dateutil>=2.4.2',
        'pytz>=2015.4',
        'scikit-image>=0.11.3',
        'scipy>=0.15.1',
        'six>=1.9.0'
    ],
    keywords=['computer vision', 'machine learning', 'deep learning',
        'convolutional neural network', 'deep dream', 'inceptionism'],
    classifiers=[],
)
