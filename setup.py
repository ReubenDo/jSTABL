#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

requirements = [
    'torchio==0.13.10',
    'pandas',
    'nibabel',
    'SimpleITK',
]

setup(
    author="Reuben Dorent",
    author_email='reuben.dorent@kcl.ac.uk',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description=(
        "joint Segmentation of Tissues and Brain Lesions"
    ),
    install_requires=requirements,
    license="MIT license",
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='jSTABL',
    name='jSTABL',
    packages=find_packages(include=['jstabl', 'jstabl.*']),
    scripts=['jstabl/jstabl_glioma', 'jstabl/jstabl_wmh', 'jstabl/jstabl_control'],
    setup_requires=[],
    tests_require=[],
    url='https://github.com/ReubenDo/jSTABL',
    version='0.0.02',
    zip_safe=False,
)
