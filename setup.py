import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neossat",
    version="2020.6",
    author="Jason Rowe",
    author_email="Jason.Rowe@ubishops.ca",
    description="A library for processing NEOSSat astronomy observations.",
    long_description=long_description,
    url="https://github.com/jasonfrowe/neossat",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
