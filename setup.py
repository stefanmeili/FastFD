'''
FastFD - GPU Accelerated Finite Differences Simulation Library
==============================================================

Copyright 2021 - Stefan Meili
MIT License.
'''



import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "FastFD",
    version = "0.1",
    author = "Stefan Meili",
    author_email = "stefan.meili@gmail.com",
    description = "A finite difference simulation library",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/stefanmeili/fastfd",
    project_urls = {
        "Bug Tracker": "https://github.com/stefanmeili/fastfd/issues",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages = setuptools.find_packages(),
    install_requires=["numpy", "scipy"],
    python_requires = '>= 3.6',
    include_package_data=True,
)
