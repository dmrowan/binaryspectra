from setuptools import setup, find_packages
from distutils.util import convert_path

setup(name = "binaryspectra",
    version = 0.1,
    description = "Code for spectroscopic binaries",
    author = "Dom Rowan",
    author_email = "",
    url = "https://github.com/dmrowan/binaryspectra",
    packages = find_packages(include=['binaryspectra', 'binaryspectra.*']),
    package_data = {'binaryspectra':['data/*']},
    include_package_data = True,
    classifiers=[
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python :: 3',
      'License :: OSI Approved :: MIT License',
      'Topic :: Scientific/Engineering :: Astronomy'
      ],
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=["astropy", "cmasher", "matplotlib", "numpy", "pandas", "scipy"]
)
