import io
import re

from setuptools import setup
from setuptools import find_packages


# PyCOMET info
file = open('pycomet/__init__.py').read()
info = {}
for k in ("author", "email", "url", "license", "version"):
  info[k] = re.search(
    r"^__{0}__ = ['\"]([^'\"]*)['\"]".format(k), file, re.M
  ).group(1)

# Description
description = "Physics-informed machine learning for scientific " \
  "computing and operator discovery."

# Dependencies
with open("requirements.txt", "r") as f:
  install_requires = [x.strip() for x in f.readlines()]

# Scripts
console_scripts = [
  'pycomet = pycomet.__main__:main'
]

# Info
with io.open("README.md", "r", encoding="utf-8") as f:
  long_description = f.read()

classifiers = [
  "Development Status :: 1 - Beta",
  "Intended Audience :: Developers",
  "Intended Audience :: Education",
  "Intended Audience :: Science/Research",
  "License :: OSI Approved :: Apache-2.0 License",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.6",
  "Programming Language :: Python :: 3.7",
  "Programming Language :: Python :: 3.8",
  "Topic :: Scientific/Engineering",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
  "Topic :: Scientific/Engineering :: Mathematics",
  "Topic :: Scientific/Engineering :: Physics",
  "Topic :: Software Development :: Libraries",
  "Topic :: Software Development :: Libraries :: Python Modules"
]

keywords = [
  "Deep Learning",
  "Machine Learning",
  "Neural Networks",
  "Neural Operators",
  "Scientific computing",
  "Differential equations",
  "IDE/ODE/PDE solver"
]

setup(
  name="pycomet",
  version=info["version"],
  description=description,
  long_description=long_description,
  long_description_content_type="text/markdown",
  author=info["author"],
  author_email=info["email"],
  url=info["url"],
  license=info["license"],
  install_requires=install_requires,
  entry_points ={'console_scripts': console_scripts},
  classifiers=classifiers,
  keywords=keywords,
  packages=find_packages(),
  include_package_data=True,
  python_requires=">=3.8",
)
