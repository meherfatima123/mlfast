import setuptools
from setuptools import setup, find_packages
import codecs
import os   

# Get the directory of the current script
here = os.path.abspath(os.path.dirname(__file__))

# Define the path to your README file, assuming it's named README1.md
readme_path = os.path.join(here, "README.md")

# Check if the README file exists
if os.path.isfile(readme_path):
    # If the file exists, read its content for long_description
    with codecs.open(readme_path, encoding="utf-8") as fh:
        long_description = fh.read()
else:
    # If the file doesn't exist, you might want to provide a fallback or raise an error.
    long_description = "No README1.md file found."

__version__ = "0.0.20"

REPO_NAME = "mlfast"
AUTHOR_USER_NAME = "Abdul-Jaweed"
SRC_REPO = "mlfast"
AUTHOR_EMAIL = "jdgaming7320@gmail.com"
