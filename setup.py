import setuptools
from setuptools import setup, find_packages
import codecs
import os

# with open("README.md", "r") as f:
#     long_description = f.read()

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README1.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

__version__ = "0.0.16"

REPO_NAME = "mlfast"
AUTHOR_USER_NAME = "Abdul-Jaweed"
SRC_REPO = "mlfast"
AUTHOR_EMAIL = "jdgaming7320@gmail.com"

setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="It's a Python machine learning package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
