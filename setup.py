import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.16"

REPO_NAME = "mlfast"
AUTHOR_USER_NAME = "Abdul-Jaweed"
SRC_REPO = "mlfast"
AUTHOR_EMAIL = "jdgaming7320@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="its a python machine learning package",
    long_description=long_description,
    long_description_content_type="text/markdown",  # Corrected field name and content type
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
