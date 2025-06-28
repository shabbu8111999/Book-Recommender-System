from setuptools import setup, find_packages

with open("README.md", "r", encoding = "utf-8")as f:
    long_description = f.read()

# edit below variables as per your requirements
REPO_NAME = "ML Based Books Recommender System"
AUTHOR_USER_NAME = "SHABAREESH NAIR"
SRC_REPO = "books_recommender"
LIST_OF_REQUIREMENTS = []


setup(
    name = SRC_REPO,
    version = "0.0.1",
    author = "SHABAREESH NAIR",
    description = "A small loacl package for ML based Books Recommendation",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/shabbu8111999/Book-Recommender-System",
    author_email = "shabareesh08@gmail.com",
    packages = find_packages(),
    license = "Apache-2.0",
    python_requires = ">=3.13.5",
    install_requires = LIST_OF_REQUIREMENTS
)