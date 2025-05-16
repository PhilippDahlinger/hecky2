from setuptools import setup, find_packages

setup(
    name="hecky2",
    version="0.1.0",
    author="Philipp",
    description="Hecky2: Play Heck Meck interactively with real dice.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
)
