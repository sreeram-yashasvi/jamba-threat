from setuptools import setup, find_packages

setup(
    name="jamba",
    version="1.0.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.8.0",
        "pandas>=1.2.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "joblib>=1.0.0",
    ],
    python_requires=">=3.7",
    author="Your Name",
    description="Jamba Threat Detection Model",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
) 