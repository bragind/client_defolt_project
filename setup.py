from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="credit_default_prediction",
    version="0.1.0",
    author="ML Engineer",
    author_email="your.email@example.com",
    description="End-to-end ML pipeline for credit default prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/credit-default-prediction",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.9",
            "black>=21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-model=src.models.train:train_model",
            "preprocess-data=src.data.preprocess:main",
            "validate-data=src.data.validation:main",
            "monitor-drift=src.monitoring.drift_detection:main",
        ],
    },
    include_package_data=True,
)