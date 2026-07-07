from setuptools import setup, find_packages

setup(
    name             = "vecsim",
    version          = "1.0.0",
    author           = "Dr. Habte Lejebo",
    description      = "Vehicular Edge Computing Simulator with DRL Scheduling",
    packages         = find_packages(where="src"),
    package_dir      = {"": "src"},
    python_requires  = ">=3.8",
    install_requires = ["numpy>=1.21", "matplotlib>=3.4"],
)