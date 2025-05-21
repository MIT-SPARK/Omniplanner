from setuptools import find_packages, setup

setup(
    name="robot_vocalizer",
    version="0.0.1",
    url="",
    author="",
    author_email="",
    description="Speech to text pipeline",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["*.yaml", "*.pddl"]},
    install_requires=[
        "deepgram-sdk==3.*",
        "openai==1.78.1 ",
    ],
)
