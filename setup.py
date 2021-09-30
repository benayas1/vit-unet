import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

packages = setuptools.find_packages(where=".")

setuptools.setup(
    name="vit_unet",
    version="0.0.1",
    author="Antonio Zarauz & Alberto Benayas",
    author_email="benayas1@gmail.com",
    description="ViT-UNet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url=gitlab project,
    classifiers=[
                "Programming Language :: Python :: 3",
                # "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent",
                ],
    python_requires='>=3.6',
    package_dir={"": "."},
    packages=packages
)