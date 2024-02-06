# Predict temporalis muscle thickness score

Prerequisites: docker

The docker will run inference with the pre-trained deep learning model to generate a temporalis muscle (TM) segmentation, and output a predicted sarcopenia score based on a subject's TMT, age, and sex.

Required inputs:
- T1 image (nifti)
- Age (float)
- Sex (string, M/F)

## Usage
1. Clone repo git clone

2. To create a docker: 
`docker build -t itmt -f Dockerfile .`

3. To run docker on example MRI: `docker run -it itmt`

4. To run docker on your own MRI: `docker run -it -v local_folder_with_nii_path:/out itmt python3.9 ./main.py --age X --gender X --img_path out/file.nii --path_to out`

where:

- local_folder_with_nii_path is the path to the folder with your MRI and file.nii is the name of the MRI file; it is also the path to the output folder where the results will be saved.
- in --age X, X should be replaced with the age of the subject in years
- in --gender X, X should be M or F, a biological gender of the subject

## References
Automated temporalis muscle quantification and growth charts for children through adulthood
[doi.org/10.1038/s41467-023-42501-1](https://doi.org/10.1038/s41467-023-42501-1)
