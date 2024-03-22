# Predict temporalis muscle thickness score

Prerequisites: docker
*Note*: this is a CPU-only release. The docker will run on CPU, and the inference time will depend on the number of subjects and the size of the MRI files.
*Note2*: this won't work on MAC OS. It is recommended to use Linux or Windows.

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

4. To run docker on your own *SINGLE MRI*: `docker run -it -v local_folder_with_nii_path:/out itmt python3.9 ./main.py --age 9 --gender F --input_path data/input/sub-pixar066_anat_sub-pixar066_T1w.nii.gz`

where:

- `local_folder_with_nii_path` is the folder that will be mounted to the docker container. It should contain the MRI file or files that you want to process. The results will be saved in the same folder in the results subfolder.
- `input_path` can be either a path to the MRI file or a path to the folder with MRI files. For instance, if you have only one file, its okay to use `--input_path data/input/sub-pixar066_anat_sub-pixar066_T1w.nii.gz`, but if you have multiple files, you should use `--input_path data/input/` (see example below).
- in --age X, X should be replaced with the age of the subject in years (in case of multiple files, this variable should not be used) 
- in --gender X, X should be sex the subject (M/F) (in case of multiple files, this variable should not be used) 

Example of running docker on *MULTIPLE MRI* files:
`docker run -it -v local_folder_with_nii_path:/out itmt python3.9 ./main.py --input_path data/input/ --meta_path data/meta.csv`

where:
- --meta_path is a csv list 3 columns: filename, age in years(float) and sex (F/M). See an example of the csv file below or in data/meta.csv file:
```
filename,age,sex
input/sub-pixar066_anat_sub-pixar066_T1w.nii.gz,6,F
input/sub-pixar067_anat_sub-pixar067_T1w.nii.gz,7,M
...
```
- `local_folder_with_nii_path` is the folder that will be mounted to the docker container.
- `data/input/` is the folder with MRI files

The result will be located in local_folder_with_nii_path/results folder. The result will be a results.csv file with the predicted iTMT and cross-sectional area, together with all images with the predicted TM segmentation and the registered and processed MRI.

## References
Automated temporalis muscle quantification and growth charts for children through adulthood
[doi.org/10.1038/s41467-023-42501-1](https://doi.org/10.1038/s41467-023-42501-1)

## Known issues
If you encounter an error with docker `AppArmor enabled on system but the docker-default profile could not be loaded`, try running the following command:
`sudo apt install -y apparmor && systemctl restart docker && service docker restart`