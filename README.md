# Artificial Intelligence Analysis of Temporalis Muscle Thickness for Monitoring Sarcopenia and Clinical Outcomes in Patients with Pediatric Brain Tumors

Containerized version of the deep learning model for temporalis muscle thickness (TMT) segmentation and quantification. Updated version of the model from:
Zapaishchykova, A., Liu, K.X., Saraf, A. et al. Automated temporalis muscle quantification and growth charts for children through adulthood. Nat Commun 14, 6863 (2023). https://doi-org.ezp-prod1.hul.harvard.edu/10.1038/s41467-023-42501-1 

## Prerequisites
- Docker OR Singularity (use tmt.def file)

## Description
The docker will run an inference with the pre-trained deep learning model to generate a temporalis muscle (TM) segmentation and output a predicted sarcopenia score based on a subject's TMT, age, and sex.

## Required Inputs
- T1 image (nifty)
- Age (float)
- Sex (string, M/F)

## Usage (Docker)
1. Clone the repo: `git clone <repository_url>` & `cd tmt2`

2. To create a docker:
`docker build -t itmt -f Dockerfile .`

3. To run docker on an example MRI:
`docker run -it itmt`

4. To run docker on your own *SINGLE MRI*:

`docker run -it -v <local_folder_with_nii_path>:/out itmt python3.9 ./main.py --age 9 --gender F --input_path data/input/sub-pixar066_anat_sub-pixar066_T1w.nii.gz`
OR with 3d option:
`docker run -it -v <local_folder_with_nii_path>:/out itmt python3.9 ./main.py --age 9 --gender F  --enable_3d True --input_path data/input/sub-pixar066_anat_sub-pixar066_T1w.nii.gz`

- `local_folder_with_nii_path`: The folder mounted to the docker container. It should contain the MRI file or files to process. Results will be saved in the same folder in the 'results' subfolder.
- `input_path`: Path to the MRI file or folder with MRI files.
- `age`: Age of the subject in years.
- `gender`: Gender of the subject (M/F).
- `enable_3d`: If True, the model will run in 3D mode. If False, the model will run in 2D mode.

Example of running docker on *MULTIPLE MRI* files:
`docker run -it -v <local_folder_with_nii_path>:/out itmt python3.9 ./main.py --input_path data/input/ --meta_path data/meta.csv`
- `meta_path`: Path to a CSV file listing filenames, ages, and sexes of subjects. The CSV file should have the following columns: 'filename','age','sex':

```
filename,age,sex
input/sub-pixar066_anat_sub-pixar066_T1w.nii.gz,6,F
input/sub-pixar067_anat_sub-pixar067_T1w.nii.gz,7,M
...
```

## References
- TBD - under review
- Zapaishchykova, A., Liu, K.X., Saraf, A. et al. Automated temporalis muscle quantification and growth charts for children through adulthood. Nat Commun 14, 6863 (2023). https://doi-org.ezp-prod1.hul.harvard.edu/10.1038/s41467-023-42501-1 

Anonymized data used in the publication is included in the `data/csv` folder together with the data dictionaries for each file.

## Notes
- This is a CPU-only release. The docker will run on the CPU, and the inference time will depend on the number of subjects and the size of the MRI files.
- This won't work on MAC OS. Linux or Windows is recommended.

## Known Issues
If you encounter an error with docker `AppArmor enabled on system but the docker-default profile could not be loaded`, try running the following command:
`sudo apt install -y apparmor && systemctl restart docker && service docker restart`
