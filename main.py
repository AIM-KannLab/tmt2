#!/usr/bin/env python3

import argparse
import os
import warnings
from predict import predict_itmt
from settings import CUDA_VISIBLE_DEVICES


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Input T1.nii.gz to predict iTMT(temporalis muscle thickness)')
    parser.add_argument('--age', '-a',type = float, default = 9.0,
                        help = 'Age of MRI subject in YEARS')
    parser.add_argument('--gender', '-g',type = str, default = 'F',
                        help = 'Gender MRI subject (M/F)')
    parser.add_argument('--input_path', '-pf', type = str, default = 'data/input/sub-pixar066_anat_sub-pixar066_T1w.nii.gz',
                        help = 'Path to input MRI subject/subjects')
    parser.add_argument('--path_to', '-pt',type = str, default = 'out/',
                        help = 'Path to save results')
    parser.add_argument('--cuda_visible_devices', '-c',type = str, default = '',
                        help = 'Specify cuda visible devices, default:None')
    parser.add_argument('--model_weight_path_selection', '-d',type = str, default = 'model_weights/densenet_itmt2.hdf5',
                        help = 'Slice selection model path')
    parser.add_argument('--model_weight_path_segmentation', '-u',type = str, default = 'model_weights/unet_itmt2.hdf5',
                        help = 'Segmentation model path')
    parser.add_argument('--df_centile_boys_csv', '-m',type = str, default = 'percentiles_chart_boys.csv',
                        help = 'CSV centiles path, boys model')
    parser.add_argument('--df_centile_girls_csv', '-w',type = str, default = 'percentiles_chart_girls.csv',
                        help = 'CSV centiles path, girls model') 
    parser.add_argument('--meta_path', '-pm',type = str, default = 'data/meta.csv',
                        help = 'Path to metafile - only for multiple subjects')
    parser.add_argument('--enable_3d', '-3d',type = bool, default = False,
                        help = 'Enable 3d calculation')
    
    args = parser.parse_args()
    itmt = predict_itmt(**vars(args))
    
