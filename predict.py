from __future__ import generators
import logging
import glob, os, functools
import sys

import SimpleITK as sitk
from scipy.signal import medfilt
import numpy as np
from numpy import median
import scipy
import nibabel as nib
import skimage
import matplotlib.pyplot as plt
import scipy.misc
from scipy import ndimage
from skimage.transform import resize,rescale
import cv2
import itk
import subprocess
from skimage import measure
from scipy.spatial.distance import cdist
import imea
from intensity_normalization.typing import Modality, TissueType
from intensity_normalization.normalize.zscore import ZScoreNormalize

import pandas as pd
import tensorflow as tf
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 
   
from scripts.densenet_regression import DenseNet
from scripts.unet import get_unet_2D
from scripts.preprocess_utils import load_nii, save_nii, find_file_in_path,iou,enhance_noN4, crop_center, get_id_and_path
from scripts.feret import Calculater
from settings import target_size_dense_net, target_size_unet, unet_classes, softmax_threshold, scaling_factor
from scripts.infer_selection import get_slice_number_from_prediction, funcy
import warnings

warnings.filterwarnings('ignore')
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# MNI templates 
age_ranges = {"golden_image/mni_templates/nihpd_asym_04.5-08.5_t1w.nii" : {"min_age":3, "max_age":7},
                "golden_image/mni_templates/nihpd_asym_07.5-13.5_t1w.nii": {"min_age":8, "max_age":13},
                "golden_image/mni_templates/nihpd_asym_13.0-18.5_t1w.nii": {"min_age":14, "max_age":35}}

# function to compute the crop line (max and min y coordinates of the contour)
def compute_crop_line(img_input,infer_seg_array_2d_1,infer_seg_array_2d_2):
    binary = img_input>-1.7
    binary_smoothed = scipy.signal.medfilt(binary.astype(int), 51)
    img = binary_smoothed.astype('uint8')
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img.shape, np.uint8)
    img = cv2.drawContours(mask, contours, -1, (255),1)

    max_y,ind_max = 0,0
    min_y,ind_min = 512,0
    if len(contours)>0:
        for i in range(0,len(contours[0])):
            x,y = contours[0][i][0]
            if y<=min_y:
                min_y,ind_min = y,i
            if y>=max_y:
                max_y,ind_max = y,i

        fig, ax = plt.subplots(1,1,figsize=(5,5))
        ax.imshow(img_input, interpolation=None, cmap=plt.cm.Greys_r)
        ax.imshow(infer_seg_array_2d_1,cmap='jet',alpha=0.5)
        ax.imshow(infer_seg_array_2d_2,cmap='jet',alpha=0.5)
        crop_line = (contours[0][ind_min][0][0]+contours[0][ind_max][0][0])/2
        
        ax.plot((crop_line, crop_line),
                (contours[0][ind_min][0][1], contours[0][ind_max][0][1]), lw=1, c='b')

        fig.show()

        return crop_line
    else:
        return 100
 

def find_exact_centile(input_tmt, age, df):
    # Find closest age 
    val,i = closest_value(df['x'], age)
    # Extract centile_tmt columns
    cents = ['X'+str(x) for x in range(1,100)]
    # Use loc to get series
    df_cent = df.iloc[i].loc[cents] 
    val,i = closest_value(df_cent, input_tmt)
    # Sort
    centile_tmt = df_cent.index[i].replace('X','')
    if centile_tmt == '1':
        centile_tmt = '<1'
    if centile_tmt == '99':
        centile_tmt = '>99'
    return centile_tmt
 
# function to select the correct MRI template based on the age  
def select_template_based_on_age(age):
    for golden_file_path, age_values in age_ranges.items():
        if age_values['min_age'] <= int(age) and int(age) <= age_values['max_age']: 
            #print(golden_file_path)
            return golden_file_path
   
# register the MRI to the template     
def register_to_template(input_image_path, output_path, fixed_image_path,rename_id,create_subfolder=True):
    fixed_image = itk.imread(fixed_image_path, itk.F)

    # Import Parameter Map
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile('golden_image/mni_templates/Parameters_Rigid.txt')

    if "nii" in input_image_path and "._" not in input_image_path:
        #print(input_image_path)

        # Call registration function
        try:        
            moving_image = itk.imread(input_image_path, itk.F)
            result_image, result_transform_parameters = itk.elastix_registration_method(
                fixed_image, moving_image,
                parameter_object=parameter_object,
                log_to_console=False)
            image_id = input_image_path.split("/")[-1]
            
            itk.imwrite(result_image, output_path+"/"+rename_id+".nii.gz")
                
            print("Registered ", rename_id)
        except:
            print("Cannot transform", rename_id)
           

def compute_distance_between_two_masks(image_array, infer_seg_array_3d_1_filtered, infer_seg_array_3d_2_filtered, 
                                       save_path):
    # Remove singleton dimensions to make arrays 2D
    infer_seg_array1 = np.squeeze(infer_seg_array_3d_1_filtered)
    infer_seg_array2 = np.squeeze(infer_seg_array_3d_2_filtered)
    
    # Find contours of each mask
    contours_1 = measure.find_contours(infer_seg_array1, 0.5)
    contours_2 = measure.find_contours(infer_seg_array2, 0.5)
    
    # Flatten the list of contours and get coordinates as points
    points_1 = np.concatenate(contours_1)
    points_2 = np.concatenate(contours_2)
    
    # Calculate pairwise distances
    d_1_to_2 = cdist(points_1, points_2)
    
    # Closest Distance calculation
    min_distance = d_1_to_2.min()  # Closest distance between any points on the contours
    min_idx = np.unravel_index(np.argmin(d_1_to_2), d_1_to_2.shape)  # Index of the closest points
    closest_point_1 = points_1[min_idx[0]]
    closest_point_2 = points_2[min_idx[1]]
    
    # Visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(image_array, cmap='gray')
    
    # Plot contours for mask 1 and mask 2 without conditional labels
    for contour in contours_1:
        plt.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)  # Mask 1
    for contour in contours_2:
        plt.plot(contour[:, 1], contour[:, 0], 'b', linewidth=2)  # Mask 2
    
    # Draw line for the closest distance
    plt.plot([closest_point_1[1], closest_point_2[1]], [closest_point_1[0], closest_point_2[0]], 'y-', linewidth=2, label="Closest Distance")
    
    # Add legend items manually after plotting
    plt.plot([], [], 'r', label="Mask 1")
    plt.plot([], [], 'b', label="Mask 2")
    plt.plot([], [], 'y-', label="Closest Distance")

    # Annotate the distances
    plt.text(10, 10, f"Closest Distance: {min_distance:.2f}", color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
    
    # Show the legend and save
    plt.legend()
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    return min_distance


def feret_3d(infer_seg_array_3d_1_filtered, infer_seg_array_3d_2_filtered, 
             threshold_mm=0.5, 
             spatial_resolution_xy=1, 
             spatial_resolution_z=1):
    # Calculate 3D shape measurements for the first mask
    img_1 = np.max(infer_seg_array_3d_1_filtered, axis=0)
    img_2 = np.max(infer_seg_array_3d_2_filtered, axis=0)
    
    df_2d_1, df_3d_1  = imea.extract.shape_measurements_3d(img_1, threshold_mm, spatial_resolution_xy, spatial_resolution_z, dalpha=9, min_object_area=10, n_objects_max=-1)
    df_2d_2, df_3d_2  = imea.extract.shape_measurements_3d(img_2, threshold_mm, spatial_resolution_xy, spatial_resolution_z, dalpha=9, min_object_area=10, n_objects_max=-1)
    # Extract relevant metrics for mask 1
    
    print(type(df_3d_1))
    
    print("Mask 1 Metrics:", df_3d_1)
    print("Mask 2 Metrics:", df_3d_2)
    
    return df_3d_1,df_3d_2           
             
# helper function to find the closest value in a list 
def closest_value(input_list, input_value):
    arr = np.asarray(input_list)
    i = (np.abs(arr - input_value)).argmin()
    return arr[i], i

# function to filter the islands in the segmentation mask, to keep only the largest one      
def filter_islands(muscle_seg):
    img = muscle_seg.astype('uint8')
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img.shape, np.uint8)
    cnt_mask = np.zeros(img.shape, np.uint8)
    area = 0
    c=0
    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        area = cv2.contourArea(c)
        mask = cv2.fillPoly(mask, pts=[c], color=(255, 0, 0))
        cnt_mask =  cv2.drawContours(cnt_mask, [c], -1, (255, 255, 255), 0)#cv.drawContours(cnt_mask, [c], 0, (255,255,0), 2)
    return mask, area, c

# predict the TMT score based on the input.nii image and age 
def predict_itmt(age = 9, gender="M",
                 input_path = 'data/t1_mris/nihm_reg/clamp_1193_v1_t1w.nii.gz', #can be a path or a file
                 meta_path = 'data/meta.csv',
                 path_to ="data/bch/", cuda_visible_devices="0",
                 model_weight_path_selection = 'model_weights/densenet_itmt2.hdf5',
                 model_weight_path_segmentation = 'model_weights/unet_itmt2.hdf5',
                 df_centile_boys_csv = 'percentiles_chart_boys.csv',
                 df_centile_girls_csv= 'percentiles_chart_girls.csv',
                 df_centile_girls_csv_csa ='percentiles_chart_girls_csa.csv',
                 df_centile_boys_csv_csa = 'percentiles_chart_boys_csa.csv',
                 enable_3d=False, n_slices=50):
    
    # load image
    threshold = 0.75
    alpha = 0.8 
    
    print("CUDA_VISIBLE_DEVICES:", cuda_visible_devices)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    if len(physical_devices) == 0:
        physical_devices = tf.config.experimental.list_physical_devices('CPU')
    else:   
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
    # load models
    model_selection = DenseNet(img_dim=(256, 256, 1), 
                    nb_layers_per_block=16, nb_dense_block=4, growth_rate=16, nb_initial_filters=16, 
                    compression_rate=0.5, sigmoid_output_activation=True, 
                    activation_type='relu', initializer='glorot_uniform', output_dimension=1, batch_norm=True )
    model_selection.load_weights(model_weight_path_selection)
    print('\n','\n','\n','loaded:' ,model_weight_path_selection)  
        
    model_unet = get_unet_2D(unet_classes,(target_size_unet[0], target_size_unet[1], 1),\
            num_convs=2,  activation='relu',
            compression_channels=[16, 32, 64, 128, 256, 512],
            decompression_channels=[256, 128, 64, 32, 16])
    model_unet.load_weights(model_weight_path_segmentation)
    print('\n','\n','\n','loaded:' ,model_weight_path_segmentation)  
    
    list_of_nii_images,list_of_ages,list_of_sexes = [],[],[]
    # check if img_path is a path to a folder or a file
    if os.path.isdir(input_path):
        # make list of all nii files in the folder
        list_of_nii_images = glob.glob(input_path+"/*.nii*")
        meta_df = pd.read_csv(meta_path, header=0) 
        ommited_files = []
        # Pull age and gender by img_path from meta_df
        temp_list = list_of_nii_images.copy()
        for img_path in list_of_nii_images:
            # find if img_path in meta_df['filename']
            age, gender = 0, 0
            for idx, row in meta_df.iterrows():
                if str(img_path) in row['filename']:
                    age = row['age']
                    gender = row['sex']
                    list_of_ages.append(age)
                    list_of_sexes.append(gender)
                    
            if age == 0:
                print("No metadata found for", img_path)
                temp_list.remove(img_path)
                ommited_files.append(img_path)
                
        list_of_nii_images = temp_list.copy()
    else:
        print("its a file")
        list_of_nii_images.append(input_path)
        list_of_sexes.append(gender)
        list_of_ages.append(age)
     
    print(list_of_nii_images,list_of_ages)
       
    for idx in range(len(list_of_nii_images)):
        #retrieve by name
        img_path = list_of_nii_images[idx]
        age = list_of_ages[idx]
        gender = list_of_sexes[idx]
        
        image, affine = load_nii(img_path)
    
        # path to store registered image in
        patient_id = img_path.split("/")[-1].split(".")[0]
        new_path_to = path_to+patient_id
        if not os.path.exists(path_to):
            os.mkdir(path_to)
        if not os.path.exists(new_path_to):
            os.mkdir(new_path_to)

        # register image to MNI template
        golden_file_path = select_template_based_on_age(age)
        print("Registering to template:", golden_file_path,new_path_to)
        register_to_template(img_path, new_path_to, golden_file_path,"registered.nii.gz", create_subfolder=False)

        # enhance and zscore normalize image
        if not os.path.exists(new_path_to+"/no_z"):
            os.mkdir(new_path_to+"/no_z")
            
        # load image and enhance it
        image_sitk =  sitk.ReadImage(new_path_to+"/registered.nii.gz")
        image_array  = sitk.GetArrayFromImage(image_sitk)
        image_array = enhance_noN4(image_array)
        image3 = sitk.GetImageFromArray(image_array)

        # save enhanced image
        sitk.WriteImage(image3,new_path_to+"/no_z/registered_no_z.nii") 
        cmd_line = "zscore-normalize "+new_path_to+"/no_z/registered_no_z.nii -o "+new_path_to+'/registered_z.nii'
        subprocess.getoutput(cmd_line)    

        image_sitk = sitk.ReadImage(new_path_to+'/registered_z.nii')    
        windowed_images  = sitk.GetArrayFromImage(image_sitk)           

        # resize image to 256x256
        resize_func = functools.partial(resize, output_shape=model_selection.input_shape[1:3],
                                                    preserve_range=True, anti_aliasing=True, mode='constant')
        series = np.dstack([resize_func(im) for im in windowed_images])
        series = np.transpose(series[:, :, :, np.newaxis], [2, 0, 1, 3])
        series_n = []

        # create MIP of 5 slices = 5mm
        for slice_idx in range(2, np.shape(series)[0]-2):
            im_array = np.zeros((256, 256, 1, 5))
            
            im_array[:,:,:,0] = series[slice_idx-2,:,:,:].astype(np.float32)
            im_array[:,:,:,1] = series[slice_idx-1,:,:,:].astype(np.float32)
            im_array[:,:,:,2] = series[slice_idx,:,:,:].astype(np.float32)
            im_array[:,:,:,3] = series[slice_idx+1,:,:,:].astype(np.float32)
            im_array[:,:,:,4] = series[slice_idx+2,:,:,:].astype(np.float32)
                    
            im_array= np.max(im_array, axis=3)
                    
            series_n.append(im_array)
            series_w = np.dstack([funcy(im) for im in series_n])
            series_w = np.transpose(series_w[:, :, :, np.newaxis], [2, 0, 1, 3])
            
        # predict slice  
        predictions = model_selection.predict(series_w)
        slice_label = get_slice_number_from_prediction(predictions)
        middle_slice = slice_label
        N_thick = n_slices
        hd = 0
        
        print("Predicted slice:", slice_label)

        img = nib.load(new_path_to+'/registered_z.nii')  
        image_array, affine = img.get_fdata(), img.affine
        infer_seg_array_3d_1,infer_seg_array_3d_2 = np.zeros(image_array.shape),np.zeros(image_array.shape)
        infer_seg_array_3d_1_filtered,infer_seg_array_3d_2_filtered = np.zeros(image_array.shape),np.zeros(image_array.shape)
        infer_seg_array_3d_merged_filtered =  np.zeros(image_array.shape)
        
        if enable_3d:
            if middle_slice==0 or middle_slice==np.shape(image_array)[2]-1:
                #wrong slice number, handle gracefully
                print("Wrong slice number, skipping image")
                result = np.array([patient_id,float(age),gender,
                            0, 0, 0,
                            0, 0, 0,
                            slice_label, 0,n_slices,
                            0,0,0,0,0,0,0,0,0,0,0,0,0,
                            0,0,0,0,0,0,0,0,0,0,0,0,0,
                            0,0,0,0,0,0,0,0,0,0,0,0,0])
                
                df_results = pd.DataFrame([result], columns=['PatientID','Age','Gender',
                                                                'TMT1','TMT2','Centile_iTMT',
                                                                'CSA_TM1','CSA_TM2','Centile_iCSA',
                                                                "Slice_label","min_distance_btw_TMs","n_slices"
                                                                "volume1","volume_convexhull1","surface_area1","diameter_volume_equivalent1","diameter_surfacearea_equivalent1","width_3d_bb1","length_3d_bb1","height_3d_bb1","feret_3d_max1","feret_3d_min1","x_max_3d1","y_max_3d1","z_max_3d1",
                                                                "volume2","volume_convexhull2","surface_area2","diameter_volume_equivalent2","diameter_surfacearea_equivalent2","width_3d_bb2","length_3d_bb2","height_3d_bb2","feret_3d_max2","feret_3d_min2","x_max_3d2","y_max_3d2","z_max_3d2",
                                                                'volume","volume_convexhull","surface_area","diameter_volume_equivalent","diameter_surfacearea_equivalent","width_3d_bb","length_3d_bb","height_3d_bb","feret_3d_max","feret_3d_min","x_max_3d","y_max_3d","z_max_3d'])
                df_results.to_csv(path_to+"/"+patient_id+"_results.csv",index=False)
                continue
            else:
                slices = [middle_slice + i for i in range(-N_thick, N_thick + 1)]
                slices = sorted(slices, key=lambda x: abs(x - middle_slice))
        else:
            slices = [middle_slice]
            
        for slice_label in slices:
            #check if slice is within the image
            if slice_label<0 or slice_label>=np.shape(image_array)[2]:
                print("Slice out of bounds, skipping slice")
                continue
            image_array_2d = rescale(image_array[:,15:-21,slice_label], scaling_factor).reshape(1,target_size_unet[0],target_size_unet[1],1) 
            
            # create 4 images - half TMT and half empty           
            img_half_11 = np.concatenate((image_array_2d[:,:256,:,:],np.zeros_like(image_array_2d[:,:256,:,:])),axis=1)
            img_half_21 = np.concatenate((np.zeros_like(image_array_2d[:,:256,:,:]),image_array_2d[:,:256,:,:]),axis=1)
            img_half_12 = np.concatenate((np.zeros_like(image_array_2d[:,256:,:,:]),image_array_2d[:,256:,:,:]),axis=1)
            img_half_22 = np.concatenate((image_array_2d[:,256:,:,:],np.zeros_like(image_array_2d[:,256:,:,:])),axis=1)

            flipped = np.flip(image_array_2d, axis=1)

            flipped_11 = np.concatenate((flipped[:,:256,:,:],np.zeros_like(flipped[:,:256,:,:])),axis=1)
            flipped_21 = np.concatenate((np.zeros_like(flipped[:,:256,:,:]),flipped[:,:256,:,:]),axis=1)
            flipped_12 = np.concatenate((np.zeros_like(flipped[:,256:,:,:]),flipped[:,256:,:,:]),axis=1)
            flipped_22 = np.concatenate((flipped[:,256:,:,:],np.zeros_like(flipped[:,256:,:,:])),axis=1)

            list_of_left_muscle = [img_half_11, img_half_21, flipped_12, flipped_22]
            list_of_right_muscle = [img_half_12,img_half_22, flipped_11, flipped_21]

            list_of_left_muscle_preds = []
            list_of_right_muscle_preds = []

            # predict left and right muscle on each of 4 images
            for image in list_of_left_muscle: 
                infer_seg_array = model_unet.predict(image)
                muscle_seg = infer_seg_array[:,:,:,1].reshape(1,target_size_unet[0],target_size_unet[1],1)               
                list_of_left_muscle_preds.append(muscle_seg)
                                
            for image in list_of_right_muscle: 
                infer_seg_array = model_unet.predict(image)
                muscle_seg = infer_seg_array[:,:,:,1].reshape(1,target_size_unet[0],target_size_unet[1],1)             
                list_of_right_muscle_preds.append(muscle_seg)
                            
            list_of_left_muscle_preds_halved = [list_of_left_muscle_preds[0][:,:256,:,:],
                                                list_of_left_muscle_preds[1][:,256:,:,:],
                                                np.flip(list_of_left_muscle_preds[2][:,256:,:,:],axis=1),
                                                np.flip(list_of_left_muscle_preds[3][:,:256,:,:],axis=1)]

            list_of_right_muscle_preds_halved = [list_of_right_muscle_preds[0][:,256:,:,:],
                                                list_of_right_muscle_preds[1][:,:256,:,:],
                                                np.flip(list_of_right_muscle_preds[2][:,:256,:,:],axis=1),
                                                np.flip(list_of_right_muscle_preds[3][:,256:,:,:],axis=1)]
            
            # average predictions and threshold              
            left_half_result = np.mean(list_of_left_muscle_preds_halved, axis=0)<=threshold # <>
            right_half_result = np.mean(list_of_right_muscle_preds_halved, axis=0)<=threshold # <>
            muscle_seg_1 = np.concatenate((left_half_result,np.zeros_like(left_half_result)),axis=1)
            muscle_seg_2 = np.concatenate((np.zeros_like(left_half_result),right_half_result),axis=1)

            infer_seg_array_3d_1_filtered,infer_seg_array_3d_2_filtered = np.zeros(image_array.shape),np.zeros(image_array.shape)
            infer_seg_array_3d_merged_filtered =  np.zeros(image_array.shape)
                    
            # filter islands
            muscle_seg_1_filtered, area_1, cnt_1 = filter_islands(muscle_seg_1[0])
            muscle_seg_2_filtered, area_2, cnt_2 = filter_islands(muscle_seg_2[0])

            # save plots 
            fg = plt.figure(figsize=(5, 5), facecolor='k')
            I = cv2.normalize(image_array_2d[0], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imwrite(new_path_to+"/"+patient_id+"_"+str(slice_label)+"_no_masks.png", I)
            im = cv2.imread(new_path_to+"/"+patient_id+"_"+str(slice_label)+"_no_masks.png")                        
            im_copy = im.copy()        
            result = im.copy()
            
            for cont in [cnt_1,cnt_2]: 
                #check if contour is integer
                if type(cont) != int and len(cont)!=0:
                    if cv2.contourArea(cont) <= 1:
                        im_copy = cv2.drawContours(im_copy, [cont], -1, (0, 0, 255), -1)
                    else:
                        im_copy = cv2.drawContours(im_copy, [cont], -1, (51, 197, 255), -1)
            filled = cv2.addWeighted(im, alpha, im_copy, 1-alpha, 0)
            for cont in [cnt_1,cnt_2]: 
                if type(cont) != int and len(cont)!=0:
                    if cv2.contourArea(cont) <= 1:
                        result = cv2.drawContours(filled, [cont], -1, (0, 0, 255), 0)
                    else:
                        result = cv2.drawContours(filled, [cont], -1, (51, 197, 255), 0)

            cv2.imwrite(new_path_to+"/"+patient_id+"_"+str(slice_label)+"_mask.png", result)
                    
            # rescale for the unet
            infer_seg_array_2d_1_filtered = rescale(muscle_seg_1_filtered,1/scaling_factor)
            infer_seg_array_2d_2_filtered = rescale(muscle_seg_2_filtered,1/scaling_factor)

            # save to 3d
            infer_seg_array_3d_1_filtered[:,:,slice_label] = np.pad(infer_seg_array_2d_1_filtered[:,:,0],[[0,0],[15,21]],'constant',constant_values=0)
            infer_seg_array_3d_2_filtered[:,:,slice_label] = np.pad(infer_seg_array_2d_2_filtered[:,:,0],[[0,0],[15,21]],'constant',constant_values=0)
                        
            concated = np.concatenate((infer_seg_array_2d_1_filtered[:100,:,0],infer_seg_array_2d_2_filtered[100:,:,0]),axis=0)    
            infer_seg_array_3d_merged_filtered[:,:,slice_label] = np.pad(concated,[[0,0],[15,21]],'constant',constant_values=0)
            
            infer_3d_path = new_path_to+"/"+patient_id+"_"+str(slice_label)+'mask.nii.gz'
            if slice_label==slices[-1] or enable_3d==False:
                save_nii(infer_seg_array_3d_merged_filtered, infer_3d_path, affine)
                
            objL_pred_minf_line, objR_pred_minf_line, objL_pred_minf, objR_pred_minf = 0,0,0,0
                            
            crop_line = compute_crop_line(image_array[:,15:-21,slice_label],infer_seg_array_2d_1_filtered,infer_seg_array_2d_2_filtered)
                            
            if np.sum(infer_seg_array_3d_1_filtered[:100,:,slice_label])>2:
                objL_pred_minf = round(Calculater(infer_seg_array_3d_1_filtered[:100,:,slice_label], edge=True).minf,2)

            if np.sum(infer_seg_array_3d_2_filtered[100:,:,slice_label])>2:
                objR_pred_minf = round(Calculater(infer_seg_array_3d_2_filtered[100:,:,slice_label], edge=True).minf,2)
                        
            CSA_PRED_TM1 = np.sum(infer_seg_array_3d_1_filtered[:100,:,slice_label])
            CSA_PRED_TM2 = np.sum(infer_seg_array_3d_2_filtered[100:,:,slice_label])
                                
            if np.sum(infer_seg_array_3d_1_filtered[:100,int(crop_line):,slice_label])>2:
                objL_pred_minf_line = round(Calculater(infer_seg_array_3d_1_filtered[:100,int(crop_line):,slice_label], edge=True).minf,2)

            if np.sum(infer_seg_array_3d_2_filtered[100:,int(crop_line):,slice_label])>2:
                objR_pred_minf_line = round(Calculater(infer_seg_array_3d_2_filtered[100:,int(crop_line):,slice_label], edge=True).minf,2)
                            
            CSA_PRED_TM1_line = np.sum(infer_seg_array_3d_1_filtered[:100,int(crop_line):,slice_label])
            CSA_PRED_TM2_line = np.sum(infer_seg_array_3d_2_filtered[100:,int(crop_line):,slice_label])
            input_csa= (CSA_PRED_TM1_line+CSA_PRED_TM2_line)/2
            '''
            if objL_pred_minf > objR_pred_minf/2:
                input_tmt = objL_pred_minf
            elif objR_pred_minf>objL_pred_minf/2:
                input_tmt = objR_pred_minf
            else:
                input_tmt = (objL_pred_minf+objR_pred_minf)/2
            '''
            if objL_pred_minf >= objR_pred_minf * 1.5:
                input_tmt = objL_pred_minf
            elif objR_pred_minf >= objL_pred_minf * 1.5:
                input_tmt = objR_pred_minf
            else:
                input_tmt = (objL_pred_minf+objR_pred_minf)/2
                
            print("Age:",str(age)," Gender:",gender)
            print("iTMT[mm]:", input_tmt)
            print("Slice label:",slice_label)
            if slice_label==middle_slice:
                print(np.shape(infer_seg_array_2d_1_filtered))
                if np.sum(infer_seg_array_2d_1_filtered)>2 and np.sum(infer_seg_array_2d_2_filtered)>2:
                    hd = compute_distance_between_two_masks(image_array[:,15:-21,slice_label],infer_seg_array_2d_1_filtered,infer_seg_array_2d_2_filtered,
                                                            new_path_to+"/"+patient_id+"_"+str(slice_label)+'_contours.png')
                else:
                    hd=0
                        
            # centiles estimation
            df_centile_boys = pd.read_csv(df_centile_boys_csv,header=0)
            df_centile_girls = pd.read_csv(df_centile_girls_csv,header=0)
            df_centile_boys_csa = pd.read_csv(df_centile_boys_csv_csa,header=0)
            df_centile_girls_csa = pd.read_csv(df_centile_girls_csv_csa,header=0)
            
            if gender =='F' or gender=='Female' or gender=='f' or gender=='F':
                centile_tmt = find_exact_centile(input_tmt, round(float(age),2), df_centile_girls)
                centile_csa = find_exact_centile(input_csa, round(float(age),2), df_centile_girls_csa)
    
            else:
                centile_tmt = find_exact_centile(input_tmt, round(float(age),2), df_centile_boys)
                centile_tmt = find_exact_centile(input_csa, round(float(age),2), df_centile_boys_csa)
            print("iTMT Centile:",centile_tmt)  
            # save results
            #if enable 3d and its last slice in range:
            if enable_3d and slice_label==slices[-1]:
                m1,m2=feret_3d(infer_seg_array_3d_1_filtered, infer_seg_array_3d_2_filtered)
                #concat two df to the end of each other: m1 and m2
                #remove from m1 and m2 any additional rows except the first one
                
                m1 = m1.iloc[0]  # Keep only the first row of m1
                m2 = m2.iloc[0]
                #save to csv
                general_m1 = pd.concat([m1,m2],axis=0)
                general_m1.to_csv(new_path_to+"/"+patient_id+'_3d_mask_metrics.csv')
                #TODO: extract metrics from 3d masks
                #volume	volume_convexhull	surface_area	diameter_volume_equivalent	diameter_surfacearea_equivalent	width_3d_bb	length_3d_bb	height_3d_bb	feret_3d_max	feret_3d_min	x_max_3d	y_max_3d	z_max_3d
                result = np.array([patient_id,float(age),gender,
                            objL_pred_minf, objR_pred_minf, centile_tmt,
                            CSA_PRED_TM1_line, CSA_PRED_TM2_line, centile_csa,
                            slice_label, 
                            hd,n_slices,
                            m1['volume'],m1['volume_convexhull'],m1['surface_area'],m1['diameter_volume_equivalent'],m1['diameter_surfacearea_equivalent'],m1['width_3d_bb'],m1['length_3d_bb'],m1['height_3d_bb'],m1['feret_3d_max'],m1['feret_3d_min'],m1['x_max_3d'],m1['y_max_3d'],m1['z_max_3d'],
                            m2['volume'],m2['volume_convexhull'],m2['surface_area'],m2['diameter_volume_equivalent'],m2['diameter_surfacearea_equivalent'],m2['width_3d_bb'],m2['length_3d_bb'],m2['height_3d_bb'],m2['feret_3d_max'],m2['feret_3d_min'],m2['x_max_3d'],m2['y_max_3d'],m2['z_max_3d'],
                            (m1['volume']+m2['volume'])/2, (m1['volume_convexhull']+m2['volume_convexhull'])/2, (m1['surface_area']+m2['surface_area'])/2, (m1['diameter_volume_equivalent']+m2['diameter_volume_equivalent'])/2, (m1['diameter_surfacearea_equivalent']+m2['diameter_surfacearea_equivalent'])/2, (m1['width_3d_bb']+m2['width_3d_bb'])/2, (m1['length_3d_bb']+m2['length_3d_bb'])/2, (m1['height_3d_bb']+m2['height_3d_bb'])/2, (m1['feret_3d_max']+m2['feret_3d_max'])/2, (m1['feret_3d_min']+m2['feret_3d_min'])/2, (m1['x_max_3d']+m2['x_max_3d'])/2, (m1['y_max_3d']+m2['y_max_3d'])/2, (m1['z_max_3d']+m2['z_max_3d'])/2])  
                df_results = pd.DataFrame([result], columns=['PatientID','Age','Gender',
                                                            'TMT1','TMT2','Centile_iTMT',
                                                            'CSA_TM1','CSA_TM2','Centile_iCSA',
                                                            "Slice_label","min_distance_btw_TMs","n_slices",
                                                            'volume1','volume_convexhull1','surface_area1','diameter_volume_equivalent1','diameter_surfacearea_equivalent1','width_3d_bb1','length_3d_bb1','height_3d_bb1','feret_3d_max1','feret_3d_min1','x_max_3d1','y_max_3d1','z_max_3d1',
                                                            'volume2','volume_convexhull2','surface_area2','diameter_volume_equivalent2','diameter_surfacearea_equivalent2','width_3d_bb2','length_3d_bb2','height_3d_bb2','feret_3d_max2','feret_3d_min2','x_max_3d2','y_max_3d2','z_max_3d2',
                                                            'volume','volume_convexhull','surface_area','diameter_volume_equivalent','diameter_surfacearea_equivalent','width_3d_bb','length_3d_bb','height_3d_bb','feret_3d_max','feret_3d_min','x_max_3d','y_max_3d','z_max_3d'])
            elif enable_3d==True:
                result = np.array([patient_id,float(age),gender,
                            objL_pred_minf, objR_pred_minf, centile_tmt,
                            CSA_PRED_TM1_line, CSA_PRED_TM2_line, centile_csa,
                            slice_label, 
                            hd,n_slices,
                            0,0,0,0,0,0,0,0,0,0,0,0,0,
                            0,0,0,0,0,0,0,0,0,0,0,0,0,
                            0,0,0,0,0,0,0,0,0,0,0,0,0])
                df_results = pd.DataFrame([result], columns=['PatientID','Age','Gender',
                                                            'TMT1','TMT2','Centile_iTMT',
                                                            'CSA_TM1','CSA_TM2','Centile_iCSA',
                                                            "Slice_label","min_distance_btw_TMs","n_slices",
                                                            'volume1','volume_convexhull1','surface_area1','diameter_volume_equivalent1','diameter_surfacearea_equivalent1','width_3d_bb1','length_3d_bb1','height_3d_bb1','feret_3d_max1','feret_3d_min1','x_max_3d1','y_max_3d1','z_max_3d1',
                                                            'volume2','volume_convexhull2','surface_area2','diameter_volume_equivalent2','diameter_surfacearea_equivalent2','width_3d_bb2','length_3d_bb2','height_3d_bb2','feret_3d_max2','feret_3d_min2','x_max_3d2','y_max_3d2','z_max_3d2',
                                                             'volume','volume_convexhull','surface_area','diameter_volume_equivalent','diameter_surfacearea_equivalent','width_3d_bb','length_3d_bb','height_3d_bb','feret_3d_max','feret_3d_min','x_max_3d','y_max_3d','z_max_3d'])
            elif enable_3d==False:
                result = np.array([patient_id,float(age),gender,
                            objL_pred_minf, objR_pred_minf, centile_tmt,
                            CSA_PRED_TM1_line, CSA_PRED_TM2_line, centile_csa,
                            slice_label, 
                            hd])
                
                df_results = pd.DataFrame([result], columns=['PatientID','Age','Gender',
                                                            'TMT1','TMT2','Centile_iTMT',
                                                            'CSA_TM1','CSA_TM2','Centile_iCSA',
                                                            "Slice_label","min_distance_btw_TMs"])
            df_results.to_csv(path_to+"/"+patient_id+"_results.csv",index=False)
            print("Results saved to:",path_to+"/"+patient_id+"_results.csv")
            
    # concatenate all results .csv files into one
    all_files = glob.glob(path_to+"/*_results.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    frame.to_csv(path_to+"/results.csv",index=False)
    
    print("All results saved to:",path_to+"/results.csv")
    
                    
                    
