# Author: Anupam Kumar
# Last Updated: 03/24/2019
# Description: Convert .xml file output exported using Export ROI XML plugin from OsiriX manual annotation software for clinical annotation purpose
# Features added:
# # March 24, 2019: Multi-Core processng for processing multiple files at once
# #

import os
import nibabel as nib
import numpy as np
import difflib
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d
import argparse
import matplotlib.pyplot as plt
import shutil
from scipy import ndimage
import time
import multiprocessing

def get_organ_label(organ_name):
    list_of_organs = ["lt parasp", "rt parasp", "lt psoas", "rt psoas", "aorta"]
    organ = difflib.get_close_matches(organ_name, list_of_organs)[0]
    label_number = list_of_organs.index(organ) + 1
    return label_number

def linterp_data(input_array):
    i_num = 21
    num_elements = len(input_array)
    output_array = np.zeros(i_num * num_elements)
    for i in range(0, len(input_array)):
        if i != num_elements - 1:
            values = np.linspace(input_array[i], input_array[i + 1], i_num)
        if i == num_elements - 1:
            values = np.linspace(input_array[i], input_array[0], i_num)
        start = (i * 10)
        end = (i * 10 + i_num)
        indices = list(range(start, end))
        output_array[indices] = values.astype(int)

    return output_array

def surrounded(x, y, img):
    row = img[x, :]
    col = img[:, y]

    nz_r = np.where(row != 0)[0]
    nz_c = np.where(col != 0)[0]

    row_h = len((np.where(nz_r > y)[0]))
    row_l = len((np.where(nz_r < y)[0]))
    col_h = len((np.where(nz_c > x)[0]))
    col_l = len((np.where(nz_c < x)[0]))

    if row_h and row_l and col_h and col_l:
        return 1
    else:
        return 0


def surrounded2(x, y, img):
    row = img[x, :]
    col = img[:, y]

    nz_r = np.where(row != 0)[0]
    nz_c = np.where(col != 0)[0]

    row_h = len((np.where(nz_r > y)[0]))
    row_l = len((np.where(nz_r < y)[0]))
    col_h = len((np.where(nz_c > x)[0]))
    col_l = len((np.where(nz_c < x)[0]))

    rh_status = row_h > 0
    rl_status = row_l > 0
    ch_status = col_h > 0
    cl_status = col_l > 0

    status = rh_status + rl_status + ch_status + cl_status

    if status > 3:
        return 1
    else:
        return 0

def fill_slice_contour(img, label):
    dim = img.shape
    for x in range(0, dim[0]):
        for y in range(0, dim[1]):
            curr_val = np.rint(img[x, y])
            if curr_val == 0:
                # not on boundary
                if surrounded(x, y, img):
                    img[x, y] = label
    return img

def fix_mask(img, label):
    dim = img.shape
    for x in range(0, dim[0]):
        for y in range(0, dim[1]):
            curr_val = np.rint(img[x, y])
            if curr_val == 0:
                # not on boundary
                if surrounded2(x, y, img):
                    img[x, y] = label
    return img

def generate_mask(xml_file, dimensions):
    
    #initialize variable for labels
    label = 0
    
    # empty mask file:
    mask_file = np.zeros(dimensions)
    
    tree = ET.parse(xml_file)
    plist = tree.getroot()

    for dict in plist:
        for sublevels in dict:

            progress_index = 1

            for sub_sublevels in sublevels:

                print('Progress: ', str(progress_index), ' of ', str(len(sublevels)))
                progress_index = progress_index + 1

                counter1 = 1
                counter_z = 0
                z = 0

                for sub2_sublevels in sub_sublevels:

                    if sub2_sublevels.text == 'ImageIndex':
                        # This is the z location of the mask
                        counter_z = counter1

                    if counter1 == counter_z + 1:
                        # get z value of slice
                        if sub2_sublevels.tag == 'integer':
                            z = dimensions[2] - int(sub2_sublevels.text)

                    counter1 = counter1 + 1
                    # now check for label name & mask pixels
                    if sub2_sublevels.tag == 'array':
                        for ROI_elem in sub2_sublevels:
                            total_options = len(ROI_elem)

                            counter2 = 0
                            counter_name = 0
                            counter3 = 0
                            counter_mask_points = 0

                            for ROI_dict in ROI_elem:
                                if ROI_dict.text == 'Name':
                                    counter_name = counter2
                                if counter2 == counter_name + 1:
                                    if ROI_dict.tag == 'string':
                                        label = get_organ_label(ROI_dict.text)
                                if ROI_dict.text == 'Point_px':
                                    counter_mask_points = counter3
                                if counter3 == counter_mask_points +1:
                                    if ROI_dict.tag == 'array':

                                        # define empty slice data...
                                        curr_slice_x = []
                                        curr_slice_y = []
                                        curr_slice_xy = []

                                        # proceed to get mask outline information
                                        counter_points = 0
                                        for line in ROI_dict:
                                            point = line.text
                                            point = point.split(", ")
                                            x = int(np.rint(float(point[0].replace("(", ""))))
                                            y = int(dimensions[1] - np.rint(float(point[1].replace(")", ""))))

                                            # add values to curr slice mask
                                            curr_slice_x.append(x)
                                            curr_slice_y.append(y)
                                            curr_slice_xy.append([x,y])

                                            counter_points = counter_points + 1

                                        # Interpolate the slice data..
                                        i = np.arange(len(curr_slice_xy))
                                        interp_i = np.linspace(0, i.max(), 10*i.max())
                                        curr_slice_x = interp1d(i, curr_slice_x, kind='cubic')(interp_i)
                                        curr_slice_y = interp1d(i, curr_slice_y, kind='cubic')(interp_i)
                                        #
                                        # # Interpolate the slice data..
                                        curr_slice_x = linterp_data(curr_slice_x)
                                        curr_slice_y = linterp_data(curr_slice_y)

                                        # Modify the mask file given all the points at current z
                                        slice = np.zeros([dimensions[0], dimensions[1]])

                                        for index in range(0, len(curr_slice_x)):
                                            x = int(curr_slice_x[index])
                                            y = int(curr_slice_y[index])
                                            slice[x,y] = label

                                        # Fill up slice contours
                                        slice = fill_slice_contour(slice, label)
                                        slice = fix_mask(slice, label)

                                        # Update mask file
                                        mask_file[:, :, z] = mask_file[:, :, z] + slice

                                counter2 = counter2 + 1
                                counter3 = counter3 + 1
    return mask_file

def QA(bg, overlay, temp_dir, QA_file_path, curr_dir):

    bg = nib.load(bg)
    bg_img = bg.get_fdata()
    overlay = nib.load(overlay)
    ol_img = overlay.get_fdata()

    # write file name to text file
    montage_list = os.path.join(curr_dir, 'temp_list.txt')
    fid = open(montage_list, 'w+')

    dim = bg.shape
    for slice_num in reversed(range(0, dim[2])):
        bg_img_slice = ndimage.rotate(bg_img[:,:,slice_num], 90)
        ol_img_slice = ndimage.rotate(ol_img[:,:,slice_num], 90)

        plt.imshow(bg_img_slice, 'gray')
        plt.imshow(ol_img_slice, 'hot', alpha=0.5, vmin=0., vmax=5)
        plt.axis('off')

        img_fname = str(int(slice_num)) + '.png'
        img_fname = os.path.join(temp_dir, img_fname)

        plt.savefig(img_fname, bbox_inches='tight', pad_inches = 0, transparent=False, facecolor='black')

        img_fname = img_fname
        # write image name to montage...
        fid.write(img_fname)
        fid.write("\n")

    fid.close()

    cmd = 'montage @' + montage_list + ' ' + QA_file_path
    os.system(cmd)
    cmd = 'rm -rf ' + montage_list
    os.system(cmd)
    cmd = 'rm -rf ' + temp_dir + '/*'
    os.system(cmd)

def process_dir(root_dir, out_dir, QA_dir, folder):
    # Get information about xml file of interest
    xml_dir = os.path.join(root_dir, 'xml')
    xml_file = os.path.join(xml_dir, os.listdir(xml_dir)[0])

    # Process all dicom files into nifti volume
    dicom_dir = os.path.join(root_dir, 'dicom')
    dicom_subdir_name = os.listdir(dicom_dir)[0]
    dicom_subdir = os.path.join(dicom_dir, dicom_subdir_name)

    # Base output directory
    base_out_dir = os.path.join(out_dir, folder)

    if not os.path.exists(base_out_dir):
        os.mkdir(base_out_dir)

    # Setup directory to save nifti file after dicom conversion
    nifti_dir = os.path.join(out_dir, folder, 'nifti')

    if not os.path.exists(nifti_dir):
        os.mkdir(nifti_dir)

    # Using external Linux based dcm2niii tool...
    if len(os.listdir(nifti_dir)) == 0: # if nifti directory is empty...
        cmd = 'dcm2nii -o ' + nifti_dir + ' ' + dicom_subdir
        os.system(cmd)

    # Get name of output nifti file
    nifti_file = os.path.join(nifti_dir, os.listdir(nifti_dir)[0])

    # Force image to be in radiological orientation
    cmd = 'fslorient -forceradiological ' + nifti_file
    os.system(cmd)

    # Use file name of volume which is to be labeled
    nifti_name = os.path.basename(nifti_file)

    # Setup label file names...
    label_dir = os.path.join(out_dir, folder, 'nifti_label')
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)

    # Build final nifti file using original nifti file..
    label_fname = nifti_name.replace('.nii', '_label.nii')
    label_file = os.path.join(label_dir, label_fname)

    # Extract data from xml and label masks 1...8 depending on mask category
    if not os.path.exists(label_file):
        nifti = nib.load(nifti_file)
        dim = nifti.header.get_data_shape()
        label_data = generate_mask(xml_file,dim)
        label_nifti = nib.Nifti1Image(label_data, nifti._affine)
        nib.save(label_nifti, label_file)

    # QA the output nifti and label using montage..
    QA_fname = folder + '.png'
    QA_file_path = os.path.join(QA_dir, QA_fname)

    # define new tempdir
    curr_dir = os.getcwd()
    temp_dir = os.path.join(curr_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    if not os.path.exists(QA_file_path):
        print('Performing QA of desired file..')
        QA(nifti_file, label_file, temp_dir, QA_file_path, curr_dir)

parser = argparse.ArgumentParser(description='Convert OsiriX XML ROI data to NIFTI volumes')

# define command line input (CLI) parameters
parser.add_argument('--src', type=str, help='The directory where the xml and dcm volume data are stored as per required method...')
parser.add_argument('--out_dir', type=str, help='This the directory where you wish to save all outputs of the process')
parser.add_argument('--QA_dir', type=str, help='This is the directory where you wish to save all the QA output')

args = parser.parse_args()

src_dir = args.src
out_dir = args.out_dir
QA_dir = args.QA_dir
dir_list = os.listdir(src_dir)

def event_manager(index):
    folder = dir_list[index]
    root_dir = os.path.join(src_dir, folder)
    process_dir(root_dir, out_dir, QA_dir, folder)


p = multiprocessing.Pool(processes=4)
p.map(event_manager, range(len(dir_list)))