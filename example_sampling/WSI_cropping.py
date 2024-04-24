import os
from os import listdir, mkdir, path, makedirs
from os.path import join
import openslide as slide
from PIL import Image
import numpy as np
import pandas as pd
from skimage import data, io, transform
from skimage.color import rgb2gray, rgb2hsv
from skimage.util import img_as_ubyte, view_as_windows
from skimage import img_as_ubyte
import time, sys, warnings, glob
import threading
import multiprocessing
from tqdm import tqdm
from xml.etree.ElementTree import parse
import shapely.geometry as shgeo
import argparse, pickle, random
warnings.simplefilter('ignore')


def parse_filename_from_directory(input_file_list):
    output_file_list = [os.path.basename(os.path.splitext(item)[0]) for item in input_file_list]
    return output_file_list

def thres_saturation(img, t=15):
    img = rgb2hsv(img)
    h, w, c = img.shape
    sat_img = img[:, :, 1]
    sat_img = img_as_ubyte(sat_img)
    ave_sat = np.sum(sat_img) / (h * w)
    return ave_sat >= t


def crop_slide(img, save_slide_path, position=(0, 0), step=(0, 0), patch_size=224, scale=10, down_scale=1): # position given as (x, y) at nx scale
    patch_name = "{}_{}".format(step[0], step[1])

    img_nx_path = join(save_slide_path, f"{patch_name}-tile-r{position[1] * down_scale}-c{position[0] * down_scale}-{patch_size}x{patch_size}.png")
    if path.exists(img_nx_path):
        return 1

    img_x = img.read_region((position[0] * down_scale, position[1] * down_scale), 0, (patch_size * down_scale, patch_size * down_scale))
    img_x = np.array(img_x)[..., :3]
    #if down_scale!=1:
    img = transform.resize(img_x, (img_x.shape[0] // down_scale, img_x.shape[0] // down_scale), order=1,  anti_aliasing=False)
    if thres_saturation(img, 80): # -1 for all
        try:
            io.imsave(img_nx_path, img_as_ubyte(img))
        except Exception as e:
            print(e)

def slide_to_patch(out_base, img_slides, patch_size, step_size, scale, down_scale=1):
    makedirs(out_base, exist_ok=True)
    for s in tqdm(range(len(img_slides))):
        img_slide = img_slides[s]
        img_name = img_slide.split(path.sep)[-1].split('.')[0]
        bag_path = join(out_base, img_name)

        makedirs(bag_path, exist_ok=True)
        img = slide.OpenSlide(img_slide)
        print("slide_to_patch --> ", img_name, img.dimensions, img.level_dimensions, img.level_count, img.properties['openslide.mpp-x'])

        try:
            if int(np.floor(float(img.properties['openslide.mpp-x'])*10)) == 2:
                down_scale = (40 // scale)
            else:
                down_scale = (20 // scale)
        except Exception as e:
            print("tiff --> No properties 'openslide.mpp-x'")

        dimension = img.level_dimensions[0]
        # dimension and step at given scale
        #print(dimension,down_scale)
        step_y_max = int(np.floor(dimension[1]/(step_size*down_scale))) # rows
        step_x_max = int(np.floor(dimension[0]/(step_size*down_scale))) # columns
        print("number :", step_x_max, step_y_max, step_x_max*step_y_max)
        num =  step_x_max*step_y_max
        count = 0
        for j in range(step_y_max):
            for i in range(step_x_max):
                start_time = time.time()
                crop_slide(img, bag_path, (i*step_size, j*step_size), step=(j, i), patch_size=patch_size, scale=scale, down_scale=down_scale)
                end_time = time.time()
                count += 1
                print(f"{count}/{num}", (end_time-start_time)/60)

def get_patch_size(pst_lt, pst_rb):
    row_start, col_start = pst_lt
    row_end, col_end = pst_rb
    return (col_end - col_start) * (row_end - row_start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop the WSIs into patches')
    parser.add_argument('--num_threads', type=int, default=16, help='Number of threads for parallel processing, too large may result in errors')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap pixels between adjacent patches')
    parser.add_argument('--patch_size', type=int, default=1120, help='Patch size')
    parser.add_argument('--scale', type=int, default=20, help='20x 10x 5x')
    parser.add_argument('--dataset', type=str, default='./dataset', help='Dataset folder name')
    parser.add_argument('--output', type=str, default='./result/tiled_patches', help='Output folder name')
    parser.add_argument('--display', default=False, help='Display patch numbers under this setting')
    parser.add_argument('--annotation', default=False, help='Obtain patches in annotation region')
    args = parser.parse_args()

    print('Cropping patches, this could take a while for big dataset, please be patient')
    step = args.patch_size - args.overlap

    # obtain dataset paths
    path_base = args.dataset
    out_base = args.output
    print("Dataset path", path_base)
    if path.isdir(path_base):
        all_slides = glob.glob(f"{path_base}/*.svs") + \
                     glob.glob(f"{path_base}/*.tif") + \
                     glob.glob(f"{path_base}/*.tiff") + \
                     glob.glob(f"{path_base}/*.mrxs") + \
                     glob.glob(f"{path_base}/*.ndpi")
    elif path.isfile(path_base):
        df = pd.read_csv(path_base)
        all_slides = df.Slide_Path.values.tolist()
    else:
        raise ValueError(f'Please check dataset folder {path_base}')

    print("Number of .svs .mrxs .ndpi .tif/f", len(all_slides))

    each_thread = int(np.floor(len(all_slides)/args.num_threads))
    threads = []
    for i in range(args.num_threads):
        if i < (args.num_threads-1):
            t = threading.Thread(target=slide_to_patch, args=(out_base, all_slides[each_thread*i:each_thread*(i+1)], args.patch_size, step, args.scale))
        else:
            t = threading.Thread(target=slide_to_patch, args=(out_base, all_slides[each_thread*i:], args.patch_size, step, args.scale))
        threads.append(t)

    for thread in threads:
        thread.start()
