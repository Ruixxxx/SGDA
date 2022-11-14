from lungmask import mask
import SimpleITK as sitk
import glob
import os
from tqdm import tqdm
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def convert_itk_image(path):
    """convert pn9 dataset to itk images
    """
    new_path = path.replace('images', 'newimages')
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    filelist = glob.glob(path + '*')
    filelist.sort()
    for i in tqdm(filelist, dynamic_ncols=True):
        itkimage = sitk.ReadImage(i)
        numpyImage = sitk.GetArrayFromImage(itkimage).squeeze()

        newimage = sitk.GetImageFromArray(numpyImage)
        newfile = sitk.WriteImage(newimage, os.path.join(new_path, os.path.split(i)[1]))
    return


def unit8_2_HU(image, HU_min=-1200.0, HU_max=600.0, HU_nan=-2000.0):
    """
    Convert HU unit into uint8 values. First bound HU values by predefined min
    and max, and then normalize
    image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
    HU_min: float, min HU value.
    HU_max: float, max HU value.
    HU_nan: float, value for nan in the raw CT image.
    """
    image_new = np.array(image)
    image_new = image_new.astype('float')

    # normalize to [0, 1]
    image_new = (image_new - 0) / 255
    # level = (HU_min + HU_max) / 2
    window = HU_max - HU_min
    image_new = image_new * window + HU_min

    return image_new

def lung_mask(path):
    """get lung mask
    """
    new_path = path.replace('images', 'masks_1')
    # new_path = path.replace('test', 'masks') #npy
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    filelist = glob.glob(path + '*')
    filelist.sort()
    for i in tqdm(filelist, dynamic_ncols=True):
        itkimage = sitk.ReadImage(i)
        numpyImage = sitk.GetArrayFromImage(itkimage)
        # numpyImage = np.load(i).squeeze()
        # newimage = unit8_2_HU(numpyImage)
        segmentation = np.zeros_like(numpyImage)
        for j in range(numpyImage.shape[0]):
            segmentation[j] = mask.apply(numpyImage[j], noHU=True)
        # newimage = sitk.GetImageFromArray(newimage)
        origin = itkimage.GetOrigin()
        direction = itkimage.GetDirection()
        spacing = itkimage.GetSpacing()
        # segmentation = mask.apply(newimage)

        newfile = sitk.GetImageFromArray(segmentation)
        newfile.SetOrigin(origin)
        newfile.SetDirection(direction)
        newfile.SetSpacing(spacing)
        newfile = sitk.WriteImage(newfile, os.path.join(new_path, os.path.split(i)[1]))
    return

def convert_img2npy(path):
    """convert pn9 itk images to npys
    """
    new_path = path.replace('images', 'preprocessed')
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    filelist = glob.glob(path + '*')
    filelist.sort()
    for i in tqdm(filelist, dynamic_ncols=True):
        itkimage = sitk.ReadImage(i)
        numpyImage = sitk.GetArrayFromImage(itkimage)
        np.save(os.path.join(new_path, os.path.split(i)[1].replace('.nii.gz', '_clean.npy')), numpyImage)

    return

def main():
    # convert_itk_image('/data1/pn9/train/images/')
    # convert_itk_image('/data1/pn9/test/images/')

    datasets_list = ['luna16', 'russia', 'tianchi']

    for dataset in datasets_list:
        if dataset == 'luna16':
            lung_mask('/home/xurui/data0/luna16/images/')
        # if dataset == 'pn9':
        #     lung_mask('/data0/pn9/train/newimages/')
        #     lung_mask('/data0/pn9/test/newimages/')
        if dataset == 'russia':
            lung_mask('/home/xurui/data0/russia/images/')
        if dataset == 'tianchi':
            lung_mask('/home/xurui/data0/tianchi/images/')
    # convert_img2npy('/data1/xurui/pn9/images/')

if __name__=='__main__':
    main()