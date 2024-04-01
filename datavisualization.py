from extract_data import extract_data
from PIL import Image
import requests
from io import BytesIO
import zipfile
import pathlib 
import random
import os
from PIL import Image
import numpy as np
import random
import os 
import requests
import zipfile
from pathlib import Path
import numpy as np
import nibabel as nib
from skimage import io
import SimpleITK as sitk
from matplotlib.pyplot import plot 
import matplotlib.pyplot as plt
from io import BytesIO
    # try:
    #     url = extract_data()
    #     url_response = requests.get(url)
    #     url_response.raise_for_status()  # Raise an error for bad responses
    #     with zipfile.ZipFile(BytesIO(url_response.content)) as z:
    #         z.extractall('.')
    #     print("Data extraction successful.")
    # except requests.exceptions.RequestException as e:
    #     print("Error downloading data:", e)
    # except zipfile.BadZipFile:
    #     print("The downloaded file is not a valid zip file.")
    # except Exception as e:
    #     print("An unexpected error occurred:", e)

def filter_dot_files(file_list):
    return [filename for filename in file_list if filename.startswith('.')]

def open_random_images(path):
    # Get a list of all files in the folder
    all_files = os.listdir(path)
    dropfiles = filter_dot_files(all_files)
    img_files = [x for x in all_files if x not in dropfiles]
    print(img_files)
    random.shuffle(img_files)
    image_names = img_files[:]
    image_paths = []
    for i in range(len(image_names)):
        image_path = os.path.join(path, image_names[i])
        image_paths.append(image_path)
    return image_paths


def visualise_image():
    try:
        url = extract_data()
        url_response = requests.get(url)
        url_response.raise_for_status()  # Raise an error for bad responses
        with zipfile.ZipFile(BytesIO(url_response.content)) as z:
            z.extractall('.')
        print("Data extraction successful.")
    except requests.exceptions.RequestException as e:
        print("Error downloading data:", e)
    except zipfile.BadZipFile:
        print("The downloaded file is not a valid zip file.")
    except Exception as e:
        print("An unexpected error occurred:", e)
    fullpath = os.path.join(os.getcwd(),"workspace/data/spleen/Task09_Spleen/imagesTr")
    y = open_random_images(fullpath)
    count = 4
    x = [random.choice(y) for _ in range(4)]
    for file in x:
        filearr = file.split("_")[-1]
        filename = filearr.split(".")[0]
        # filename = "spleen_1_seg.nii.gz"
        #filename = "/home/ubuntu/deeplearning/monai/MONAI/monai_training_dir/workspace/data/spleen/Task09_Spleen/imagesTr/spleen_10.nii.gz"
        # Read the .nii image containing the volume with SimpleITK
        # sitk_t1 = sitk.ReadImage(file)
        # # and access the numpy array:
        # t1 = sitk.GetArrayFromImage(sitk_t1)
        # print(t1.shape)
        # for y in range(len(t1[0])):
        #     im = Image.fromarray((t1[y,:,:] ).astype(np.uint8))
        #     print(im)
        # Load the .nii.gz file
    
        img = nib.load(file)
        # Get the image data
        img_data = img.get_fdata()
        # Display one of the image slices (you can change the slice index)
        slice_index = img_data.shape[2] // 2  # Choose a slice index in the z-axis
        slice_image = img_data[:, :, slice_index]

        # Normalize the image data to the range [0, 255]
        slice_image = (slice_image - slice_image.min()) / (slice_image.max() - slice_image.min()) * 255

        # Convert the image data to an unsigned 8-bit integer array
        slice_image = slice_image.astype('uint8')
        img_png = Image.fromarray(slice_image)
        img_png.save(f'{filename}.png')
        count=count+1
        #plt.imshow(img_data[:, :, slice_index], cmap='gray')
        plt.axis('off')  # Turn off axis
        plt.show()




    # glioma_tumor_images = os.listdir(os.path.join(os.getcwd(),"Training/glioma_tumor"))
    # meningioma_tumor_images = os.listdir(os.path.join(os.getcwd(),"Training/meningioma_tumor"))
    # no_tumor_images = os.listdir(os.path.join(os.getcwd(),"Training/meningioma_tumor"))
    # pituitory_tumor_images = os.listdir(os.path.join(os.getcwd(),"Training/pituitary_tumor"))
    # path = pathlib.Path(os.path.join(os.getcwd(),"Training"))
    # def open_random_image(path):
    #     # Get a list of all files in the folder
    #     all_files = os.listdir(path)
    #     random_image_file = random.choice(all_files)
    #     image_path = os.path.join(path, random_image_file)
    #     image = Image.open(image_path)
    #     return image
    # glioma_tumor_image = open_random_image(os.path.join(os.getcwd(),"Training/glioma_tumor"))
    # meningioma_tumor_image = open_random_image(os.path.join(os.getcwd(),"Training/meningioma_tumor"))
    # no_tumor_image = open_random_image(os.path.join(os.getcwd(),"Training/no_tumor"))
    # pituitory_tumor_image = open_random_image(os.path.join(os.getcwd(),"Training/pituitary_tumor"))
    # glioma_tumor_image.save('glioma_tumor.jpg')
    # meningioma_tumor_image.save('meningioma_tumor.jpg')
    # no_tumor_image.save('no_tumor.jpg')
    # pituitory_tumor_image.save('pituitory_tumor.jpg')
    #return path,glioma_tumor_images,meningioma_tumor_images,no_tumor_images,pituitory_tumor_images


visualise_image()