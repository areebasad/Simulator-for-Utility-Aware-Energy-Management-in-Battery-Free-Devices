from PIL import Image, ImageFile
import image_slicer 
import io                      # for image2byte array
import numpy as np
import pandas as pd
from math import sqrt, ceil, floor
import os                     # for timestamp function 


def image_to_byte_array(image:Image):
    
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='png')
    imgByteArr = imgByteArr.getvalue()
    
    return imgByteArr


def nlargest_indices(arr:np.ndarray, n:int) -> np.ndarray:

    indices = np.argsort(arr, axis=None)               # Sort indices in assending order 
    x, y = np.unravel_index(indices, arr.shape)        # get x and y 
    x = x[::-1]                                        # Reverse array, since it is in ascending order, make it in descending order
    y = y[::-1]                                        # Reverse array, since it is in ascending order, make it in descending order
    
    return x[0:n],y[0:n]

def adjust_indices_for_slicerPackage(x:np.ndarray, y:np.ndarray) -> list:
    
    x = list(map(lambda x:x+1, x)) # We increament +1 to all since array index starts with 0,0 while image slicer index with 1,1
    y = list(map(lambda x:x+1, y)) # We increament +1 to all since array index starts with 0,0 while image slicer index with 1,1
    indices = list(zip(y, x))  # We swap x,y to y,x since slicer package takes input as column, row
    
    return indices


def select_tiles (all_image_tiles, indices:list):
    
    selected_tiles = []
    
    for tile in all_image_tiles:
        if(tile.position in indices):
            selected_tiles.append(tile)
            #ic(tile.image)
    
    return selected_tiles   


# For testing and visualization
def display_tiles(tiles):
    #ic(tiles)
    for tile in tiles:
        tile.image.show() 
        
        
def make_timestamp(image_path:str, camera_name: str):
    # Get timestamp from img name. TODO: move out of the loop. All these detections come from the same image, so timestamp is the same
    # splits path into folders and file and keep the last one
    # E.g.: ('../images/ie/idi/norwai/svv/ski-trd/jervskogen_1_2021-12-11_09-00-03.png') -> jervskogen_1_2021-12-11_09-00-03.png'
    _, stamp = os.path.split(image_path)

    # Remove camera name
    # E.g. 'jervskogen_1_2021-12-11_09-00-03.png' -> '2021-12-11_09-00-03.png'
    stamp = stamp.replace(camera_name + '_', '')

    # Remove file extension
    # E.g. '2021-12-11_09-00-03.png' -> '2021-12-11_09-00-03'
    stamp = stamp.replace('.png', '')

    # Split between date and time
    # E.g. '2021-12-11_09-00-03' -> '2021-12-11' + '09-00-03'
    date, time = stamp.split('_')

    # Replace hiffen in time with colon
    # E.g. '09-00-03' -> '09:00:03'
    time = time.replace('-', ':')

    # Finally, build the final timestamp string
    # E.g. '2021-12-11' + 'T' + '09:00:03' -> '2021-12-11T09:00:03'
    stamp = date + 'T' + time
    
    return stamp


"""
This function overlaps the given/selected tiles on the
reference or background image.
"""
def overlap_selected_tiles_on_background_image(
    tiles_to_overlap,
    total_number_of_tiles: int,
    reference_image_path: str = None,
    
):

    #reference_image_path = './images/jervskogen_1_2021-12-11_11-30-03.png'

    # Slice image into n parts/tiles
    reference_image_tiles = image_slicer.slice(reference_image_path, total_number_of_tiles, save=False)  # accepts even number

    reference_image_tiles = list(reference_image_tiles)  # Convert to list
    # ic(reference_image_tiles)

    for index, tile in enumerate(reference_image_tiles):
        for tile_to_overlap in tiles_to_overlap:
            if (tile.number is tile_to_overlap.number):
                reference_image_tiles[index] = tile_to_overlap  # replace reference tile with the tile to overlap
    
    
    overlapped_tiles_image = reference_image_tiles                             
   
    # Join patches/tiles of image
    overlapped_joined_image = image_slicer.join(overlapped_tiles_image)

    # display image
    #overlapped_joined_image.show()

    return overlapped_joined_image