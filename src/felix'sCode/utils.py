import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import statistics
from tqdm import tqdm

####Radial Distortion

"""
radial_distortion

Given k1,k2,k3, will create a distorted image with missing information
returns the distorted image and an arrary that records the unmapped points
"""
def radial_distortion(img,distortion_k,x_optic=None,y_optic=None,x_focal=None,y_focal=None):
    ## Setting up basic paramertersabs
    x_dim = img.shape[0]
    y_dim = img.shape[1]
    img_distorted = np.zeros(img.shape).astype(np.uint8)
    img_distorted_bool_array = np.zeros((x_dim,y_dim)).astype(np.uint8)

    visited=set()
    ## assumed the center to be the optical center, can be changed later
    if x_optic is None:
        x_optic= x_dim//2
    if y_optic is None:
        y_optic= y_dim//2
    if x_focal is None:
        x_focal=x_dim//2
    if y_focal is None:
        y_focal= y_dim//2
    ## unpacking the distortion coefficients
    k1,k2,k3= distortion_k
    r_max = np.sqrt(2) ## since x_norm <=1 and y_norm <=1
    scale = 1+ k1*(r_max**2) + k2*(r_max**4) + k3*(r_max**6)

    for x in range (x_dim):
        for y in range (y_dim):
        #Normalized image coordinates are calculated from pixel coordinates by translating to the optical center and dividing by the focal length in pixels.
            x_norm = (x-x_optic)/x_focal ## value will be between 0~1
            y_norm = (y-y_optic)/y_focal ## value will be between 0~1
            r = np.sqrt(x_norm**2 + y_norm**2)         # r= x**2 + y**2 
            x_dist_norm = x_norm*(1+k1*(r**2) + k2*(r**4) + k3*(r**6))/scale
            y_dist_norm = y_norm*(1+k1*(r**2) + k2*(r**4) + k3*(r**6))/scale
            x_distorted = int(x_dist_norm*x_optic + x_optic) 
            y_distorted = int(y_dist_norm*y_optic + y_optic) 
            img_distorted_bool_array[x_distorted][y_distorted]=1
            if (x_distorted,y_distorted) not in visited:
                visited.add((x_distorted,y_distorted))
                img_distorted[x_distorted][y_distorted]=img[x][y]
           
            
    name="k1_"+str(distortion_k[0])+"k2_"+str(distortion_k[1])+"k3_"+str(distortion_k[2])+"unfixed_radial.png"
    save_image(img_distorted,name)      
    return img_distorted,img_distorted_bool_array
"""
get_unmapped_points(img_distorted_bool_array)

Given the bool array from radial distortion, output the indices of unmapped points
"""
def get_unmapped_points(img_distorted_bool_array):
    ## Gets unmapped points
    indices=np.where(img_distorted_bool_array==[0])
    coordinates = zip(indices[0], indices[1])
    unmapped_points=list(coordinates)
    return unmapped_points

"""
save_image(img,name)

Save image with name
"""
def save_image(img,name):
    im = Image.fromarray(img)
    im.save(name)
"""
get_neighbors_median(x,y,image,k=1)

Get the median pixel of a given point x,y of a certain image with depth k=1
"""
def get_neighbors_median(x,y,image,k=1):
    def is_valid(x,y):
        x_dim=image.shape[0]
        y_dim=image.shape[1]
        return x<x_dim and x >=0 and y <y_dim and y >=0
    neighbors=[(x+k,y),(x-k,y),(x,y+k),(x,y-k),(x+k,y+k),(x-k,y-k),(x-k,y+k),(x+k,y-k)]
    valid_neighbor=[]
    mean_value=0
    r=[]
    g=[]
    b=[]
    for (x,y) in neighbors:
        if is_valid(x,y):
            valid_neighbor.append((x,y))
            r.append(int(image[x,y][0]))
            g.append(int(image[x,y][1]))
            b.append(int(image[x,y][2]))
    if len(r)==0 or len(g)==0 or len(b)==0:
        return np.zeros(3).astype("uint8")
    #mean_value//=len(valid_neighbor)
    r=np.uint8(statistics.median(np.array(r)))
    g=np.uint8(statistics.median(np.array(g)))
    b=np.uint8(statistics.median(np.array(b)))
    #print(r,g,b)
    
    
    res=np.array([r,g,b])
        
    return res

"""
get_non_zero_median(x,y,image):

Will continue to run get_neighbors_median until the surrounding neighbor is not black
"""
def get_non_zero_median(x,y,image,threshold=5):
    if image[x,y].any()!=0:
        return image[x,y]
    k=1
    temp=image.copy()
    while True:
        res=get_neighbors_median(x,y,temp,k)
        if res.any()!=0:
            return res
        k+=1
        if k>=threshold:
            return res

"""
remove_black_dots
Run get_non_zero_median to fix the whole image
"""
def remove_black_dots(unmapped_points,image):
    temp=image.copy()
    for x,y in unmapped_points:
        temp[x,y]=get_non_zero_median(x,y,temp)
    return temp

"""
radial_distortion_fixed
create a radial distortion image that is fixed
"""
def radial_distortion_fixed(img,distortion_k,iterations=5):
    img_distorted,img_distorted_bool_array= radial_distortion(img,distortion_k)
    unmapped_points=get_unmapped_points(img_distorted_bool_array)
    with tqdm(total=iterations) as pbar:
        for i in range(iterations):
            #name= "fixed"+str(i)+".png"
            fixed=remove_black_dots(unmapped_points,img_distorted)
            img_distorted=fixed #fixed=remove_black_dots(unmapped_points,fixed)
            pbar.update(1)
        name="k1_"+str(distortion_k[0])+"k2_"+str(distortion_k[1])+"k3_"+str(distortion_k[2])+"fixed_radial.png"
        save_image(fixed,name)
        plt.imshow(fixed)
        return fixed
### Tangential Distortion
def tangential_distortion(img,distortion_p,x_optic=None,y_optic=None,x_focal=None,y_focal=None):
    x_dim = img.shape[0]
    y_dim = img.shape[1]
    img_distorted = np.zeros(img.shape).astype(np.uint8)
    img_distorted_bool_array = np.zeros((x_dim,y_dim)).astype(np.uint8)

    ## assumed the center to be the optical center, can be changed later
    if x_optic is None:
        x_optic= x_dim//2
    if y_optic is None:
        y_optic= y_dim//2
    if x_focal is None:
        x_focal=x_dim//2
    if y_focal is None:
        y_focal= y_dim//2
    p1,p2= distortion_p
    r_max = np.sqrt(2) ## since x_norm <=1 and y_norm <=1
    x_scale = 1+ (2*p1+p2*(r_max**2+2))
    y_scale = 1+ (p1*(r_max**2+2)+2*p2)
    
    for x in range (x_dim):
        for y in range (y_dim):
        #Normalized image coordinates are calculated from pixel coordinates by translating to the optical center and dividing by the focal length in pixels.
            x_norm = (x-x_optic)/x_focal
            y_norm = (y-y_optic)/y_focal
        # r= x**2 + y**2 
            r = np.sqrt(x_norm**2 + y_norm**2)
            x_dist_norm = (x_norm+(2*p1*x_norm*y_norm+p2*(r**2+2*x_norm**2)))/x_scale
            y_dist_norm = (y_norm+(p1*(r**2+2*y_norm**2)+2*p2*x_norm*y_norm))/y_scale
            x_distorted = int(x_dist_norm*x_focal + x_optic) 
            y_distorted = int(y_dist_norm*y_focal + y_optic) 
            img_distorted_bool_array[x_distorted][y_distorted]=1

            try:
               # print(x_distorted,y_distorted,x,y)
                img_distorted[x_distorted][y_distorted]=img[x][y]
            except:
                print("dimension unmatched",x_distorted,y_distorted)
                pass
    name="p1_"+str(distortion_p[0])+"p2_"+str(distortion_p[1])+"unfixed_tan_"+".png"
    save_image(img_distorted,name)
                
    return img_distorted,img_distorted_bool_array

def tangential_distortion_fixed(img,distortion_p,iterations=5):
    img_tan,img_tan_unmapped=tangential_distortion(img,distortion_p)
    unmapped_tan=get_unmapped_points(img_tan_unmapped)
    with tqdm(total=iterations) as pbar:
        for i in range(iterations):
            # name= "fixed"+str(i)+".png"
            fixed=remove_black_dots(unmapped_tan,img_tan)
            img_distorted=fixed #fixed=remove_black_dots(unmapped_points,fixed)
            pbar.update(1)
        name="p1_"+str(distortion_p[0])+"p2_"+str(distortion_p[1])+"fixed_tan"+".png"
        save_image(fixed,name)
        plt.imshow(fixed)
        return fixed
