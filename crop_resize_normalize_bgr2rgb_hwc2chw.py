# OpenCV implementation of crop、resize、normalize、bgr2rgb and hwc2chw

import os
import shutil
import numpy as np
import cv2


def crop_yolo_resized_with_affine_transform(src_img, roi_xywh, des_width, des_height, isFixResize=True, isSymmetryPad=False):    

    '''
    image roi:
    -----------------------------------
    -----------------------------------
    ----(x0,y0)*************(x1,y1)----
    ----***************************----
    ----***************************----
    ----***************************----
    ----***************************----
    ----***************************----
    ----***************************----
    ----***************************----
    ----***************************----
    ----(x3,y3)************(x2,y2)-----
    -----------------------------------
    -----------------------------------
    '''
    src_points = [
            [roi_xywh[0], roi_xywh[1]], 
            [roi_xywh[0] + roi_xywh[2], roi_xywh[1]], 
            [roi_xywh[0] + roi_xywh[2], roi_xywh[1] + roi_xywh[3]]]
    src_points = np.array(src_points, dtype=np.float32)
        
    if isFixResize:
        roi_width = roi_xywh[2]
        roi_height = roi_xywh[3]
        
        scale_x = des_width / roi_width
        scale_y = des_height / roi_height
        scale = scale_x if scale_x < scale_y else scale_y
               
        src_new_width  = roi_width * scale
        src_new_height = roi_height * scale
        '''
        roi_height > roi_width                roi_height < roi_width
        ****************000000                ********************
        ****************000000                ********************
        ****************000000                ********************
        ****************000000                ********************
        ****************000000                ********************
        ****************000000                ********************
        ****************000000                ********************
        ****************000000                00000000000000000000
        ****************000000                00000000000000000000
        ****************000000                00000000000000000000
        '''
        x = 0
        y = 0        
        if isSymmetryPad:
            '''
            roi_height > roi_width                roi_height < roi_width
            000****************000                00000000000000000000
            000****************000                00000000000000000000
            000****************000                ********************
            000****************000                ********************
            000****************000                ********************
            000****************000                ********************
            000****************000                ********************
            000****************000                ********************
            000****************000                00000000000000000000
            000****************000                00000000000000000000
            '''        
            x = (des_width - src_new_width) / 2
            y = (des_height - src_new_height) / 2

        
        des_points = [[x, y], [x + src_new_width, y], [x + src_new_width, y + src_new_height]]
    else:
        des_points = [[0, 0], [des_width, 0], [des_width, des_height]]
    
    des_points = np.array(des_points, dtype=np.float32)

    M = cv2.getAffineTransform(src_points, des_points)
    print("Src pooints is : ", src_points)
    print("Dst pooints is : ", des_points)
    print("Transform matrix is : ", M)
    crop_and_yolo_resized_with_affine_transform = cv2.warpAffine(src_img, M, (des_width, des_height))

    return crop_and_yolo_resized_with_affine_transform
    

def crop_resized_normalize(img_path, des_size, isFixResize=True, isSymmetryPad=False, roi_xywh=None):
    img = cv2.imread(img_path)
    if roi_xywh is None:
        src_height, src_width = img.shape[:2]
        # xmin ymin xmax ymax
        roi_xywh = [0, 0, src_width, src_height]
    # /**********************warpAffine**********************/
    img_after_warpAffine = crop_yolo_resized_with_affine_transform(img, roi_xywh , des_size, des_size, isFixResize, isSymmetryPad)
    cv2.imwrite("img_after_warpAffine_with_python.jpg", img_after_warpAffine)
    
    # /**********************bgr2rgb*************************/
    img_after_bgr2rgb = img_after_warpAffine[:,:,::-1]
    cv2.imwrite("img_after_bgr2rgb_with_python.jpg", img_after_bgr2rgb)

    # /********************uint8 -> float********************/
    img_after_bgr2rgb_float = img_after_bgr2rgb.astype(np.float32)
    
    # /*********************(x - a) / b**********************/
    img_after_bgr2rgb_float_normalize = (img_after_bgr2rgb_float - 0) * 0.00392157 

    # /**********************hwc2chw*************************/
    img_after_bgr2rgb_float_normalize_chw = img_after_bgr2rgb_float_normalize.transpose((2, 0, 1))
    # np.savetxt('img_after_bgr2rgb_float_normalize_chw_with_python.txt', img_after_bgr2rgb_float_normalize_chw.flatten(), fmt="%.5f")

def copy_image(des_size, isFixResize, isSymmetryPad, roi_xywh):
    root_dir = "py_des_size_%d_isFixResize_%d_isSymmetryPad_%d"%(des_size, isFixResize, isSymmetryPad)
    if roi_xywh is not None:
        root_dir += "_roi_xywh_%d_%d_%d_%d"%(roi_xywh[0], roi_xywh[1], roi_xywh[2], roi_xywh[3])
    os.makedirs(root_dir, exist_ok=True)
    shutil.move("img_after_warpAffine_with_python.jpg", root_dir)
    shutil.move("img_after_bgr2rgb_with_python.jpg", root_dir)


if __name__ == "__main__":
    img_path = "Joe_Biden_presidential_portrait.jpg" # face roi(xywh) is : [374, 132, 462, 624]
    des_size = 640
    
    isFixResize = False
    isSymmetryPad = False
    roi_xywh = None
    crop_resized_normalize(img_path, des_size, isFixResize, isSymmetryPad, roi_xywh)
    copy_image(des_size, isFixResize, isSymmetryPad, roi_xywh)

    isFixResize = True
    isSymmetryPad = False
    roi_xywh = None
    crop_resized_normalize(img_path, des_size, isFixResize, isSymmetryPad, roi_xywh)
    copy_image(des_size, isFixResize, isSymmetryPad, roi_xywh)

    isFixResize = True
    isSymmetryPad = True
    roi_xywh = None
    crop_resized_normalize(img_path, des_size, isFixResize, isSymmetryPad, roi_xywh)
    copy_image(des_size, isFixResize, isSymmetryPad, roi_xywh)

    isFixResize = False
    isSymmetryPad = False
    roi_xywh = [374, 132, 462, 624]
    crop_resized_normalize(img_path, des_size, isFixResize, isSymmetryPad, roi_xywh)
    copy_image(des_size, isFixResize, isSymmetryPad, roi_xywh)

    isFixResize = True
    isSymmetryPad = False
    roi_xywh = [374, 132, 462, 624]
    crop_resized_normalize(img_path, des_size, isFixResize, isSymmetryPad, roi_xywh)
    copy_image(des_size, isFixResize, isSymmetryPad, roi_xywh)

    isFixResize = True
    isSymmetryPad = True
    roi_xywh = [374, 132, 462, 624]
    crop_resized_normalize(img_path, des_size, isFixResize, isSymmetryPad, roi_xywh)
    copy_image(des_size, isFixResize, isSymmetryPad, roi_xywh)