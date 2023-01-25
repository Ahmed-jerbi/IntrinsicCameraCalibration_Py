import sys
import cv2
#using opencv4 instead
#assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob
# You should replace these 3 lines with the output in calibration step
DIM=(2592, 1944)
K=np.array([[1023.5812051715526, 0.0, 1326.6993328536932], [0.0, 1024.705723910922, 1085.184478823344], [0.0, 0.0, 1.0]])
D=np.array([[-0.09093861316302901], [0.0072126638839035235], [-0.005205831659436931], [0.00516023366178049]])
def undistort(img_path):    
    img = cv2.imread(img_path)
    h,w = img.shape[:2]    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) 

    #create window
    cv2.namedWindow("undistorted", cv2.WINDOW_NORMAL)

    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)