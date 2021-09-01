import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('/home/aynasir/Documents/python_work/monocular3D/rightmonoboulanger/f/*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        #cv2.imshow('img',img)
        #cv2.waitKey(500)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print('Intrinsic_mtx_right', mtx) #output camera matrix
print('dist_right', dist)          # output vector of distortion coefficients 4th order output,The output vector length depends on the flags.
print('Rotation_matrix', rvecs)  #Output rotation matrix between the 1st and the 2nd camera coordinate systems.
print('Translation_vector', tvecs)  #Output translation vector between the coordinate systems of the cameras.

img1 = cv2.imread('/home/aynasir/Documents/python_work/monocular3D/rightmonoboulanger/f/right3.png')
h,  w = img1.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
print('Intrinsic_mtx_new', newcameramtx) #output optimised new camera matrix
print('Region of interest', roi) #output Region of Interest

# undistort
dst = cv2.undistort(img1, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)
cv2.imshow('calibresult_undistorted.png',dst)

# # undistort using remap
# mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
# dst = cv2.remap(img1,mapx,mapy,cv2.INTER_LINEAR)

# # crop the image
# x,y,w,h = roi
# dst2 = dst[y:y+h, x:x+w]
# cv2.imwrite('remap calibresult.png',dst2)
# cv2.imshow('calibresult_undistorted_remap.png',dst2)

mean_error= 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
tot_err = mean_error/len(objpoints)

print("total error: ", tot_err) 

# def draw(img, corners, imgpts):
    # corner = tuple(corners[0].ravel())
    # img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    # img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    # img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    # return img
    
def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

img2 = cv2.imread('calibresult.png')
_, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
axis = np.float32( [ [0,0,0], [0,3,0], [3,3,0], [3,0,0], [0,0,-3], [0,3,-3], [3,3,-3], [3,0,-3] ] )
imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
img2 = draw(img2,corners2,imgpts)
cv2.imshow('proj_img',img2)
cv2.waitKey(500)

image = cv2.imread("/home/aynasir/Documents/python_work/monocular3D/rightmonoboulanger/oct1/right2047.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


blurred= cv2.medianBlur(image, 9)
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow('gaussian_blur_9_5', gaussian_blur) 


# sift = cv2.xfeatures2d.SIFT_create()
# keypoints, descriptors = sift.detectAndCompute(gaussian_blur, None)
# imgkp = cv2.drawKeypoints(gaussian_blur, keypoints, None)
# cv2.imshow("imageptsift", imgkp)
# print("SIFT # kps: {}, descriptors: {}".format(len(keypoints), descriptors.shape))


# surf = cv2.xfeatures2d.SURF_create()
# keypoints, descriptors = surf.detectAndCompute(gaussian_blur, None)
# imgkp = cv2.drawKeypoints(gaussian_blur, keypoints, None)
# cv2.imshow("imageptsurf", imgkp)
# print("SURF # kps: {}, descriptors: {}".format(len(keypoints), descriptors.shape)) 


orb = cv2.ORB_create(nfeatures=10000)
keypoints, descriptors = orb.detectAndCompute(gaussian_blur, None)
print("ORB # kps: {}, descriptors: {}".format(len(keypoints), descriptors.shape))
imgkp = cv2.drawKeypoints(gaussian_blur, keypoints, None)
cv2.imshow("imageptorb", imgkp)
cv2.waitKey(0)

cv2.destroyAllWindows()
