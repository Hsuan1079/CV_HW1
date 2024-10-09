import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32) # 建立一個7*7的矩陣，每個元素都是[0,0,0] 代表著角點的3D世界座標係
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2) # 將x,y座標值填入obj,z=0 

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('data3/*.jpg')

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #轉灰階
    plt.imshow(gray) #顯示灰階圖片

    #Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None) #ret是一個bool值，corners是一個2D座標的矩陣

    # If found, add object points, image points
    if ret == True:
        print('find the chessboard corners of',fname)
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        plt.imshow(img)
    else:
        print('cannot find the chessboard corners of',fname)


#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
img_size = img[0].shape
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
Vr = np.array(rvecs) #Vr是旋轉向量
Tr = np.array(tvecs) #Tr是平移向量
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6) #extrinsics是外部參數，包含旋轉向量和平移向量 [R|t]

# 1. Use the points in each images to find Hi
Hi = [] # Hi = [h11, h12, h13, h21, h22, h23, h31, h32, h33]  共9個元素
# 總共有corner_x*corner_y個角點，產生corner_x*corner_y*2個方程式
# 總共有len(objpoints)張圖片，產生len(objpoints)*corner_x*corner_y*2個方程式
for i in range(len(objpoints)):
    P = np.zeros((corner_x*corner_y*2,9))
    obj_points = objpoints[i]
    img_points = imgpoints[i].reshape(-1,2)
    idx = 0
    for j in range(corner_x * corner_y):
        U, V = obj_points[j,0], obj_points[j,1]
        u, v = img_points[j,0], img_points[j,1]
        P[idx] = np.array([U,V,1,0,0,0,-u*U,-u*V,-u])
        P[idx+1] = np.array([0,0,0,U,V,1,-v*U,-v*V,-v])
        idx += 2
    U, D, Vt = np.linalg.svd(P) #奇異值分解 P = UDV^T
    h = Vt.T[:,-1] #取最後一行
    if h[-1] < 0: 
        h = -h
    h = h/h[-1] #最後一個元素除以自己
    H = h.reshape(3,3)
    Hi.append(H)

# print('Hi:',Hi)

# 2. Use Hi to find out the intrinsic matrix K
# Hi = K[R|t] => Hi = K[r1 r2 t]
# 每個H可以對應到一個Ｋ，每個Ｋ有兩個方程式可以解
# 1. 旋轉矩陣r1 r2 orthogonal => r1*r2 = 0
# 2. r1*r1 = r2*r2
v = np.zeros((2 * len(Hi), 6)) # 總共有len(Hi)張圖片，每張圖片有2個方程式
idx = 0
for H in Hi:
    v[idx] = [H[0,0]*H[0,1],
            H[0,0]*H[1,1]+H[1,0]*H[0,1],
            H[0,0]*H[2,1]+H[2,0]*H[0,1], 
            H[1,0]*H[1,1],
            H[1,0]*H[2,1]+H[2,0]*H[1,1],
            H[2,0]*H[2,1]]
    v[idx+1] = [H[0,0]*H[0,0]-H[0,1]*H[0,1],
                2*(H[0,0]*H[1,0]-H[0,1]*H[1,1]),
                2*(H[0,0]*H[2,0]-H[0,1]*H[2,1]),
                H[1,0]*H[1,0]-H[1,1]*H[1,1],
                2*(H[1,0]*H[2,0]-H[1,1]*H[2,1]),
                H[2,0]*H[2,0]-H[2,1]*H[2,1]]
    idx += 2

U, D, Vt = np.linalg.svd(v)
b = Vt.T[:,-1]
if b[0] < 0 or b[3] < 0 or b[5] < 0:
    b = -b
B = np.array([[b[0], b[1], b[2]],
              [b[1], b[3], b[4]],
              [b[2], b[4], b[5]]])

B11, B12, B13, B22, B23, B33 = b[0], b[1], b[2], b[3], b[4], b[5]

# B = K^(-T) * K^(-1)
# compute intrinsic matrix K
# K = [fx, s, cx, 0, fy, cy, 0, 0, 1]
l = np.linalg.cholesky(B) # Cholesky decomposition 是一種將矩陣分解成下三角矩陣的方法 B = LL^T
K = np.linalg.inv(l.T) # K = (L^T)^(-1)
# normalize K
K = K / K[2,2]

# print('Intrinsic matrix K:')
# print(K)
# print("-----------------")

# 3. Find out the extrinsics matrix of each images.
# Hi = K[R|t] => [R|t] = K^(-1)Hi
extrinsics_find = np.zeros((len(Hi), 6))
for i, H in enumerate(Hi):
    lambda_value = 1 / np.linalg.norm(np.dot(np.linalg.inv(K), H[:,0]))
    K_inv = np.linalg.inv(K)
    h1 = H[:,0]
    h2 = H[:,1]
    h3 = H[:,2]
    r1 = lambda_value*np.dot(K_inv, h1) #r1 是一個3*1的向量
    r2 = lambda_value*np.dot(K_inv, h2) #r2 是一個3*1的向量
    t = lambda_value*np.dot(K_inv, h3) #t 是一個3*1的向量
    r3 = np.cross(r1, r2) #r3 是一個3*1的向量
    rotation_matrix = np.column_stack((r1, r2, r3)) 
    extrinsics_find[i] = np.concatenate((cv2.Rodrigues(rotation_matrix)[0].reshape(1,3), t.reshape(1,3)), axis=1).reshape(6)

print('Extrinsics matrix we find:')
formatted_output = np.array2string(extrinsics_find, precision=8, floatmode='fixed', suppress_small=True)
print(formatted_output)
print("-----------------")
print('Extrinsics matrix from cv2:')
formatted_output = np.array2string(extrinsics, precision=8, floatmode='fixed', suppress_small=True)
print(formatted_output)
# extrinsics = extrinsics_find

# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d') 
# camera setting
camera_matrix = mtx
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()

#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""
