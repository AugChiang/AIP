import cv2
import numpy as np
import random
# Color constant
BLUE = (255,0,0)
BLACK = (0,0,0)
WHITE = (255,255,255)
# weight constant
ALPHA = 0.8
BETA = 1 - ALPHA
GAMMA = 0

def Empty(NA):
    pass

def Get_cont(img, img_fill):
    ZEROS = np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8)
    # print("ZEROS SIZE:", ZEROS.shape)
    # store contour coordinates
    # RETR_EXTERNAL: grab only outsider of the objects, not holes'
    # para: CHAIN_APPROX_NONE: save all contour coordinates
    contours, hierar = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(len(contours)) # num of the contour set
    for cnt in contours:
        # area = contour region circled area
        area = cv2.contourArea(cnt)
        # print("CNT: ",cnt)
        # print(type(cnt)) # type:np.array
        # print(area)
        if area >= 100:
            # draw contour lines
            cv2.drawContours(img_contour, cnt, -1, BLUE,2)
            # fill the poly-region by contour coordinates.
            cv2.fillConvexPoly(img_fill, cnt, BLUE)
            mask = cv2.fillConvexPoly(ZEROS, cnt, BLUE)
            # cv2.addWeighted(img_fill, ALPHA, mask, BETA, GAMMA)
    return mask

# def Correct_Rate(img,mask):
#     height, width = img.shape[0],img.shape[1]
#     total = height * width
#     # print(total)
#     cnt = np.count_nonzero(mask)
#     rate = str(round(cnt/total) * 100) + "%"
#     # print(rate)
#     return rate

kernel = np.ones((3,3), np.uint8)
img = cv2.imread(r"C:\Users\August\VSCode_Python\GUI\AIP_HW7\image\street05.jpg")
height, width = img.shape[0],img.shape[1]
img = cv2.resize(img,(width//2,height//2)) # width, height

# zero of img
ZEROS = img.copy()
ZEROS[:,:,:] = (0,0,0)

# add a boundary = BLACK
height, width = img.shape[0],img.shape[1]
# print(img.shape[0],img.shape[1])
# img[0,:,:] = (0,0,0)
# img[height-1,:,:] = (0,0,0)
# img[:,width-1,:] = (0,0,0)
# img[:,0,:] = (0,0,0)
# up most row all = 0
# print(img[0][:][:])
# print(img)
# print(img.shape)
detect_res = np.zeros(img.shape, np.uint8)

'''Track bars to tune the parameters'''
cv2.namedWindow("Track_Bar")
cv2.resizeWindow("Track_Bar",800,200)
cv2.createTrackbar("Morph_open_min","Track_Bar",10,250,Empty)
cv2.createTrackbar("Morph_open_max","Track_Bar",250,250,Empty)
cv2.createTrackbar("Morph_close_min","Track_Bar",10,250,Empty)
cv2.createTrackbar("Morph_close_max","Track_Bar",250,250,Empty)


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_contour = img.copy()
img_fill_open = img.copy()
img_fill_close = img.copy()
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_blur = cv2.GaussianBlur(img_gray,(5,5),0)
#img_canny1 = cv2.Canny(img_blur,100,100)
img_canny2 = cv2.Canny(img_blur,50,80)
# img_dil = cv2.dilate(img_canny2, kernel, iterations=1) # enforce the edges

#while True:
img_fill_open = img.copy()
img_fill_close = img.copy()
open_min = cv2.getTrackbarPos("Morph_open_min","Track_Bar")
open_max = cv2.getTrackbarPos("Morph_open_max","Track_Bar")
close_min = cv2.getTrackbarPos("Morph_close_min","Track_Bar")
close_max = cv2.getTrackbarPos("Morph_close_max","Track_Bar")

opening = cv2.morphologyEx(img_blur, cv2.MORPH_OPEN, kernel)
img_canny3 = cv2.Canny(opening,open_min,open_max)
closing = cv2.morphologyEx(img_blur, cv2.MORPH_CLOSE, kernel)
img_canny4 = cv2.Canny(closing,close_min,close_max)

seg = []
# for i in range(100):
#     print(random.randint(0,img.shape[0]),random.randint(0,img.shape[1]))
mask_open = Get_cont(img_canny3, img_fill_open)
mask_close = Get_cont(img_canny4, img_fill_close)
img_stack = np.hstack([img, img_fill_open, img_fill_close])

# calculate match rate
r_open = round(np.count_nonzero(mask_open)*100/(height*width*0.6))
r_close = round(np.count_nonzero(mask_close)*100/(height*width*0.6))
print(f"Match Rate(open_morph): {r_open}%.")
print(f"Match Rate(close_morph): {r_close}%.")
# img_stack2 = np.hstack([img_blur, opening, closing])
# img_stack3 = np.hstack([img_canny2,img_canny3,img_canny4])
# img_stack4 = np.vstack([img_stack2,img_stack3])
# cv2.imshow("Output Window", img_stack)
# cv2.imshow("Output Window", img_stack4)
# cv2.imshow("Output Window", closing)
# cv2.imshow("Output Window", img_stack)
# cv2.imshow("Output Window", img_canny2)
cv2.imshow("Output Window", img_stack)
# print(img.shape)
cv2.waitKey(0)
# cv2.destroyAllWindows()