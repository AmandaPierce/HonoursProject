from PIL import Image
import cv2
import os
import glob
import numpy as np
import imutils
import math
from scipy import signal
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def perform_character_segmentation(filename):
    image = cv2.imread(filename)
    
    baseline_extraction(image)


def baseline_extraction(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    image = cv2.dilate(image, kernel)
    image = cv2.erode(image, kernel)
    image_copy = np.copy(image)
    baselines = []

    # cv2.imshow("Before baseline extraction", image)
    # cv2.waitKey(0)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_not(image)

    height, width = image.shape[:2]

    biggest_total = 0
    center_y = 0

    for y in range(0, height - 1):
        current_total = 0
        for x in range(0, width - 1):
            current_total += image[y][x]

        if (current_total > biggest_total):
            biggest_total = current_total
            center_y = y

    cv2.line(image_copy, (0, center_y), (width, center_y), (0,255,0), 1)
    cv2.imshow("After baseline extraction", image_copy)
    cv2.waitKey(0)

    image_crop_bottom = image[center_y:height, 0: width]

    # cv2.imshow("After baseline extraction", image_crop_bottom)
    # cv2.waitKey(0)

    y_vals = []
    x_vals = []

    height_bottom, width_bottom = image_crop_bottom.shape[:2]

    current_y = height_bottom - 1
    for x in range(0, width_bottom - 1):
        while (current_y >= 0) and (image_crop_bottom[current_y][x] < 253):
            current_y -= 1
       
        y_vals.append(current_y)
        x_vals.append(x)
        current_y = height_bottom - 1

    y = np.array(y_vals)
    x = np.array(x_vals)

    sortId = np.argsort(x)
    x = x[sortId]
    y = y[sortId]

    peaks, _ = signal.find_peaks(y)

    peaks_list = peaks.tolist()

    x = []
    y = []
    for val in peaks_list: 
        y.append(y_vals[val])
        x.append(x_vals[val])
        # cv2.circle(image_copy,(x_vals[val],y_vals[val] + center_y), 2, (255,0,255), -1)

    # cv2.imshow("After baseline extraction", image_copy)
    # cv2.waitKey(0)


    all_M = []
    for v in peaks:
        angles = []
        y_tmp_vals = []

        for c in peaks:
            if(peaks_list.index(c) != peaks_list.index(v)):
                myradians = math.atan2(y_vals[v] - y_vals[c], x_vals[v]-x_vals[c])
                mydegrees = math.degrees(myradians)
                angles.append(mydegrees)
                y_tmp_vals.append(y_vals[c])

        xpts = np.zeros(0)
        ypts = np.zeros(0)
        labels = np.zeros(0)

        angle_matrix = np.array(angles)

        xpts = np.hstack((xpts, angle_matrix))
        
        ypts = np.hstack((ypts, y_tmp_vals))

        labels = np.hstack((labels, np.ones(len(y_tmp_vals)) * len(y_tmp_vals)))

        # fig0, ax0 = plt.subplots()
        # ax0.plot(xpts, ypts, '.')
        # ax0.set_title('Gradients')
        # plt.show()

        fig1, axes1 = plt.subplots()
        alldata = np.vstack((xpts, ypts))
    #     fpcs = []

        colours = ['b', 'g']

        fpcs = []
        c_mem = []

        for ncenters in range(2):
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, ncenters + 1, 2, error=0.005, maxiter=1000, init=None)

            fpcs.append(fpc)
            c_mem.append(u)
    

        idx_to_use = fpcs.index(max(fpcs))
        
    # #         # Store fpc values for later
    # #         fpcs.append(fpc)
        
        # Plot assigned clusters, for each data point in training set
        cluster_membership = np.argmax(c_mem[idx_to_use], axis=0)

        c1 = None
        c2 = None
        final_vals_1 = None
        final_vals_2 = None

        for j in range(2):
            if j == 0:
                c1 = zip(tuple(xpts[cluster_membership == j].tolist()), tuple(ypts[cluster_membership == j].tolist()))
                final_vals_1 = tuple(ypts[cluster_membership == j].tolist()) 
            else:
                c2 = zip(tuple(xpts[cluster_membership == j].tolist()), tuple(ypts[cluster_membership == j].tolist()))
                final_vals_2 = tuple(ypts[cluster_membership == j].tolist()) 

            # axes1.plot(xpts[cluster_membership == j], ypts[cluster_membership == j], '.', color=colours[j])

        # if(len(tuple(c1)) > len(tuple(c2))):
        #     print(tuple(c1))
        # else:
        #     print(tuple(c2))

        c1_t = tuple(c1)
        c2_t = tuple(c2)

        l1 = len(c1_t)
        l2 = len(c2_t)

        # print((x[y.index(int(4))], y[y.index(int(4))]))
        
        all_M_tmp = []
        if(l1 > l2):
            for i in final_vals_1:
                 all_M_tmp.append((float(x[y.index(int(i))]), float(y[y.index(int(i))])))
            # print(y[y.index(int(i))])
            # all_M_tmp.append(c1_t)
        else:
            for i in final_vals_2:
                 all_M_tmp.append((float(x[y.index(int(i))]), float(y[y.index(int(i))])))
            # all_M_tmp.append(c2_t)
    
        all_M.append(tuple(all_M_tmp))

    final_baseline = (float("inf"), 0)
    for val in all_M:
        mean = getMean(val)
        sumDist = linearRegression(x, y, mean, image_copy, center_y)

        if(sumDist[0] < final_baseline[0]):
            final_baseline = sumDist
    
   
    cv2.line(image_copy,(int(min(x)),int(min(final_baseline[1]) + center_y)),(int(max(x)),int(max(final_baseline[1]) + center_y)),(255,0,0),1)
    cv2.imshow("linregress", image_copy)
    cv2.waitKey(0)

    baselines.append(((int(min(x)),int(min(final_baseline[1]) + center_y)),(int(max(x)),int(max(final_baseline[1]) + center_y))))

    image_crop_top = image[0:center_y, 0: width]

    y_vals_top = []
    x_vals_top = []

    height_top, width_top = image_crop_top.shape[:2]

    cv2.imshow("AAA", image_crop_top)
    cv2.waitKey(0)

    current_y = 0
    usable_y = height_top - 1
    for x in range(0, width_top - 1):
        while (current_y < height_top) and (image_crop_top[current_y][x] < 253):
            current_y += 1
            usable_y -= 1
       
        y_vals_top.append(usable_y)
        x_vals_top.append(x)
        current_y = 0
        usable_y = height_top - 1

    # for val in range(len(x_vals_top)): 
    #     # y_vals_top.append(y_vals_top[val])
    #     # x_vals_top.append(x_vals_top[val])

    #     cv2.circle(image_copy,(x_vals_top[val],y_vals_top[val]), 2, (0,0,255), -1)
    #     print(y_vals_top[val])
    
    # cv2.imshow("testing top", image_copy)
    # cv2.waitKey(0)

    y_top = np.array(y_vals_top)
    x_top = np.array(x_vals_top)

    sortId = np.argsort(x_top)
    x_top = x_top[sortId]
    y_top = y_top[sortId]

    peaks_top, _ = signal.find_peaks(y_top)

    peaks_list = peaks_top.tolist()

    x_top = []
    y_top = []
    for val in peaks_list: 
        y_top.append((height_top - 1) - y_vals_top[val])
        x_top.append(x_vals_top[val])

    # for val in range(len(x_top)): 
    #     cv2.circle(image_copy,(x_top[val],center_y - y_top[val] ), 2, (0,0,255), -1)
    
    # cv2.imshow("testing top", image_copy)
    # cv2.waitKey(0)

    dist = []
    for idx in range(len(y_top)):
        dist.append(calculate_vertical_distance(y_top[idx], center_y, x_top[idx], final_baseline[2][0], final_baseline[2][1]))

        # cv2.circle(image_copy,(x_vals_top[val],y_vals_top[val]), 2, (0,0,255), -1)
    all_M_top = None

    xpts_top = np.zeros(0)
    ypts_top = np.zeros(0)
    labels_top = np.zeros(0)

    dist_matrix = np.array(dist)

    xpts_top = np.hstack((xpts_top, dist_matrix))
    
    ypts_top = np.hstack((ypts_top, y_top))

    labels_top = np.hstack((labels, np.ones(len(y_top)) * len(y_top)))

    fig1, axes1 = plt.subplots()
    alldata = np.vstack((xpts_top, ypts_top))
    
    colours = ['b', 'g']
    fpcs = []
    c_mem = []

    for ncenters in range(2):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, ncenters + 1, 2, error=0.005, maxiter=1000, init=None)

        fpcs.append(fpc)
        c_mem.append(u)
    

    idx_to_use = fpcs.index(max(fpcs))


    # Plot assigned clusters, for each data point in training set
    cluster_membership_top = np.argmax(c_mem[idx_to_use], axis=0)

    c1 = None
    c2 = None

    if idx_to_use > 0:
        c1 = zip(tuple(xpts_top[cluster_membership_top == 0].tolist()), tuple(ypts_top[cluster_membership_top == 0].tolist()))

        c2 = zip(tuple(xpts_top[cluster_membership_top == 1].tolist()), tuple(ypts_top[cluster_membership_top == 1].tolist()))

        c1_t = tuple(c1)
        c2_t = tuple(c2)

        l1 = len(c1_t)
        l2 = len(c2_t)
        
        if(l1 > l2):
            all_M_top = (c1_t)
        else:
            all_M_top = (c2_t)
    else:
        c1 = zip(tuple(xpts_top[cluster_membership_top == 0].tolist()), tuple(ypts_top[cluster_membership_top == 0].tolist()))
        c1_t = tuple(c1) 
        all_M_top = (c1_t)
    
    top_mean = getMean(all_M_top)

    x_smallest = x_top[x_top.index(min(x_top))]
    # y_smallest = y_top[x_top.index(min(x_top))]
    x_largest = x_top[x_top.index(max(x_top))]
    # y_largest =  y_top[x_top.index(max(x_top))]

    # cv2.circle(image_copy,(x_smallest,y_top[x_top.index(min(x_top))]), 2, (0,0,255), -1)

    Y_pred_top_start = final_baseline[2][0]*x_smallest + (top_mean)
    Y_pred_top_end = final_baseline[2][0]*x_largest + (top_mean)

    cv2.line(image_copy,(int(x_smallest),int(Y_pred_top_start)),(int(x_largest),int(Y_pred_top_end)),(255,0,0),1)
    cv2.imshow("linregress", image_copy)
    cv2.waitKey(0)

    baselines.append(((int(x_smallest),int(Y_pred_top_start)),(int(x_largest),int(Y_pred_top_end))))

    baselines.append(((baselines[0][0][0], int((baselines[0][0][1] + baselines[1][0][1])/2)),(baselines[0][1][0], int((baselines[0][1][1] + baselines[1][1][1])/2))))

    print(baselines)

    cv2.line(image_copy,(baselines[2][0]),(baselines[2][1]),(255,255,0),1)
    cv2.imshow("linregress", image_copy)
    cv2.waitKey(0)
    
def getMean(values):
    mean = 0.0
    num = 0.0
    for val in values:
        mean += val[1]
        num += 1.0
    
    return mean/num

def linearRegression(x, y, mean, image_copy, center_y):
    X_mean = np.mean(mean)
    Y_mean = mean

    x = np.array(x)
    y = np.array(y)

    num = 0
    den = 0
    for i in range(len(x)):
        num += (x[i] - X_mean)*(y[i] - Y_mean)
        den += (x[i] - X_mean)**2
    m = num / den
    c = Y_mean - m*X_mean

    Y_pred = m*x + c

    total = 0.0
    for idx in range(len(Y_pred)):
        dist = y[idx] - Y_pred[idx]
        total += dist**2

    return (total, Y_pred, (m,c))

def calculate_vertical_distance(y_pos, center_y, x, m, c):
    y_pred = m*x + c
    return (y_pred + center_y) - y_pos

        
    


