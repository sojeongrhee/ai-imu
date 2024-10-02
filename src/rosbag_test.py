import math
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import csv
#import utm
import os

#from icp import icp


if __name__ == '__main__':
    # set seed for reproducible results
    np.random.seed(12345)

    exp_num = 2
    path_name = "../dataset/sheco_data"
    csv_name = "sample{}.csv".format(exp_num)
    #bag_name = "ekf_output{}.csv".format(exp_num)
    #bag_name = "ekf_final_{}.csv".format(exp_num)
    bag_name = "ekf_py_{}.csv".format(exp_num)
    # read csv file
    df_ref = pd.read_csv(os.path.join(path_name, bag_name))
    df_align = pd.read_csv(os.path.join(path_name,csv_name))


    num_row = len(df_ref.iloc[:,0])
    # x_ = np.array(df_ref.iloc[:,1]).reshape(-1,1)
    # y_ = np.array(df_ref.iloc[:,2]).reshape(-1,1)
    x_ = np.array(df_ref.iloc[:,2]).reshape(-1,1)
    y_ = np.array(df_ref.iloc[:,3]).reshape(-1,1)
    #fig, ax = plt.subplots()   
    #ax.plot(x_[:, 0], y_[:, 0], 'rx', label='reference points')
    #plt.show()
    # reference_points = np.hstack((xs.reshape(-1,1),ys.reshape(-1,1)))

    reference_points = np.hstack((x_, y_))
    # transform the set of reference points to create a new set of
    # points for testing the ICP implementation

    # 1. remove some points

    xs_ = np.array(df_align.iloc[:,1]).reshape(-1,1)
    ys_ = np.array(df_align.iloc[:,2]).reshape(-1,1)
    points_to_be_aligned = np.hstack((xs_,ys_))

    # 2. apply rotation to the new point set
    theta = math.radians(0)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s],
                    [s, c]])
    points_to_be_aligned = np.dot(points_to_be_aligned, rot)

    # 3. apply translation to the new point set
    ##1 
    #points_to_be_aligned += np.array([5,20])
    ##2
    #points_to_be_aligned += np.array([7,28])
    ##3 
    #points_to_be_aligned += np.array([8,27])
    #points_to_be_aligned += np.array([313200, 4132856])

    # run icp
    #transformation_history, aligned_points = icp(reference_points, points_to_be_aligned, verbose=True)

    # show results
    fig, ax = plt.subplots()   
    ax.plot(reference_points[:, 0], reference_points[:, 1], 'rx', label='reference points')
    #fig, ax = plt.subplots()   
    ax.plot(points_to_be_aligned[:, 0], points_to_be_aligned[:, 1], 'b1', label='points to be aligned')
    #plt.plot(aligned_points[:, 0], aligned_points[:, 1], 'g+', label='aligned points')
    #plt.legend()
    fig.savefig(os.path.join("./plot","exp_{}_python.png".format(exp_num)))
    #plt.show()

    # t_gt = np.array(df_align.iloc[:,0]).reshape(-1,1)
    # t_gt = t_gt - t_gt[0]
    # t_ = np.array(df_ref.iloc[:,0]).reshape(-1,1)
    # t_ = t_ - t_[0]
    # ang_gt = np.array(df_align.iloc[:,7:10]).reshape(-1,3)
    # ang = np.array(df_ref.iloc[:,7:10]).reshape(-1,3)
    # fig2, ax2 = plt.subplots(3,1, sharex= True, figsize=(20, 10))
    # ax2[0].plot(t_gt, ang_gt[:,0])
    # ax2[0].plot(t_, ang[:,0])
    # ax2[1].plot(t_gt, ang_gt[:,1])
    # ax2[1].plot(t_, ang[:,1])
    # ax2[2].plot(t_gt, ang_gt[:,2])
    # ax2[2].plot(t_, ang[:,2])
    # ax2[0].set(xlabel='time (s)', ylabel=r'$\phi_n$ (rad)', title="Orientation")
    # ax2[1].set(xlabel='time (s)', ylabel=r'$\theta_n$ (rad)', title="Orientation")
    # ax2[2].set(xlabel='time (s)', ylabel=r'$\psi_n$ (rad)', title="Orientation")
    # ax2[0].legend([r'$\phi_n^x$', r'$\hat{\phi}_n^x$'])
    # ax2[1].legend([r'$\theta_n^y$',r'$\hat{\theta}_n^y$'])
    # ax2[2].legend([r'$\psi_n^z$', r'$\hat{\psi}_n^z$'])
    # plt.show()

    # plot heading
    # head_ref = np.deg2rad(np.array(df_ref.iloc[:,5]))
    # index = np.array(range(len(head_ref)))
    # head_align = np.array(df_align.iloc[:,9])
    # dt_align = np.array(df_align.iloc[:,0])
    # #dt_align = (dt_align[1:]-dt_align[:-1])/1000000000
    # #dt_align = np.hstack(([0], dt_align))
    # #print(dt_align)
    # fig, ax = plt.subplots()   
    # ax.plot(dt_align,head_align)
    # fig, ax = plt.subplots()   
    # ax.plot(index,head_ref)
    # #plt.show()


    # fig, ax = plt.subplots()
    # # imu heading 
    
    # x_dir = np.cos(np.array(df_align.iloc[:,9]))
    # y_dir = np.sin(np.array(df_align.iloc[:,9]))
    # num = np.array(range(len(df_align.iloc[:,1])))
    # #num = np.array(range(20000,24000))
    # index = np.array(range(len(num)))
    # quiver =ax.quiver(np.array(df_align.iloc[:,1])[num], np.array(df_align.iloc[:,2])[num], x_dir[num], y_dir[num], index, cmap='jet')

    # plt.colorbar(quiver, ax=ax, label='time')

    # plt.show()
