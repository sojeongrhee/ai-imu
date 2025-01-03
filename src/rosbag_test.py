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
    bag_name = "ekf_final_{}.csv".format(exp_num)
    py_bag_name = "ekf_py_{}.csv".format(exp_num)
    py_fixed_bag = "ekf_py_test_{}.csv".format(exp_num)
    cp_bag_name = "ekf_ext_{}.csv".format(exp_num)
    # read csv file

    ## df_ref : python
    ## df_align : sample
    ## df_ref2 : c++
    df_ref = pd.read_csv(os.path.join(path_name, bag_name))
    df_align = pd.read_csv(os.path.join(path_name,csv_name))
    df_ref2 = pd.read_csv(os.path.join(path_name, cp_bag_name))
    df_ref3 = pd.read_csv(os.path.join(path_name, py_bag_name))
    df_ref4 = pd.read_csv(os.path.join(path_name, py_fixed_bag))
    num_row = len(df_ref.iloc[:,0])
    # x_ = np.array(df_ref.iloc[:,1]).reshape(-1,1)
    # y_ = np.array(df_ref.iloc[:,2]).reshape(-1,1)
    x_ = np.array(df_ref4.iloc[:,2]).reshape(-1,1)
    y_ = np.array(df_ref4.iloc[:,3]).reshape(-1,1)
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

    #### 1. plot position
    fig, ax = plt.subplots()   
    ax.plot(reference_points[:, 0], reference_points[:, 1], 'rx', label='reference points')
    #fig, ax = plt.subplots()   
    #ax.plot(points_to_be_aligned[:, 0], points_to_be_aligned[:, 1], 'b1', label='points to be aligned')
    #plt.plot(aligned_points[:, 0], aligned_points[:, 1], 'g+', label='aligned points')
    #plt.legend()
    fig.savefig(os.path.join("./plot/ekf_fix","exp_{}_python_fix.png".format(exp_num)))
    # #plt.show()

    #### 2. plot orientation
    t_gt = np.array(df_align.iloc[:,0]).reshape(-1,1)
    t_gt = t_gt - t_gt[0]
    t_ = np.array(df_ref.iloc[:,0]).reshape(-1,1)
    t_ = t_ - t_[0]
    t_2 = np.array(df_ref2.iloc[:,0]).reshape(-1,1)
    t_2 = t_2 - t_2[0]
    t_3 = np.array(df_ref3.iloc[:,0]).reshape(-1,1)
    t_3 = t_3 - t_3[0]
    t_4 = np.array(df_ref4.iloc[:,0]).reshape(-1,1)
    t_4 = t_4 - t_4[0]
    #### 1. plot position
    pos_gt = np.array(df_align.iloc[:,1:4]).reshape(-1,3)
    pos_py = np.array(df_ref.iloc[:,2:5]).reshape(-1,3)
    #pos_cp = np.array(df_ref2.iloc[:,1:4]).reshape(-1,3)
    pos_py2 = np.array(df_ref3.iloc[:,2:5]).reshape(-1,3)
    pos_py3 = np.array(df_ref4.iloc[:,2:5]).reshape(-1,3)    
    fig1, ax1 = plt.subplots(3,1, sharex= True, figsize=(20, 10))
    # start = 15000
    # end = 15500
    # start_1 = 15040
    # end_1 = 15540
    # start_2 = 46837
    # end_2 = 48397

    # ax1[0].plot(t_gt[start_1:end_1], pos_gt[start_1:end_1,0])
    # ax1[0].plot(t_[start:end], pos_py[start:end,0])
    # ax1[0].plot(t_2[start:end], pos_cp[start:end,0])
    # ax1[0].plot(t_3[start_2:end_2], pos_py2[start_2:end_2,0])
    # ax1[1].plot(t_gt[start_1:end_1], pos_gt[start_1:end_1,1])
    # ax1[1].plot(t_[start:end], pos_py[start:end,1])
    # ax1[1].plot(t_2[start:end], pos_cp[start:end,1])
    # ax1[1].plot(t_3[start_2:end_2], pos_py2[start_2:end_2,1])
    # ax1[2].plot(t_gt[start_1:end_1], pos_gt[start_1:end_1,2])
    # ax1[2].plot(t_[start:end], pos_py[start:end,2])
    # ax1[2].plot(t_2[start:end], pos_cp[start:end,2])
    # ax1[2].plot(t_3[start_2:end_2], pos_py2[start_2:end_2,2])
    ax1[0].plot(t_gt[:], pos_gt[:,0])
    ax1[0].plot(t_[:], pos_py[:,0])
    #ax1[0].plot(t_2[:], pos_cp[:,0])
    ax1[0].plot(t_3[:], pos_py2[:,0])
    ax1[0].plot(t_4[:], pos_py3[:,0])

    ax1[1].plot(t_gt[:], pos_gt[:,1])
    ax1[1].plot(t_[:], pos_py[:,1])
    #ax1[1].plot(t_2[:], pos_cp[:,1])
    ax1[1].plot(t_3[:], pos_py2[:,1])
    ax1[1].plot(t_4[:], pos_py3[:,1])

    ax1[2].plot(t_gt[:], pos_gt[:,2])
    ax1[2].plot(t_[:], pos_py[:,2])
    #ax1[2].plot(t_2[:], pos_cp[:,2])
    ax1[2].plot(t_3[:], pos_py2[:,2])
    ax1[2].plot(t_4[:], pos_py3[:,2])
    ax1[0].set(xlabel='time (s)', ylabel=r'$x$ (m)', title="Position")
    ax1[1].set(xlabel='time (s)', ylabel=r'$y$ (m)', title="Position")
    ax1[2].set(xlabel='time (s)', ylabel=r'$z$ (m)', title="Position")
    # ax1[0].legend(['x_gt', 'x_py', 'x_cpp', 'x_py_100hz'])
    # ax1[1].legend(['y_gt', 'y_py', 'y_cpp', 'y_py_100hz'])
    # ax1[2].legend(['z_gt', 'z_py', 'z_cpp', 'z_py_100hz'])
    ax1[0].legend(['x_gt', 'x_py', 'x_py_100hz', 'x_py_fix'])
    ax1[1].legend(['y_gt', 'y_py', 'y_py_100hz', 'y_py_fix'])
    ax1[2].legend(['z_gt', 'z_py', 'z_py_100hz', 'z_py_fix'])
    #fig1.savefig(os.path.join("./plot","exp_{}_pos_{}_{}.png".format(exp_num, start, end)))
    fig1.savefig(os.path.join("./plot/ekf_fix","exp_{}_pos_fix".format(exp_num)))

    ang_gt = np.array(df_align.iloc[:,7:10]).reshape(-1,3)
    ang_py = np.array(df_ref.iloc[:,8:11]).reshape(-1,3)
    ang_cp = np.array(df_ref2.iloc[:,7:10]).reshape(-1,3)
    ang_py2 = np.array(df_ref3.iloc[:,8:11]).reshape(-1,3)
    ang_py3 = np.array(df_ref4.iloc[:,8:11]).reshape(-1,3)
    fig2, ax2 = plt.subplots(3,1, sharex= True, figsize=(20, 10))
    ax2[0].plot(t_gt, ang_gt[:,0])
    ax2[0].plot(t_, ang_py[:,0])
    #ax2[0].plot(t_2, ang_cp[:,0])
    ax2[0].plot(t_3, ang_py2[:,0])
    ax2[0].plot(t_4, ang_py3[:,0])

    ax2[1].plot(t_gt, ang_gt[:,1])
    ax2[1].plot(t_, ang_py[:,1])
    #ax2[1].plot(t_2, ang_cp[:,1])
    ax2[1].plot(t_3, ang_py2[:,1])
    ax2[1].plot(t_4, ang_py3[:,1])

    ax2[2].plot(t_gt, ang_gt[:,2])
    ax2[2].plot(t_, ang_py[:,2])
    #ax2[2].plot(t_2, ang_cp[:,2])
    ax2[2].plot(t_3, ang_py2[:,2])
    ax2[2].plot(t_4, ang_py3[:,2])
    ax2[0].set(xlabel='time (s)', ylabel=r'$\phi_n$ (rad)', title="Orientation")
    ax2[1].set(xlabel='time (s)', ylabel=r'$\theta_n$ (rad)', title="Orientation")
    ax2[2].set(xlabel='time (s)', ylabel=r'$\psi_n$ (rad)', title="Orientation")
    # ax2[0].legend([r'$\phi_n^x$', r'$\hat{\phi}_n^x$'])
    # ax2[1].legend([r'$\theta_n^y$',r'$\hat{\theta}_n^y$'])
    # ax2[2].legend([r'$\psi_n^z$', r'$\hat{\psi}_n^z$'])
    # ax2[0].legend(['roll_gt', 'roll_py', 'roll_cpp', 'roll_py_100hz'])
    # ax2[1].legend(['pitch_gt', 'pitch_py', 'pitch_cpp', 'pitch_py_100hz'])
    # ax2[2].legend(['yaw_gt', 'yaw_py', 'yaw_cpp', 'yaw_py_100hz'])
    ax2[0].legend(['roll_gt', 'roll_py', 'roll_py_100hz','roll_py_fix', ])
    ax2[1].legend(['pitch_gt', 'pitch_py', 'pitch_py_100hz','pitch_py_fix'])
    ax2[2].legend(['yaw_gt', 'yaw_py', 'yaw_py_100hz', 'yaw_py_fix'])
    #plt.show()
    fig2.savefig(os.path.join("./plot/ekf_fix","exp_{}_ori_fix.png".format(exp_num)))
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


    #### 3. plot velocity
    vel_gt = np.array(df_align.iloc[:,4:7]).reshape(-1,3)
    vel_py = np.array(df_ref.iloc[:,5:8]).reshape(-1,3)
    vel_cp = np.array(df_ref2.iloc[:,4:7]).reshape(-1,3)
    vel_py2 = np.array(df_ref3.iloc[:,5:8]).reshape(-1,3)
    vel_py3 = np.array(df_ref4.iloc[:,5:8]).reshape(-1,3)
    fig3, ax3 = plt.subplots(3,1, sharex= True, figsize=(20, 10))
    # ax3[0].plot(t_gt[start_1:end_1], vel_gt[start_1:end_1,0])
    # ax3[0].plot(t_[start:end], vel_py[start:end,0])
    # ax3[0].plot(t_2[start:end], vel_cp[start:end,0])
    # ax3[0].plot(t_3[start_2:end_2], vel_py2[start_2:end_2,0])
    # ax3[1].plot(t_gt[start_1:end_1], vel_gt[start_1:end_1,1])
    # ax3[1].plot(t_[start:end], vel_py[start:end,1])
    # ax3[1].plot(t_2[start:end], vel_cp[start:end,1])
    # ax3[1].plot(t_3[start_2:end_2], vel_py2[start_2:end_2,1])
    # ax3[2].plot(t_gt[start_1:end_1], vel_gt[start_1:end_1,2])
    # ax3[2].plot(t_[start:end], vel_py[start:end,2])
    # ax3[2].plot(t_2[start:end], vel_cp[start:end,2])
    # ax3[2].plot(t_3[start_2:end_2], vel_py2[start_2:end_2,2])
    ax3[0].plot(t_gt, vel_gt[:,0])
    ax3[0].plot(t_, vel_py[:,0])
    #ax3[0].plot(t_2, vel_cp[:,0])
    ax3[0].plot(t_3, vel_py2[:,0])
    ax3[0].plot(t_4, vel_py3[:,0])

    ax3[1].plot(t_gt, vel_gt[:,1])
    ax3[1].plot(t_, vel_py[:,1])
    #ax3[1].plot(t_2, vel_cp[:,1])
    ax3[1].plot(t_3, vel_py2[:,1])
    ax3[1].plot(t_4, vel_py3[:,1])

    ax3[2].plot(t_gt, vel_gt[:,2])
    ax3[2].plot(t_, vel_py[:,2])
    #ax3[2].plot(t_2, vel_cp[:,2])
    ax3[2].plot(t_3, vel_py2[:,2])
    ax3[2].plot(t_4, vel_py3[:,2])
    ax3[0].set(xlabel='time (s)', ylabel='v_x (m/s)', title="Speed")
    ax3[1].set(xlabel='time (s)', ylabel='v_y (m/s)', title="Speed")
    ax3[2].set(xlabel='time (s)', ylabel='v_z (m/s)', title="Speed")
    # ax3[0].legend(['v_x_gt', 'v_x_py', 'v_x_cpp', 'v_x_py_100hz'])
    # ax3[1].legend(['v_y_gt', 'v_y_py', 'v_y_cpp', 'v_y_py_100hz'])
    # ax3[2].legend(['v_z_gt', 'v_z_py', 'v_z_cpp', 'v_z_py_100hz'])
    ax3[0].legend(['v_x_gt', 'v_x_py', 'v_x_py_100hz', 'v_x_py_fix'])
    ax3[1].legend(['v_y_gt', 'v_y_py', 'v_y_py_100hz', 'v_y_py_fix'])
    ax3[2].legend(['v_z_gt', 'v_z_py', 'v_z_py_100hz', 'v_z_py_fix'])
    #fig3.savefig(os.path.join("./plot","exp_{}_vel_{}_{}.png".format(exp_num, start, end)))
    fig3.savefig(os.path.join("./plot/ekf_fix","exp_{}_vel_fix.png".format(exp_num)))

    assert False

    #### 4. plot bias
    #vel_gt = np.array(df_align.iloc[:,4:7]).reshape(-1,3)
    bias_py = np.array(df_ref.iloc[:,11:14]).reshape(-1,3)
    bias_cp = np.array(df_ref2.iloc[:,10:13]).reshape(-1,3)
    fig4, ax4 = plt.subplots(3,1, sharex= True, figsize=(20, 10))
    ax4[0].plot(t_, bias_py[:,0])
    ax4[0].plot(t_2, bias_cp[:,0])
    ax4[1].plot(t_, bias_py[:,1])
    ax4[1].plot(t_2, bias_cp[:,1])
    ax4[2].plot(t_, bias_py[:,2])
    ax4[2].plot(t_2, bias_cp[:,2])
    ax4[0].set(xlabel='time (s)', ylabel='$b_x (m/s^2)$', title="Bias Acc")
    ax4[1].set(xlabel='time (s)', ylabel='$b_y (m/s^2)$', title="Bias Acc")
    ax4[2].set(xlabel='time (s)', ylabel='$b_z (m/s^2)$', title="Bias Acc")
    ax4[0].legend(['bias_acc_x_py', 'bias_acc_x_cpp'])
    ax4[1].legend(['bias_acc_y_py', 'bias_acc_y_cpp'])
    ax4[2].legend(['bias_acc_z_py', 'bias_acc_z_cpp'])
    fig4.savefig(os.path.join("./plot","exp_{}_bias.png".format(exp_num)))



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
