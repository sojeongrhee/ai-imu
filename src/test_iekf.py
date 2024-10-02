import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from termcolor import cprint
import pandas as pd
import math

from main_usv import USVParameters, USVDataset, USVArgs
from main_kitti import KITTIArgs, KITTIDataset, KITTIParameters
from utils_torch_filter import TORCHIEKF
from utils_numpy_filter import NUMPYIEKF as IEKF
from utils import prepare_data, create_folder
from utils_plot import results_filter
from utils import umeyama_alignment
from train_torch_filter import precompute_lost, prepare_loss_data

import copy
import pickle
from itertools import chain
from tqdm import tqdm

# def run_dr_bias() : 
#     datapath = "../dataset/sheco_data"
#     exp_list = [1,2,3]
#     for exp_num in exp_list : 
#         # Load GPS and IMU data
#         ekf_df = pd.read_csv(os.path.join(datapath,'ekf_ext_{}.csv'.format(exp_num)))
#         t = np.arrary(ekf_df.loc[:,'%time']).reshape(-1,1)
#         ang_gt = np.array(ekf_df.loc[:,['field.ori_r', 'field.ori_p', 'field.ori_y']]).reshape(-1,3)
#         p_gt = np.array(ekf_df.loc[:,['field.pos_x', 'field.pos_y', 'field.pos_z']]).reshape(-1,3)
#         v_gt = np.array(ekf_df.loc[:,['field.vel_x', 'field.vel_y', 'field.vel_z']]).reshape(-1,3)
#         u = np.array(ekf_df.loc[:,])
#         b_acc = 
#         b_gyr = 

def run_imu_dr(args, dataset) : 
    iekf = IEKF()
    #torch_iekf = TORCHIEKF()

    iekf.filter_parameters = USVParameters()  
    iekf.set_param_attr()
    
    for i in range(0, len(dataset.datasets)):
        dataset_name = dataset.dataset_name(i)
        if dataset_name not in dataset.odometry_benchmark.keys():
            continue
        if dataset_name not in ["merged_output_py_final_1","merged_output_py_final_2","merged_output_py_final_3"] : 
            continue
        # if dataset_name not in ["2011_09_30_drive_0027_extract","2011_09_30_drive_0033_extract","2011_09_30_drive_0034_extract"] : 
        #     continue
        print("Test filter on sequence: " + dataset_name)

        t, ang_gt, p_gt, v_gt, u = prepare_data(args, dataset, dataset_name, i, to_numpy=True)
        
        start_time = time.time()
        #Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, measurements_cov
        
        
        dt = t[1:] - t[:-1]  # (s)
        N = u.shape[0]
        ## init variables
        Rot = np.zeros((N, 3, 3))
        v = np.zeros((N, 3))
        p = np.zeros((N, 3))
        b_omega = np.zeros((N, 3))
        b_acc = np.zeros((N, 3))
        Rot_c_i = np.zeros((N, 3, 3))
        t_c_i = np.zeros((N, 3))
        Rot_c_i[0] = np.eye(3)
        
        # ang0 = ang_gt[0]
        # #Rot[0] = IEKF.from_rpy(ang0[0], ang0[1], ang0[2])
        # Rot[0] = IEKF.from_rpy(0, 0, ang0[2])
        # v[0] = v_gt[0]
        # b_acc[0] = -2*iekf.g
                
        measurements_covs = np.zeros((N,2))
        ## imu dr
        # Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P = \
        #         self.propagate(Rot[i-1], v[i-1], p[i-1], b_omega[i-1], b_acc[i-1], Rot_c_i[i-1],
        #                        t_c_i[i-1], P, u[i], dt[i-1])
        
        #  Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P = \
        #         self.update(Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P, u[i],
        #                     i, measurements_covs[i])
        interval = 200
        start_index = np.arange(0, N, interval)
        start_index = np.concatenate((start_index, [N]))
        #start_index = np.array([0,N])
        if args.add_extra : 
            print("with bias")
            u_bias , _ = dataset.get_extra_data(dataset_name)
            u_bias = u_bias.cpu().double().numpy()
            b_omega = u_bias[:,3:6]
            b_acc = u_bias[:,:3]
            #start_index = np.array([0, N])
        for i in range(len(start_index)-1) :        
            for j in range(start_index[i], start_index[i+1]) : 
                ### init 
                if j == start_index[i] : 
                    Rot_c_i[j] = np.eye(3)
                    ang0 = ang_gt[j]
                    #Rot[j] = IEKF.from_rpy(0, 0, ang0[2])
                    Rot[j] = IEKF.from_rpy(ang0[0], ang0[1], ang0[2])
                    v[j] = v_gt[j]
                    p[j] = p_gt[j]
                    #b_acc[j] = -2*iekf.g
                    #print("init rpy : {}, init rot : {}, init_v : {}".format(ang0, Rot[j], v[j]))
                    continue
                #print("u : {}, b_acc : {}, acc : {}, g : {}".format(u[j],b_acc[j-1], Rot[j-1].dot(u[j][3:6] - b_acc[j-1]),iekf.g))
                #print("rot : {}, rpy: {}".format(Rot[j-1], IEKF.to_rpy(Rot[j-1])))
                dt_ = dt[j-1]
                acc = Rot[j-1].dot(u[j][3:6] - b_acc[j-1]) - iekf.g
                v[j] = v[j-1] + acc * dt_
                p[j] = p[j-1] + v[j-1]*dt_ + 1/2 * acc * dt_**2
                #print("dt : {}, a : {}, v : {}, p : {}".format(dt_,acc,v[j],p[j]))
                print("v_gt: {}, v : {}".format(v_gt[j], v[j]))
                print("p_gt: {}, p : {}".format(p_gt[j], p[j]))
                omega = u[j][:3] - b_omega[j-1]
                Rot[j] = Rot[j-1].dot(IEKF.so3exp(omega * dt_))
                #b_omega[j] = b_omega[j-1]
                #b_acc[j] = b_acc[j-1]
                # Rot_c_i = Rot_c_i_prev
                # t_c_i = t_c_i_prev
        diff_time = time.time() - start_time

        print("Execution time: {:.2f} s (sequence time: {:.2f} s)".format(diff_time, t[-1] - t[0]))

        # 결과 저장
        mondict = {
            't': t, 'Rot': Rot, 'v': v, 'p': p, 'b_omega': b_omega, 'b_acc': b_acc,
            'Rot_c_i': Rot_c_i, 't_c_i': t_c_i,
            'measurements_covs': measurements_covs,
        }
        dataset.dump(mondict, args.path_results, dataset_name + "_imu.p")
        
def get_estimates(dataset_name, path_results, prefix):
    #  Obtain  estimates
    file_name = os.path.join(path_results, dataset_name + prefix)
    if not os.path.exists(file_name):
        print('No result for ' + dataset_name)
        return
    
    with open(file_name, "rb") as file_pi:
        pickle_dict = pickle.load(file_pi)
    mondict = pickle_dict
    Rot = mondict['Rot']
    v = mondict['v']
    p = mondict['p']
    b_omega = mondict['b_omega']
    b_acc = mondict['b_acc']
    Rot_c_i = mondict['Rot_c_i']
    t_c_i = mondict['t_c_i']
    measurements_covs = mondict['measurements_covs']
    return Rot, v, p , b_omega, b_acc, Rot_c_i, t_c_i, measurements_covs
        
def results_dr(args, dataset):
    for i in range(0, len(dataset.datasets)) : 
        plt.close('all')
        dataset_name = dataset.dataset_name(i)
        # if dataset_name not in ["2011_09_30_drive_0027_extract","2011_09_30_drive_0033_extract","2011_09_30_drive_0034_extract"] : 
        #     continue
        if dataset_name not in ["merged_output_py_final_1","merged_output_py_final_2","merged_output_py_final_3"] : 
            continue
        file_name = os.path.join(dataset.path_results, dataset_name + "_imu.p")
        if not os.path.exists(file_name):
            print('No result for ' + dataset_name)
            continue

        print("\nResults for: " + dataset_name)
        
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, measurements_covs = get_estimates(
            dataset_name, args.path_results, "_imu.p")
        t, ang_gt, p_gt, v_gt, u = dataset.get_data(dataset_name)
        p_gt = (p_gt - p_gt[0]).cpu().numpy()
        
        ang = np.zeros((Rot.shape[0], 3))
        #Rot_gt = torch.zeros((Rot.shape[0], 3, 3))
        ang_gt = ang_gt.cpu().numpy()
        for j in range(Rot.shape[0]):
            roll, pitch, yaw = TORCHIEKF.to_rpy(torch.from_numpy(Rot[j]))
            ang[j, 0] = roll.numpy()
            ang[j, 1] = pitch.numpy()
            ang[j, 2] = yaw.numpy()
        # # unwrap
        #     Rot_gt[j] = TORCHIEKF.from_rpy(torch.Tensor([ang_gt[j, 0]]),
        #                                 torch.Tensor([ang_gt[j, 1]]),
        #                                 torch.Tensor([ang_gt[j, 2]]))
        #     roll, pitch, yaw = TORCHIEKF.to_rpy(Rot_gt[j])
        #     ang_gt[j, 0] = roll.numpy()
        #     ang_gt[j, 1] = pitch.numpy()
        #     ang_gt[j, 2] = yaw.numpy()



        Rot_align, t_align, _ = umeyama_alignment(p_gt[:, :3].T, p[:, :3].T)
        p_align = (Rot_align.T.dot(p[:, :3].T)).T - Rot_align.T.dot(t_align)
        # orientation, bias gyro and bias accelerometer
        fig2, ax2 = plt.subplots(3,1, sharex= True, figsize=(20, 10))
        # position in plan
        fig3, ax3 = plt.subplots(figsize=(20, 10))
        # position in plan after alignment
        fig4, ax4 = plt.subplots(figsize=(20, 10))
        
        ax2[0].plot(t, ang_gt[:,0])
        ax2[0].plot(t, ang[:,0])
        ax2[1].plot(t, ang_gt[:,1])
        ax2[1].plot(t, ang[:,1])
        ax2[2].plot(t, ang_gt[:,2])
        ax2[2].plot(t, ang[:,2])

        ax3.plot(p_gt[:, 0], p_gt[:, 1])
        ax3.plot(p[:, 0], p[:, 1])
        ax3.axis('equal')
        ax4.plot(p_gt[:, 0], p_gt[:, 1])
        ax4.plot(p_align[:, 0], p_align[:, 1])
        ax4.axis('equal')
        
        # ax2[0].set(xlabel='time (s)', ylabel=r'$\phi_n, \theta_n, \psi_n$ (rad)',
        #             title="Orientation")
        ax2[0].set(xlabel='time (s)', ylabel=r'$\phi_n$ (rad)',
                    title="Orientation")
        ax2[1].set(xlabel='time (s)', ylabel=r'$\theta_n$ (rad)',
                    title="Orientation")
        ax2[2].set(xlabel='time (s)', ylabel=r'$\psi_n$ (rad)',
                    title="Orientation")
        ax3.set(xlabel=r'$p_n^x$ (m)', ylabel=r'$p_n^y$ (m)', title="Position on $xy$")
        ax4.set(xlabel=r'$p_n^x$ (m)', ylabel=r'$p_n^y$ (m)', title="Aligned position on $xy$")
        #ax2.grid()
        ax3.grid()
        ax4.grid()
        ax2[0].legend([r'$\phi_n^x$', r'$\hat{\phi}_n^x$'])
        ax2[1].legend([r'$\theta_n^y$',r'$\hat{\theta}_n^y$'])
        ax2[2].legend([r'$\psi_n^z$', r'$\hat{\psi}_n^z$'])
        ax3.legend(['ground-truth trajectory', 'proposed'])
        ax4.legend(['ground-truth trajectory', 'proposed'])
        
        folder_path = os.path.join(args.path_results, dataset_name)
        create_folder(folder_path)
        
        figs = [fig2, fig3, fig4]
        figs_name = ["orientation_bias","position_xy", "position_xy_aligned"]
        
        
        for l, fig in enumerate(figs):
            fig_name = figs_name[l]
            fig.savefig(os.path.join(folder_path, fig_name + "_imu.png"))
        
def compare_filter(args, dataset):

    for i in range(0, len(dataset.datasets)):
        plt.close('all')
        dataset_name = dataset.dataset_name(i)
        if dataset_name not in ["merged_output_final_1","merged_output_final_2","merged_output_final_3"] : 
            continue
        prop_name = os.path.join(dataset.path_results, dataset_name + "_prop.p")
        filter_name = os.path.join(dataset.path_results, dataset_name + "_filter.p")
        if not os.path.exists(filter_name):
            print('No result for ' + dataset_name)
            continue
        
        if not os.path.exists(prop_name):
            print('No result for ' + dataset_name)
            continue

        print("\nResults for: " + dataset_name)

        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, measurements_covs = get_estimates(
            dataset_name, args.path_results, "_filter.p")
        
        Rot_b, v_b, p_b, b_omega_b, b_acc_b, Rot_c_i_b, t_c_i_b, measurements_covs = get_estimates(
            dataset_name,args.path_results, "_prop.p")


        # get data
        t, ang_gt, p_gt, v_gt, u = dataset.get_data(dataset_name)
        # get data for nets
        u_normalized = dataset.normalize(u).numpy()
        # shift for better viewing
        u_normalized[:, [0, 3]] += 5
        u_normalized[:, [2, 5]] -= 5

        t = (t - t[0]).numpy()
        u = u.cpu().numpy()
        ang_gt = ang_gt.cpu().numpy()
        v_gt = v_gt.cpu().numpy()
        p_gt = (p_gt - p_gt[0]).cpu().numpy()
        print("Total sequence time: {:.2f} s".format(t[-1]))

        ang = np.zeros((Rot.shape[0], 3))
        ang_b = np.zeros((Rot.shape[0], 3))

        Rot_gt = torch.zeros((Rot.shape[0], 3, 3))
        for j in range(Rot.shape[0]):
            roll, pitch, yaw = TORCHIEKF.to_rpy(torch.from_numpy(Rot[j]))
            ang[j, 0] = roll.numpy()
            ang[j, 1] = pitch.numpy()
            ang[j, 2] = yaw.numpy()

            roll, pitch, yaw = TORCHIEKF.to_rpy(torch.from_numpy(Rot_b[j]))
            ang_b[j, 0] = roll.numpy()
            ang_b[j, 1] = pitch.numpy()
            ang_b[j, 2] = yaw.numpy()
        # unwrap
            Rot_gt[j] = TORCHIEKF.from_rpy(torch.Tensor([ang_gt[j, 0]]),
                                        torch.Tensor([ang_gt[j, 1]]),
                                        torch.Tensor([ang_gt[j, 2]]))
            roll, pitch, yaw = TORCHIEKF.to_rpy(Rot_gt[j])
            ang_gt[j, 0] = roll.numpy()
            ang_gt[j, 1] = pitch.numpy()
            ang_gt[j, 2] = yaw.numpy()

        #Rot_align, t_align, _ = umeyama_alignment(p_gt[:, :3].T, p[:, :3].T)
        #p_align = (Rot_align.T.dot(p[:, :3].T)).T - Rot_align.T.dot(t_align)
        v_norm = np.sqrt(np.sum(v_gt ** 2, 1))
        v_norm /= np.max(v_norm)

        # # Compute various errors
        # error_p = np.abs(p_gt - p)
        # # MATE
        # mate_xy = np.mean(error_p[:, :2], 1)
        # mate_z = error_p[:, 2]

        # # CATE
        # cate_xy = np.cumsum(mate_xy)
        # cate_z = np.cumsum(mate_z)

        # # RMSE
        # rmse_xy = 1 / 2 * np.sqrt(error_p[:, 0] ** 2 + error_p[:, 1] ** 2)
        # rmse_z = error_p[:, 2]

        RotT = torch.from_numpy(Rot).float().transpose(-1, -2)
        RotT_b = torch.from_numpy(Rot_b).float().transpose(-1, -2)
        v_r = (RotT.matmul(torch.from_numpy(v).float().unsqueeze(-1)).squeeze()).numpy()
        v_r_b = (RotT_b.matmul(torch.from_numpy(v_b).float().unsqueeze(-1)).squeeze()).numpy()
        v_r_gt = (Rot_gt.transpose(-1, -2).matmul(
            torch.from_numpy(v_gt).float().unsqueeze(-1)).squeeze()).numpy()

        p_r = (RotT.matmul(torch.from_numpy(p).float().unsqueeze(-1)).squeeze()).numpy()
        p_r_b = (RotT_b.matmul(torch.from_numpy(p_b).float().unsqueeze(-1)).squeeze()).numpy()
        p_bis = (Rot_gt.matmul(torch.from_numpy(p_r).float().unsqueeze(-1)).squeeze()).numpy()
        error_p = p_gt - p_bis

        # plot and save plot
        folder_path = os.path.join(args.path_results, dataset_name, 'iekf')
        create_folder(folder_path)

        # position, velocity and velocity in body frame
        fig1, axs1 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
        # orientation, bias gyro and bias accelerometer
        fig2, axs2 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
        # position in plan
        fig3, ax3 = plt.subplots(figsize=(20, 10))
        # position in plan after alignment
        #fig4, ax4 = plt.subplots(figsize=(20, 10))
        # #  Measurement covariance in log scale and normalized inputs
        # fig5, axs5 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
        # # input: gyro, accelerometer
        # fig6, axs6 = plt.subplots(2, 1, sharex=True, figsize=(20, 10))
        # # errors: MATE, CATE  RMSE
        # fig7, axs7 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))

        axs1[0].plot(t, p_gt)
        axs1[0].plot(t, p)
        axs1[0].plot(t, p_b)
        axs1[1].plot(t, v_gt)
        axs1[1].plot(t, v)
        axs1[1].plot(t, v_b)
        axs1[2].plot(t, v_r_gt)
        axs1[2].plot(t, v_r)
        axs1[2].plot(t, v_r_b)
        axs2[0].plot(t, ang_gt)
        axs2[0].plot(t, ang)
        axs2[0].plot(t, ang_b)
        axs2[1].plot(t, b_omega)
        axs2[1].plot(t, b_omega_b)
        axs2[2].plot(t, b_acc)
        axs2[2].plot(t, b_acc_b)
        ax3.plot(p_gt[:, 0], p_gt[:, 1])
        ax3.plot(p[:, 0], p[:, 1])
        ax3.plot(p_b[:, 0], p_b[:, 1])
        ax3.axis('equal')
        # ax4.plot(p_gt[:, 0], p_gt[:, 1])
        # ax4.plot(p_align[:, 0], p_align[:, 1])
        # ax4.axis('equal')

        # axs5[0].plot(t, np.log10(measurements_covs))
        # axs5[1].plot(t, u_normalized[:, :3])
        # axs5[2].plot(t, u_normalized[:, 3:])

        # axs6[0].plot(t, u[:, :3])
        # axs6[1].plot(t, u[:, 3:6])

        # axs7[0].plot(t, mate_xy)
        # axs7[0].plot(t, mate_z)
        # axs7[0].plot(t, rmse_xy)
        # axs7[0].plot(t, rmse_z)
        # axs7[1].plot(t, cate_xy)
        # axs7[1].plot(t, cate_z)
        # axs7[2].plot(t, error_p)

        axs1[0].set(xlabel='time (s)', ylabel='$\mathbf{p}_n$ (m)', title="Position")
        axs1[1].set(xlabel='time (s)', ylabel='$\mathbf{v}_n$ (m/s)', title="Velocity")
        axs1[2].set(xlabel='time (s)', ylabel='$\mathbf{R}_n^T \mathbf{v}_n$ (m/s)',
                    title="Velocity in body frame")
        axs2[0].set(xlabel='time (s)', ylabel=r'$\phi_n, \theta_n, \psi_n$ (rad)',
                    title="Orientation")
        axs2[1].set(xlabel='time (s)', ylabel=r'$\mathbf{b}_{n}^{\mathbf{\omega}}$ (rad/s)',
                    title="Bias gyro")
        axs2[2].set(xlabel='time (s)', ylabel=r'$\mathbf{b}_{n}^{\mathbf{a}}$ (m/$\mathrm{s}^2$)',
                    title="Bias accelerometer")
        ax3.set(xlabel=r'$p_n^x$ (m)', ylabel=r'$p_n^y$ (m)', title="Position on $xy$")
        #ax4.set(xlabel=r'$p_n^x$ (m)', ylabel=r'$p_n^y$ (m)', title="Aligned position on $xy$")
        # axs5[0].set(xlabel='time (s)', ylabel=r' $\mathrm{cov}(\mathbf{y}_{n})$ (log scale)',
        #              title="Covariance on the zero lateral and vertical velocity measurements (log "
        #                    "scale)")
        # axs5[1].set(xlabel='time (s)', ylabel=r'Normalized gyro measurements',
        #              title="Normalized gyro measurements")
        # axs5[2].set(xlabel='time (s)', ylabel=r'Normalized accelerometer measurements',
        #            title="Normalized accelerometer measurements")
        # axs6[0].set(xlabel='time (s)', ylabel=r'$\omega^x_n, \omega^y_n, \omega^z_n$ (rad/s)',
        #             title="Gyrometer")
        # axs6[1].set(xlabel='time (s)', ylabel=r'$a^x_n, a^y_n, a^z_n$ (m/$\mathrm{s}^2$)',
        #             title="Accelerometer")
        # axs7[0].set(xlabel='time (s)', ylabel=r'$|| \mathbf{p}_{n}-\hat{\mathbf{p}}_{n} ||$ (m)',
        #             title="Mean Absolute Trajectory Error (MATE) and Root Mean Square Error (RMSE)")
        # axs7[1].set(xlabel='time (s)',
        #             ylabel=r'$\Sigma_{i=0}^{n} || \mathbf{p}_{i}-\hat{\mathbf{p}}_{i} ||$ (m)',
        #             title="Cumulative Absolute Trajectory Error (CATE)")
        # axs7[2].set(xlabel='time (s)', ylabel=r' $\mathbf{\xi}_{n}^{\mathbf{p}}$',
        #             title="$SE(3)$ error on position")

        #for ax in chain(axs1, axs2, axs5, axs6, axs7):
        for ax in chain(axs1, axs2):
            ax.grid()
        ax3.grid()
        #ax4.grid()
        axs1[0].legend(
            ['$p_n^x$', '$p_n^y$', '$p_n^z$', '$\hat{p}_n^x$', '$\hat{p}_n^y$', '$\hat{p}_n^z$','$\hat{p}_nprop^x$', '$\hat{p}_nprop^y$', '$\hat{p}_nprop^z$'])
        axs1[1].legend(
            ['$v_n^x$', '$v_n^y$', '$v_n^z$', '$\hat{v}_n^x$', '$\hat{v}_n^y$', '$\hat{v}_n^z$','$\hat{v}_nprop^x$', '$\hat{v}_nprop^y$', '$\hat{v}_nprop^z$'])
        axs1[2].legend(
            ['$v_n^x$', '$v_n^y$', '$v_n^z$', '$\hat{v}_n^x$', '$\hat{v}_n^y$', '$\hat{v}_n^z$','$\hat{v}_nprop^x$', '$\hat{v}_nprop^y$', '$\hat{v}_nprop^z$'])
        axs2[0].legend([r'$\phi_n^x$', r'$\theta_n^y$', r'$\psi_n^z$', r'$\hat{\phi}_n^x$',r'$\hat{\theta}_n^y$'
                        r'$\hat{\psi}_n^z$',r'$\hat{\phi}_nprop^x$', r'$\hat{\theta}_nprop^y$',r'$\hat{\psi}_nprop^z$'])
        axs2[1].legend(
            ['$b_n^x$', '$b_n^y$', '$b_n^z$', '$\hat{b}_n^x$', '$\hat{b}_n^y$', '$\hat{b}_n^z$','$\hat{b}_nprop^x$', '$\hat{b}_nprop^y$', '$\hat{b}_nprop^z$'])
        axs2[2].legend(
            ['$b_n^x$', '$b_n^y$', '$b_n^z$', '$\hat{b}_n^x$', '$\hat{b}_n^y$', '$\hat{b}_n^z$','$\hat{b}_nprop^x$', '$\hat{b}_nprop^y$', '$\hat{b}_nprop^z$'])
        ax3.legend(['ground-truth trajectory', 'proposed', 'propagated'])
        #ax4.legend(['ground-truth trajectory', 'proposed'])
        # axs5[0].legend(['zero lateral velocity', 'zero vertical velocity'])
        # axs6[0].legend(['$\omega_n^x$', '$\omega_n^y$', '$\omega_n^z$'])
        # axs6[1].legend(['$a_n^x$', '$a_n^y$', '$a_n^z$'])
        # if u.shape[1] > 6:
        #     axs6[2].legend(['$m_n^x$', '$m_n^y$', '$m_n^z$'])
        # axs7[0].legend(['MATE xy', 'MATE z', 'RMSE xy', 'RMSE z'])
        # axs7[1].legend(['CATE xy', 'CATE z'])

        # save figures
        #figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, ]
        #figs_name = ["position_velocity", "orientation_bias", "position_xy", "position_xy_aligned",
        #            "measurements_covs", "imu", "errors", "errors2"]
        figs = [fig1, fig2, fig3]
        figs_name = ["position_velocity", "orientation_bias", "position_xy"]
        for l, fig in enumerate(figs):
            fig_name = figs_name[l]
            fig.savefig(os.path.join(folder_path, fig_name + ".png"))

        plt.show(block=True)

    
def run_filter(args, dataset) : 
    iekf = IEKF()
    #torch_iekf = TORCHIEKF()

    iekf.filter_parameters = USVParameters()  
    iekf.set_param_attr()
    #torch_iekf.filter_parameters = USVParameters() 
    #torch_iekf.set_param_attr()

    #torch_iekf.load(args, dataset)
    #iekf.set_learned_covariance(torch_iekf)
    
    
    ## use fixed covariance 
    #print(iekf.Q)

    for i in range(0, len(dataset.datasets)):
        dataset_name = dataset.dataset_name(i)
        if dataset_name not in dataset.odometry_benchmark.keys():
            continue

        print("Test filter on sequence: " + dataset_name)

        # 데이터 준비 (USV 데이터에 맞게 조정)
        t, ang_gt, p_gt, v_gt, u = prepare_data(args, dataset, dataset_name, i, to_numpy=True)
        N = None
        #u_t = torch.from_numpy(u).double()
        #measurements_covs = torch_iekf.forward_nets(u_t)
        #measurements_covs = measurements_covs.detach().numpy()
        #measurements_covs = np.array([[iekf.cov_lat, iekf.cov_up]]*len(u))
        measurements_covs = np.array([[1e12,1e12]]*len(u))
        #print(measurements_covs)
        
        
        start_time = time.time()
        # Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf.run(
        #     t, u, measurements_covs, v_gt, p_gt, N, ang_gt[0]
        # )

        dt = t[1:] - t[:-1]  # (s)
        if N is None:
            N = u.shape[0]
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P = iekf.init_run(dt, u, p_gt, v_gt,
                                       ang_gt[0], N)
        if args.add_extra : 
            print("with bias")
            u_bias , _ = dataset.get_extra_data(dataset_name)
            u_bias = u_bias.cpu().double().numpy()
            #b_omega[0] = u_bias[0,3:6]
            #b_acc[0] = u_bias[0,:3]
            #b_acc[0] = -2 * iekf.g

        Rot_b, v_b, p_b, b_omega_b, b_acc_b, Rot_c_i_b, t_c_i_b = iekf.init_saved_state(dt, N, ang_gt[0])
        # Rot_b[0] = iekf.from_rpy(ang_gt[0,0], ang_gt[0,1], ang_gt[0,2])
        # v_b[0] = v_gt[0]

        interval = 28
        start_index = np.arange(0, N, interval)
        start_index = np.concatenate((start_index, [N]))
        #start_index = np.array([0,N])
        for j in range(len(start_index)-1) : 
            for k in range(start_index[j], start_index[j+1]) :
                if k == start_index[j] : 
                    Rot_c_i[k] = np.eye(3)
                    Rot[k] = iekf.from_rpy(ang_gt[k,0], ang_gt[k,1], ang_gt[k,2])
                    v[k] = v_gt[k]
                    p[k] = p_gt[k]
                    
                    Rot_b[k] = iekf.from_rpy(ang_gt[k,0], ang_gt[k,1], ang_gt[k,2])
                    Rot_c_i_b[k] = Rot_c_i[k]
                    v_b[k] = v_gt[k]
                    p_b[k] = p_gt[k]
                    
                    b_omega[k] = u_bias[k, 3:6]
                    b_omega_b[k] = u_bias[k, 3:6]
                    b_acc[k] = u_bias[k, :3]
                    b_acc_b[k] = u_bias[k, :3]
                    continue 
                
                Rot[k], v[k], p[k], b_omega[k], b_acc[k], Rot_c_i[k], t_c_i[k], P = \
                iekf.propagate(Rot[k-1], v[k-1], p[k-1], b_omega[k-1], b_acc[k-1], Rot_c_i[k-1],
                    t_c_i[k-1], P, u[k], dt[k-1])

                Rot_b[k] = Rot[k]
                v_b[k] = v[k]
                p_b[k] = p[k]
                b_omega_b[k] = b_omega[k]
                b_acc_b[k] = b_acc[k]
                Rot_c_i_b[k] = Rot_c_i[k]
                t_c_i_b[k] = t_c_i[k]

                Rot[k], v[k], p[k], b_omega[k], b_acc[k], Rot_c_i[k], t_c_i[k], P = \
                iekf.update(Rot[k], v[k], p[k], b_omega[k], b_acc[k], Rot_c_i[k], t_c_i[k], P, u[k],
                    k, measurements_covs[k])
                # correct numerical error every second
                if k % iekf.n_normalize_rot == 0:
                    Rot[k] = iekf.normalize_rot(Rot[k])
                # correct numerical error every 10 seconds
                if k % iekf.n_normalize_rot_c_i == 0:
                    Rot_c_i[k] = iekf.normalize_rot(Rot_c_i[k])

        diff_time = time.time() - start_time

        print("Execution time: {:.2f} s (sequence time: {:.2f} s)".format(diff_time, t[-1] - t[0]))
        print("diff_p : ",np.sum(p - p_b),"diff_v", np.sum(v-v_b),"diff_rot",np.sum(Rot - Rot_b))
        # 결과 저장
        mondict = {
            't': t, 'Rot': Rot, 'v': v, 'p': p, 'b_omega': b_omega, 'b_acc': b_acc,
            'Rot_c_i': Rot_c_i, 't_c_i': t_c_i,
            'measurements_covs': measurements_covs,
        }
        
        mondict_prop = {
            't': t, 'Rot': Rot_b, 'v': v_b, 'p': p_b, 'b_omega': b_omega_b, 'b_acc': b_acc_b,
            'Rot_c_i': Rot_c_i_b, 't_c_i': t_c_i_b,
            'measurements_covs': measurements_covs,
        }
        dataset.dump(mondict, args.path_results, dataset_name + "_filter.p")
        dataset.dump(mondict_prop, args.path_results, dataset_name + "_prop.p")
    
def plot_iekf_loss(args, dataset) : 
    #iekf = IEKF()
    iekf = TORCHIEKF()
    criterion = torch.nn.MSELoss(reduction="mean")
    iekf.filter_parameters = USVParameters()  
    iekf.set_param_attr()
    loss_res = []
    for i, (dataset_name, Ns) in enumerate(dataset.datasets_train_filter.items()):
        #if dataset_name not in ["merged_output_final_2","merged_output_final_3"] : 
        #    continue
        if dataset_name not in ["merged_output_py_final_2","merged_output_py_final_3"] : 
            continue
        seq_dim = args.seq_dim
        # get data with trainable instant
        # t, ang_gt, p_gt, v_gt, u, N0 = prepare_data_filter(dataset, dataset_name, Ns,
        #                                                         iekf, seq_dim)
        t, ang_gt, p_gt, v_gt,  u = dataset.get_data(dataset_name)
        t = t[Ns[0]: Ns[1]]
        ang_gt = ang_gt[Ns[0]: Ns[1]]
        p_gt = (p_gt[Ns[0]: Ns[1]] - p_gt[Ns[0]])
        v_gt = v_gt[Ns[0]: Ns[1]]
        u = u[Ns[0]: Ns[1]]
        prepare_loss_data(args, dataset)
        iekf.g = torch.Tensor(iekf.g).double().to('cuda')
        
        list_rpe = dataset.list_rpe[dataset_name]
        list_rpe[2] = list_rpe[2].to('cuda')
        interval = 100
        loss_tmp = []
        for start_idx in tqdm(range(Ns[0], Ns[1]-seq_dim, interval)) :
            N0 = start_idx - Ns[0]
            N = N0 + seq_dim
            t_ = t[N0: N].double().to('cuda')
            ang_gt_ = ang_gt[N0: N].double().to('cuda')
            p_gt_ = (p_gt[N0: N] - p_gt[N0]).double().to('cuda')
            v_gt_ = v_gt[N0: N].double().to('cuda')
            u_ = u[N0: N].double().to('cuda')
            # add noise
            #u = dataset.add_noise(u)
            iekf.set_Q()
            measurements_covs = torch.Tensor([[1e5, 1e6]]*len(u)).double().to('cuda')
            Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf.run(t_, u_, measurements_covs,
                                                        v_gt_, p_gt_, t_.shape[0],
                                                        ang_gt_[0])
            
        
            #print("t, ang_gt, p_gt, v_gt, u :", t, ang_gt, p_gt, v_gt, u )
            
            
            # loss = mini_batch_step(dataset, dataset_name, iekf,
            #                 dataset.list_rpe[dataset_name], t, ang_gt, p_gt, v_gt, u, N0)
            delta_p, delta_p_gt, min_tmp, max_tmp = precompute_lost(Rot, p, list_rpe, N0)
            #print("length : {}".format(len(delta_p)))
            if delta_p is None : 
                loss = -1
                min_tmp = -1
                max_tmp = -1
            else : 
                loss = criterion(delta_p, delta_p_gt).detach().item()
                min_tmp = min_tmp.cpu()
                max_tmp = max_tmp.cpu()
            loss_tmp.append([start_idx,loss,min_tmp, max_tmp])
            print("loss {} : {} ".format(i, loss))
        # plot loss
        folder_path = os.path.join(args.path_results, dataset_name, 'loss')
        create_folder(folder_path)
        fig1, ax1 = plt.subplots(figsize=(20, 10))
        fig2, ax2 = plt.subplots(figsize=(20, 10))
        loss_tmp = np.array(loss_tmp)
        ax1.plot(loss_tmp[:,0], loss_tmp[:,1])
        fig1.savefig(os.path.join(folder_path, "total_loss.png"))
        ax2.plot(loss_tmp[:,0], loss_tmp[:,2])
        ax2.plot(loss_tmp[:,0], loss_tmp[:,3])
        fig2.savefig(os.path.join(folder_path, "length_per_loss.png"))
            
    
if __name__ == "__main__" : 
    #torch.set_default_device('cuda') 
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    args = USVArgs()  
    dataset = USVDataset(args)  
    #args = KITTIArgs()
    #dataset = KITTIDataset(args)
    #run_filter(args, dataset)
    #results_filter(args, dataset)
    #compare_filter(args, dataset)
    #run_imu_dr(args, dataset)
    #results_dr(args, dataset)
    plot_iekf_loss(args, dataset)