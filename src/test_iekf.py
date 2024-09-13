import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from termcolor import cprint
import pandas as pd
from main_usv import USVParameters, USVDataset, USVArgs
from main_kitti import KITTIArgs, KITTIDataset, KITTIParameters
from utils_torch_filter import TORCHIEKF
from utils_numpy_filter import NUMPYIEKF as IEKF
from utils import prepare_data, create_folder
from utils_plot import results_filter
from utils import umeyama_alignment
import copy
import pickle


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
        interval = 100
        start_index = np.arange(1, N, interval)
        start_index = np.concatenate((start_index, [N]))
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
                acc = Rot[j-1].dot(u[j][3:6] - b_acc[j-1]) + iekf.g
                v[j] = v[j-1] + acc * dt_
                p[j] = p[j-1] + v[j-1]*dt_ + 1/2 * acc * dt_**2
                #print("dt : {}, a : {}, v : {}, p : {}".format(dt_,acc,v[j],p[j]))
                omega = u[j][:3] - b_omega[j-1]
                Rot[j] = Rot[j-1].dot(IEKF.so3exp(omega * dt_))
                # b_omega = b_omega_prev
                b_acc[j] = b_acc[j-1]
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
        
def get_imu_estimates(dataset_name, path_results):
    #  Obtain  estimates
    file_name = os.path.join(path_results, dataset_name + "_imu.p")
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
        file_name = os.path.join(dataset.path_results, dataset_name + "_imu.p")
        if not os.path.exists(file_name):
            print('No result for ' + dataset_name)
            continue

        print("\nResults for: " + dataset_name)
        
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, measurements_covs = get_imu_estimates(
            dataset_name, args.path_results)
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
    print(iekf.Q)

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
        measurements_covs = np.array([[iekf.cov_lat, iekf.cov_up]]*len(u))
        #print(measurements_covs)
        
        
        start_time = time.time()
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf.run(
            t, u, measurements_covs, v_gt, p_gt, N, ang_gt[0]
        )
        diff_time = time.time() - start_time

        print("Execution time: {:.2f} s (sequence time: {:.2f} s)".format(diff_time, t[-1] - t[0]))

        # 결과 저장
        mondict = {
            't': t, 'Rot': Rot, 'v': v, 'p': p, 'b_omega': b_omega, 'b_acc': b_acc,
            'Rot_c_i': Rot_c_i, 't_c_i': t_c_i,
            'measurements_covs': measurements_covs,
        }
        dataset.dump(mondict, args.path_results, dataset_name + "_filter.p")
    
if __name__ == "__main__" : 
    args = USVArgs()  
    dataset = USVDataset(args)  
    #args = KITTIArgs()
    #dataset = KITTIDataset(args)
    #run_filter(args, dataset)
    #results_filter(args, dataset)
    run_imu_dr(args, dataset)
    results_dr(args, dataset)