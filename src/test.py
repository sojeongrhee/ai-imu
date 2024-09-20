import matplotlib
import matplotlib.pyplot as plt
import os 
import pandas as pd
import csv
import numpy as np
import torch
from main_kitti import KITTIArgs, KITTIDataset, KITTIParameters
from main_usv import USVDataset, USVArgs
from utils_torch_filter import TORCHIEKF
from utils import prepare_data

def plot_gt(args, dataset) : 
    # 16599, 53663
    #dataset_list = ['2011_09_30_drive_0033_extract']
    dataset_list = ['2011_09_30_drive_0028_extract']
    for dataset_name in dataset_list : 
        t, ang_gt, p_gt, v_gt, u = dataset.get_data(dataset_name)
        t = (t - t[0]).numpy()
        u = u.cpu().numpy()
        ang_gt = ang_gt.cpu().numpy()
        v_gt = v_gt.cpu().numpy()
        p_gt = (p_gt - p_gt[0]).cpu().numpy()
        print(p_gt.shape)
        fig, ax = plt.subplots()
        start = 9250 
        end = 9450
        #ax.plot(p_gt[start:end, 0], p_gt[start:end, 1])
        ax.plot(p_gt[start:end, 0], p_gt[start:end, 1], marker='x')
        fig.savefig(os.path.join("./plot", "{}_{}.png".format(start, end)))
        #plt.show()

def plot_sc(args, dataset) :
    exp_num = 1 
    # 29912
    csv_path = "../dataset/sheco_data"
    csv_name = "sample{}.csv".format(exp_num)
    #csv_name = "ekf_output{}.csv"
    df_gt = pd.read_csv(os.path.join(csv_path, csv_name))
    x_ = np.array(df_gt.iloc[:,1]).reshape(-1,1)
    y_ = np.array(df_gt.iloc[:,2]).reshape(-1,1)
    fig, ax = plt.subplots()
    print(x_.shape)
    start = 5500 
    end = 5600
    ax.plot(x_[start:end], y_[start:end], marker='x')
    #ax.plot(x_[start:end], y_[start:end],)
    fig.savefig(os.path.join("./plot", "{}_{}.png".format(start, end)))

def plot_timestamp(folder, name, offset=None) : 
    if offset is None : 
        offset = np.zeros(3)
    exp_num = 2
    start = 9
    end = 15
    data_path = "../dataset/sheco_data"
    
    imu_name = "imu{}.bag".format(exp_num)
    gps_name = "gt{}.bag".format(exp_num)
    ref_name = "sample{}.csv".format(exp_num)
    ekf_name = "{}{}.csv".format(name,exp_num)
    df_imu = pd.read_csv(os.path.join(data_path, imu_name))
    df_gps = pd.read_csv(os.path.join(data_path, gps_name))
    df_ekf = pd.read_csv(os.path.join(data_path, ekf_name))
    df_ref = pd.read_csv(os.path.join(data_path, ref_name))
    imu_time = np.array(df_imu.iloc[:,0]).reshape(-1,1)
    gps_time = np.array(df_gps.iloc[:,0]).reshape(-1,1)
    ekf_time = np.array(df_ekf.iloc[:,0]).reshape(-1,1) - offset[exp_num-1]
    ref_time = np.array(df_ref.iloc[:,0]).reshape(-1,1)
    print(gps_time[:5])
    print(ekf_time[start:start+5])
    print(ref_time[:5])
    print(len(imu_time), len(gps_time), len(ekf_time))
    gps_frame = [[1] if frame=='smc_plus' else [2] for frame in df_gps.iloc[:,3]]
    imu_time = np.concatenate((imu_time, [[0]]*len(imu_time), [[-1]]*len(imu_time)),axis=1)
    gps_time = np.concatenate((gps_time, gps_frame, [[-1]]*len(gps_time)),axis=1)
    ekf_time = np.concatenate((ekf_time, [[3]]*len(ekf_time), np.arange(len(ekf_time)).reshape(-1,1)),axis=1)
    merged = np.vstack((np.vstack((imu_time, gps_time)),ekf_time))
    print(len(merged))
    sorted_indices = np.argsort(merged[:,0])
    sorted_list = merged[sorted_indices]
    sorted_list = sorted_list-np.array([sorted_list[0,0],0,0])
    fig1, ax1 = plt.subplots()
    start_ = np.where(sorted_list[:,2]==start)[0][0]
    end_ = np.where(sorted_list[:,2]==end)[0][0]
    print(start, end)
    ax1.plot(sorted_list[start_:end_,0], sorted_list[start_:end_,1], marker='o')
    ax1.set_yticks([0,1,2,3])
    ax1.set_yticklabels(['imu','gps_2000','gps_plus','ekf'])
    #ax.plot(x_[stat:end], y_[start:end])
    fig1.savefig(os.path.join("./plot/exp{}".format(exp_num),folder, "{}({})_{}({}).png".format(start,start_,end,end_)))
    
    fig2, ax2 = plt.subplots()
    x_ = np.array(df_ekf.iloc[:,1]).reshape(-1,1)
    y_ = np.array(df_ekf.iloc[:,2]).reshape(-1,1)
    ax2.plot(x_[start:end], y_[start:end], marker='x')
    #ax.plot(x_[start:end], y_[start:end],)
    fig2.savefig(os.path.join("./plot/exp{}".format(exp_num),folder, "{}_{}.png".format(start, end)))

def compute_dist(Rot, p):
    Rot = Rot[::10]
    p = p[::10]
    #print("Rot, p : ", Rot[0:5], p[0:5])

    step_size = 10  # every second
    distances = np.zeros(p.shape[0])
    distances_xy = np.zeros(p.shape[0])
    dp = p[1:] - p[:-1]  #  this must be ground truth
    #print("dp : ", dp)
    distances[1:] = dp.norm(dim=1).cumsum(0).numpy()
    dp = dp[:,:2]
    distances_xy[1:] = dp.norm(dim=1).cumsum(0).numpy()
    print("distances diff : ", distances[-1]-distances_xy[-1])
    return distances[-1]

def compute_subseq(Rot, p):
    list_rpe = [[], [], []]
    Rot = Rot[::10]
    p = p[::10]
    #print("Rot, p : ", Rot[0:5], p[0:5])

    step_size = 10  # every second
    distances = np.zeros(p.shape[0])
    dp = p[1:] - p[:-1]  #  this must be ground truth
    #print("dp : ", dp)
    distances[1:] = dp.norm(dim=1).cumsum(0).numpy()

    seq_lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    k_max = int(Rot.shape[0] / step_size) - 1
    num = [0,0,0,0,0,0,0,0]
    for k in range(0, k_max):
        idx_0 = k * step_size
        #print("idx_0 : ", idx_0)
        #print("distances[-1]", distances[-1])
        idx = 0 
        for seq_length in seq_lengths:
            if seq_length + distances[idx_0] > distances[-1]:
                #print("3333333333")
                idx+=1
                continue
            num[idx] += 1
            #print("1111111111111111")
            idx_shift = np.searchsorted(distances[idx_0:], distances[idx_0] + seq_length)
            idx_end = idx_0 + idx_shift
            #print("22222222222222222")
            list_rpe[0].append(idx_0)
            #print("list_rpe[0]", list_rpe[0])
            list_rpe[1].append(idx_end)
            #print("distances : ", distances[idx_0], distances[-1])
            idx+= 1
    idxs_0 = list_rpe[0]
    idxs_end = list_rpe[1]
    delta_p = Rot[idxs_0].transpose(-1, -2).matmul(
            ((p[idxs_end] - p[idxs_0]).float()).unsqueeze(-1)).squeeze()
    list_rpe[2] = delta_p
    print(len(list_rpe[0]))
    print(num)
        #print("list_rpe : ", list_rpe[0], list_rpe[1], list_rpe[2])
    return list_rpe

def plot_distance(args, dataset):
    list_rpe = {}
    for dataset_name, Ns in dataset.datasets_train_filter.items():
        t, ang_gt, p_gt, v_gt, u = prepare_data(args, dataset, dataset_name, 0)
        p_gt = p_gt.double()
        Ns[1] = t.shape[0] if Ns[1] is None else Ns[1]
        print(Ns)
        Rot_gt = torch.zeros(Ns[1], 3, 3)
        for k in range(Ns[1]):
            ang_k = ang_gt[k]
            Rot_gt[k] = TORCHIEKF.from_rpy(ang_k[0], ang_k[1], ang_k[2]).double()
        list_rpe[dataset_name] = compute_dist(Rot_gt[:Ns[1]], p_gt[:Ns[1]])
        #list_rpe[dataset_name] = compute_subseq(Rot_gt[:Ns[1]], p_gt[:Ns[1]])

    #for k, v in list_rpe.items() : 
    #    print(k, len(v[0]))

    list_rpe_validation = {}
    for dataset_name, Ns in dataset.datasets_validatation_filter.items():
        t, ang_gt, p_gt, v_gt, u = prepare_data(args, dataset, dataset_name, 0)
        p_gt = p_gt.double()
        Rot_gt = torch.zeros(Ns[1], 3, 3)
        print(Ns)
        for k in range(Ns[1]):
            ang_k = ang_gt[k]
            Rot_gt[k] = TORCHIEKF.from_rpy(ang_k[0], ang_k[1], ang_k[2]).double()
        list_rpe_validation[dataset_name] = compute_dist(Rot_gt[:Ns[1]], p_gt[:Ns[1]])
        #list_rpe_validation[dataset_name] = compute_subseq(Rot_gt[:Ns[1]], p_gt[:Ns[1]])
    #print(list_rpe_validation)
    #for k, v in list_rpe_validation.items() : 
    #    print(k, len(v[0]))

def plot_dt_sc() : 
    data_path = "../dataset/sheco_data"
    exp_list = [1,2,3]
    for exp_num in exp_list : 
        ekf_name = "sample{}.csv".format(exp_num)
        df_ekf = pd.read_csv(os.path.join(data_path, ekf_name))
        t = torch.tensor(np.array(df_ekf.iloc[:,0 ]))
        dt = t[1:]-t[:-1]
        fig1, ax1 = plt.subplots()
        ax1.hist(dt, bins=50, density=True, histtype='stepfilled',cumulative=True, color='blue')
        fig1.savefig(os.path.join("./plot", "exp{}_dt.png".format(exp_num)))
        
        
def plot_dist_sc() : 
    data_path = "../dataset/sheco_data"
    exp_list = [1,2,3]
    for exp_num in exp_list : 

        ekf_name = "sample{}.csv".format(exp_num)
        df_ekf = pd.read_csv(os.path.join(data_path, ekf_name))
        p = torch.tensor(np.array(df_ekf.iloc[:,1:4 ]).reshape(-1,3))
        ekf_rpy = torch.tensor(np.array(df_ekf.iloc[:,7: ]).reshape(-1,3))
        Rot = torch.zeros(p.shape[0], 3, 3)
        for k in range(p.shape[0]):
            Rot[k] = TORCHIEKF.from_rpy(ekf_rpy[k,0], ekf_rpy[k,1], ekf_rpy[k,2]).double()
        Rot = Rot[::3]
        p = p[::3]
        #print("Rot, p : ", Rot[0:5], p[0:5])

        step_size = 10  # every second
        distances = np.zeros(p.shape[0])
        distances_xy = np.zeros(p.shape[0])
        dp = p[1:] - p[:-1]  #  this must be ground truth
        #print("dp : ", dp)
        distances[1:] = dp.norm(dim=1).cumsum(0).numpy()
        dp = dp[:,:2]
        distances_xy[1:] = dp.norm(dim=1).cumsum(0).numpy()
        print("distance : ",distances[-1])
        print("distances diff : ", distances[-1]-distances_xy[-1])
        seq_lengths = np.array([100, 200, 300, 400, 500, 600, 700, 800])
        scale = 0.1
        seq_lengths = scale*seq_lengths
        k_max = int(Rot.shape[0] / step_size) - 1
        num = [0,0,0,0,0,0,0,0]
        list_rpe = [[], [], []]
        for k in range(0, k_max):
            idx_0 = k * step_size
            #print("idx_0 : ", idx_0)
            #print("distances[-1]", distances[-1])
            idx = 0 
            for seq_length in seq_lengths:
                if seq_length + distances[idx_0] > distances[-1]:
                    idx+=1
                    continue
                num[idx] += 1
                idx_shift = np.searchsorted(distances[idx_0:], distances[idx_0] + seq_length)
                idx_end = idx_0 + idx_shift
                list_rpe[0].append(idx_0)
                #print("list_rpe[0]", list_rpe[0])
                list_rpe[1].append(idx_end)
                #print("distances : ", distances[idx_0], distances[-1])
                idx+= 1
        idxs_0 = list_rpe[0]
        idxs_end = list_rpe[1]
        delta_p = Rot[idxs_0].transpose(-1, -2).matmul(
                ((p[idxs_end] - p[idxs_0]).float()).unsqueeze(-1)).squeeze()
        list_rpe[2] = delta_p
        print(len(list_rpe[0]))
        print(num)

def plot_bias() :
    data_path = "../dataset/sheco_data"
    exp_list = [1,2,3]
    for exp_num in exp_list : 
        ekf_name = "ekf_ext_rev_{}.csv".format(exp_num)
        gt_name = "imu{}.bag".format(exp_num)
        df_gt = pd.read_csv(os.path.join(data_path, gt_name))
        gt_ = np.array(df_gt.iloc[:,0]).reshape(-1,1)
        gt_ = gt_ -gt_[0]
        gt_acc = np.array(df_gt.iloc[:,29:32]).reshape(-1,3)
        gt_gyr = np.array(df_gt.iloc[:,17:20]).reshape(-1,3)
        
        df_ekf = pd.read_csv(os.path.join(data_path, ekf_name))
       
        b_acc = np.array(df_ekf.iloc[:,10:13]).reshape(-1,3)
        b_gyr = np.array(df_ekf.iloc[:,13:16]).reshape(-1,3)
        t_ = np.array(df_ekf.iloc[:,0]).reshape(-1,1)
        t_ = t_- t_[0]
        
        
        fig1, ax1 = plt.subplots(3,1,sharex=True,figsize=(20,10))
        ax1[0].plot(gt_, gt_acc[:,0])
        ax1[0].plot(t_, b_acc[:,0])        
        ax1[1].plot(gt_, gt_acc[:,1])
        ax1[1].plot(t_, b_acc[:,1])
        ax1[2].plot(gt_, gt_acc[:,2])
        ax1[2].plot(t_, b_acc[:,2])
        
        
        ax1[0].set(xlabel='time (s)',ylabel=r'$a_x (m/s^2)$',title="x acceleration")
        ax1[1].set(xlabel='time (s)',ylabel=r'$a_y (m/s^2)$',title="y acceleration")
        ax1[2].set(xlabel='time (s)',ylabel=r'$a_z (m/s^2)$',title="z acceleration")
        ax1[0].legend(['measurement', 'bias'])
        ax1[1].legend(['measurement', 'bias'])
        ax1[2].legend(['measurement', 'bias'])
        
        fig1.savefig(os.path.join("./plot", "exp{}_b_acc.png".format(exp_num)))
        
        fig2, ax2 = plt.subplots(3,1,sharex=True,figsize=(20,10))
        ax2[0].plot(gt_, gt_gyr[:,0])
        ax2[0].plot(t_, b_gyr[:,0])
        ax2[1].plot(gt_, gt_gyr[:,1])
        ax2[1].plot(t_, b_gyr[:,1])
        ax2[2].plot(gt_, gt_gyr[:,2])
        ax2[2].plot(t_, b_gyr[:,2])
       
        
        ax2[0].set(xlabel='time (s)',ylabel=r'$w_x (rad/s)$',title="x angular velocity")
        ax2[1].set(xlabel='time (s)',ylabel=r'$w_y (rad/s)$',title="y angular velocity")
        ax2[2].set(xlabel='time (s)',ylabel=r'$w_z (rad/s)$',title="z angular velocity")
        ax2[0].legend(['measurement', 'bias'])
        ax2[1].legend(['measurement', 'bias'])
        ax2[2].legend(['measurement', 'bias'])
        fig2.savefig(os.path.join("./plot", "exp{}_w_acc.png".format(exp_num)))
"""
- sample : 2, 0, 2
- output : 19, 13, 13
- ext : 19, 13, 13
- ext rev : 19, 13, 12 
"""
def calc_offset(off_dict, name) : 
    exp_list = [1,2,3]
    data_path = "../dataset/sheco_data"
    res = []
    for exp_num in exp_list : 
        ref_name = "sample{}.csv".format(exp_num)
        ekf_name = "{}{}.csv".format(name, exp_num)
        df_ref = pd.read_csv(os.path.join(data_path, ref_name))
        df_ekf = pd.read_csv(os.path.join(data_path, ekf_name))
        ref_time = np.array(df_ref.iloc[:,0]).reshape(-1,1)
        ekf_time = np.array(df_ekf.iloc[:,0]).reshape(-1,1)
        res.append(ekf_time[off_dict[name][exp_num-1]][0] - ref_time[off_dict["sample"][exp_num-1]][0])
    return res
if __name__=='__main__' :
    #plot_dt_sc()
    #args = KITTIArgs()
    #dataset = KITTIDataset(args)
    #plot_distance(args, dataset)
    #args = USVArgs()
    #dataset = USVDataset(args)
    #print(dataset.datasets)
    name = "ekf_output"
    off_dict_gps = {"sample" : [2,0,2], "ekf_output" : [13,10,13], "ekf_ext_" : [13,13,13], "ekf_ext_rev_" : [13,13,12]} 
    off_dict_imu = {"sample" : [0,1,0], "ekf_output" : [9,7,9], "ekf_ext_" : [9,7,9], "ekf_ext_rev_" : [9,7,9]}
    offset = calc_offset(off_dict_gps, name)
    print(offset)
    folder_dict = {"sample" : ".", "ekf_output" : "output", "ekf_ext_" : "ext", "ekf_ext_rev_" : "rev"}
    #[26201680711117849, 26201887724724675, 26202270859414101]
    plot_timestamp(folder_dict[name], name,offset)
    #plot_bias()