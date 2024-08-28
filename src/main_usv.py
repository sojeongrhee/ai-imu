import os
import shutil
import numpy as np
import pandas as pd  # Pandas 추가
from collections import namedtuple
import glob
import time
import datetime
import pickle
import torch
import matplotlib.pyplot as plt
from termcolor import cprint
from navpy import lla2ned
from collections import OrderedDict
from dataset import BaseDataset
from utils_torch_filter import TORCHIEKF
from utils_numpy_filter import NUMPYIEKF as IEKF
from utils import prepare_data
from train_torch_filter import train_filter
from utils_plot import results_filter


def launch(args):
    if args.read_data:
        args.dataset_class.read_data(args)  # 'args' 인자를 전달하도록 수정
    dataset = args.dataset_class(args)

    if args.train_filter:
        train_filter(args, dataset)

    if args.test_filter:
        test_filter(args, dataset)

    if args.results_filter:
        results_filter(args, dataset)


class USVParameters(IEKF.Parameters):  # 클래스 이름 변경
    # gravity vector
    g = np.array([0, 0, -9.80655])

    cov_omega = 2e-4
    cov_acc = 1e-3
    cov_b_omega = 1e-8
    cov_b_acc = 1e-6
    cov_Rot_c_i = 1e-8
    cov_t_c_i = 1e-8
    cov_Rot0 = 1e-6
    cov_v0 = 1e-1
    cov_b_omega0 = 1e-8
    cov_b_acc0 = 1e-3
    cov_Rot_c_i0 = 1e-5
    cov_t_c_i0 = 1e-2
    cov_lat = 1
    cov_up = 10

    def __init__(self, **kwargs):
        super(USVParameters, self).__init__(**kwargs)  # 클래스 이름에 맞게 수정
        self.set_param_attr()
        
    def set_param_attr(self):
        attr_list = [a for a in dir(USVParameters) if  # 클래스 이름에 맞게 수정
                     not a.startswith('__') and not callable(getattr(USVParameters, a))]  # 클래스 이름에 맞게 수정
        for attr in attr_list:
            setattr(self, attr, getattr(USVParameters, attr))  # 클래스 이름에 맞게 수정


class USVDataset(BaseDataset):  # 데이터셋 클래스 유지
    def __init__(self, args):
        super(USVDataset, self).__init__(args)
        self.pickle_path = os.path.join(args.path_data_base, 'merged_output.p')
        self.datasets_train_filter["merged_output"] = [0, 15000]

    @staticmethod
    def read_data(args):
        # Load data from merged_output.p
        with open(args.path_data_base + '/merged_output.p', 'rb') as f:
            data = pickle.load(f)

        # Store the data in the dataset object
        args.dataset_instance = USVDataset(args)
        args.dataset_instance.timestamps = pd.to_datetime(data['%time'])
        args.dataset_instance.lat_oxts = data['lat'].to_numpy()
        args.dataset_instance.lon_oxts = data['lon'].to_numpy()
        args.dataset_instance.alt_oxts = data['alt'].to_numpy()
        args.dataset_instance.roll_oxts = data['roll'].to_numpy()
        args.dataset_instance.pitch_oxts = data['pitch'].to_numpy()
        args.dataset_instance.yaw_oxts = data['yaw'].to_numpy()
        args.dataset_instance.vx_oxts = data['vn'].to_numpy()  # North velocity
        args.dataset_instance.vy_oxts = data['ve'].to_numpy()  # East velocity
        args.dataset_instance.vz_oxts = data['vu'].to_numpy()  # Up velocity
        args.dataset_instance.ax_oxts = data['ax'].to_numpy()  # X acceleration
        args.dataset_instance.ay_oxts = data['ay'].to_numpy()  # Y acceleration
        args.dataset_instance.az_oxts = data['az'].to_numpy()  # Z acceleration
        args.dataset_instance.wx_oxts = data['wx'].to_numpy()  # X angular velocity
        args.dataset_instance.wy_oxts = data['wy'].to_numpy()  # Y angular velocity
        args.dataset_instance.wz_oxts = data['wz'].to_numpy()  # Z angular velocity
        # # Save datasets to the dataset_instance
        # args.dataset_instance.datasets_train_filter = OrderedDict()
        for i, time_stamp in enumerate(args.dataset_instance.timestamps):
            args.dataset_instance.datasets_train_filter[time_stamp] = i


def test_filter(args, dataset):
    iekf = IEKF()
    torch_iekf = TORCHIEKF()

    # USVParameters를 사용하여 필터 매개변수 설정
    iekf.filter_parameters = USVParameters()  # USVParameters로 수정
    iekf.set_param_attr()
    torch_iekf.filter_parameters = USVParameters()  # USVParameters로 수정
    torch_iekf.set_param_attr()

    # 토치 필터 로드
    torch_iekf.load(args, dataset)
    iekf.set_learned_covariance(torch_iekf)

    for i in range(0, len(dataset.datasets)):
        dataset_name = dataset.dataset_name(i)
        if dataset_name not in dataset.odometry_benchmark.keys():
            continue

        print("Test filter on sequence: " + dataset_name)

        # 데이터 준비 (USV 데이터에 맞게 조정)
        t, ang_gt, p_gt, v_gt, u = prepare_data(args, dataset, dataset_name, i, to_numpy=True)

        N = None
        u_t = torch.from_numpy(u).double()
        measurements_covs = torch_iekf.forward_nets(u_t)
        measurements_covs = measurements_covs.detach().numpy()

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


class USVArgs:  # 클래스 이름 및 경로 수정
    path_data_base = "/home/sjrhee/sjrhee/ai-imu_0828/ai-imu-dr/data"
    path_data_save = "../data"
    path_results = "../results"
    path_temp = "../temp"

    epochs = 400
    seq_dim = 6000

    # training, cross-validation and test dataset
    cross_validation_sequences = ['IMU_converted_processed']
    test_sequences = ['IMU_converted_processed']
    continue_training = True

    # choose what to do
    read_data = 1
    train_filter = 1
    test_filter = 0
    results_filter = 0
    dataset_class = USVDataset
    parameter_class = USVParameters  # 클래스 이름에 맞게 수정


if __name__ == '__main__':
    args = USVArgs()  # USVArgs의 인스턴스 생성
    dataset = USVDataset(args)  # dataset 인스턴스 생성
    launch(args)  # 'args' 객체를 전달
