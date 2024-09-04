import os
import shutil
import numpy as np
import pandas as pd  
import glob
import time
import datetime
import pickle
import torch
import matplotlib.pyplot as plt
from termcolor import cprint
from navpy import lla2ned
from collections import OrderedDict, namedtuple
from dataset import BaseDataset
from utils_torch_filter import TORCHIEKF
from utils_numpy_filter import NUMPYIEKF as IEKF
from utils import prepare_data
from train_torch_filter import train_filter
from utils_plot import results_filter


def launch(args):
    if args.read_data:
        args.dataset_class.read_data(args)
    dataset = args.dataset_class(args)
    dataset.post_init_processing()

    if args.train_filter:
        train_filter(args, dataset)

    if args.test_filter:
        test_filter(args, dataset)

    if args.results_filter:
        results_filter(args, dataset)


class USVParameters(IEKF.Parameters):  # 클래스 이름 변경
    # gravity vector
    #g = np.array([0, 0, -9.80655])
    g = np.array([0, 0, -9.81])
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

    odometry_benchmark = OrderedDict()
    odometry_benchmark["merged_output1"] = [0, 20000]
    # odometry_benchmark["merged_output_2"] = [0, 25697]
    # odometry_benchmark["merged_output_3"] = [0, 25982]

    def __init__(self, args):
        # super(USVDataset, self).__init__(args)
        self.path_data_save = args.path_data_save
        self.path_results = args.path_results
        self.path_temp = args.path_temp
        self.datasets_test = args.test_sequences
        self.datasets_validation = args.cross_validation_sequences

        self.datasets = []
        self.datasets_train = []
        self.datasets_validatation_filter = OrderedDict()
        self.datasets_train_filter = OrderedDict()

        #self.sigma_gyro = 1.e-4
        #self.sigma_acc = 1.e-4
        #self.sigma_b_gyro = 1.e-5
        #self.sigma_b_acc = 1.e-4
        self.sigma_gyro = 0.005
        self.sigma_acc = 0.05
        self.sigma_b_gyro = 1.e-4
        self.sigma_b_acc = 1.e-4

        self.num_data = 0
        self.normalize_factors = None
        self.get_datasets()  

        # self.pickle_path = os.path.join(args.path_data_base, 'merged_output.p')
        self.datasets_train_filter["merged_output1"] = [0, 20000]
        # self.datasets_train_filter["merged_output_2"] = [0, 30000]
        # self.datasets_validatation_filter['merged_output1'] = [20001, 29000]

        #########################################
        # # Load the pickle file
        # with open('../0828/merged_output.p', 'rb') as f:
        #     loaded_df = pickle.load(f)
        # print(loaded_df.head())
        ############################################

    def post_init_processing(self):
        """ 데이터를 로드한 후, normalize factors를 설정 """
        self.set_normalize_factors()

    @staticmethod
    def load_oxts_packets_and_poses(data):
        """Converts data from merged_output DataFrame (or dict) to the required format."""
        # Check if data is a dict and convert it to DataFrame
        if isinstance(data, dict):
            data = pd.DataFrame(data)

        print("data : {}".format(data[:5]))

        poses = []
        packets = []
        
        # Define the packet structure with a valid identifier for time
        OxtsPacket = namedtuple('OxtsPacket', 'time, lat, lon, alt, roll, pitch, yaw, vn, ve, vu, ax, ay, az, wx, wy, wz')

        for _, row in data.iterrows():
            # Create the pose matrix (4x4 identity matrix)
            pose = np.eye(4)
            # Convert roll, pitch, yaw to rotation matrix
            Rot = USVDataset.rotz(row['yaw']).dot(USVDataset.roty(row['pitch'])).dot(USVDataset.rotx(row['roll']))
            pose[:3, :3] = Rot
            # Convert lat, lon, alt to NED coordinates
            pose[:3, 3] = lla2ned(row['lat'], row['lon'], row['alt'], row['lat'], row['lon'], row['alt'])
            
            # Create the OxtsPacket including the 'time' (formerly '%time')
            packet = OxtsPacket(row['%time'], row['lat'], row['lon'], row['alt'], row['roll'], row['pitch'], row['yaw'],
                                row['vn'], row['ve'], row['vu'], row['ax'], row['ay'], row['az'],
                                row['wx'], row['wy'], row['wz'])
            
            packets.append(packet)
            poses.append(pose)
        
        return list(zip(packets, poses))

    @staticmethod
    def read_data(args):
        """
        Read the data from the USV dataset pickle file and prepare it
        for training, similar to the KITTI dataset read_data method.

        :param args: Arguments object containing dataset paths and other configs
        """
        t_tot = 0
        # Load data from merged_output.p
        with open(args.path_data_base + '/merged_output1.p', 'rb') as f:
            data = pickle.load(f)

            oxts = USVDataset.load_oxts_packets_and_poses(data)

            # Initialize arrays for storing processed data
            num_samples = len(oxts)
            t = np.zeros(num_samples)
            acc = np.zeros((num_samples, 3))
            gyro = np.zeros((num_samples, 3))
            p_gt = np.zeros((num_samples, 3))
            v_gt = np.zeros((num_samples, 3))
            ang_gt = np.zeros((num_samples, 3))

            k_max = num_samples
            for k in range(k_max):
                oxts_k = oxts[k]
                # t[k] = k * 0.01  # Assuming a constant timestep for simplicity; replace with actual timestamp if available
                t[k] = oxts_k[0].time / 1e9
                p_gt[k, 0] = oxts_k[0].lat  # Position from the pose matrix
                p_gt[k, 1] = oxts_k[0].lon
                p_gt[k, 2] = oxts_k[0].alt
                v_gt[k, 0] = oxts_k[0].ve
                v_gt[k, 1] = oxts_k[0].vn
                v_gt[k, 2] = oxts_k[0].vu
                ang_gt[k, 0] = oxts_k[0].roll
                ang_gt[k, 1] = oxts_k[0].pitch
                ang_gt[k, 2] = oxts_k[0].yaw
                acc[k, :] = [oxts_k[0].ax, oxts_k[0].ay, oxts_k[0].az]
                gyro[k, :] = [oxts_k[0].wx, oxts_k[0].wy, oxts_k[0].wz]

            # Normalize timestamps
            t0 = t[0]
            t = t - t0

            u = np.concatenate((gyro, acc), -1)

            # Convert numpy arrays to PyTorch tensors
            t = torch.from_numpy(t).float()
            p_gt = torch.from_numpy(p_gt).float()
            v_gt = torch.from_numpy(v_gt).float()
            ang_gt = torch.from_numpy(ang_gt).float()
            u = torch.from_numpy(u).float()

        mondict = {
            't': t, 'p_gt': p_gt, 'ang_gt': ang_gt, 'v_gt': v_gt,
            'u': u, 'name': 'merged_output1', 't0': t0
        }
        USVDataset.dump(mondict, args.path_data_save, 'merged_output1')
        # Delete the original merged_output.p file
        if os.path.exists('../data/merged_output1.p'):
            os.remove('../data/merged_output1.p')
            print("Original merged_output1.p file deleted.")
        else:
            print("The file does not exist.")
        
        #########################################################
        with open('../data/merged_output1.p', 'rb') as f:
            data = pickle.load(f)

            # if isinstance(data, dict):
            #     print("Data type: dict")
            #     print("Data keys: {}".format(list(data.keys())))  # dict의 키들을 출력

            #     # 각 키에 해당하는 값의 앞부분을 출력 (예: 앞의 5개 요소)
            #     for key, value in data.items():
            #         print("\nKey: {}".format(key))
            #         if isinstance(value, list) or isinstance(value, tuple):
            #             print("First 5 elements of {}: {}".format(key, value[:5]))
            #         elif isinstance(value, dict):
            #             print("Keys of {}: {}".format(key, list(value.keys())[:5]))
            #         elif isinstance(value, (int, float, str)):
            #             print("Value of {}: {}".format(key, value))
            #         elif isinstance(value, torch.Tensor):
            #             print("Tensor shape: {}, First 5 elements: {}".format(value.shape, value[:5]))
            #         else:
            #             print("Type: {}, First 5 elements: {}".format(type(value), str(value)[:100])) 

            # else:
            #     print("Data is of type: {}".format(type(data)))
            ######################################################################################
                
            print("\n Total dataset duration : {:.2f} s".format(t[-1] - t[0]))

    def set_normalize_factors(self):
        super().set_normalize_factors()

    @staticmethod
    def rotx(t):
        """Rotation about the x-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    @staticmethod
    def roty(t):
        """Rotation about the y-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @staticmethod
    def rotz(t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # @staticmethod
    # def pose_from_oxts_packet(packet, scale):
    #     """Helper method to compute a SE(3) pose matrix from an OXTS packet."""
    #     er = 6378137.  # earth radius (approx.) in meters

    #     # Use a Mercator projection to get the translation vector
    #     tx = scale * packet.lon * np.pi * er / 180.
    #     ty = scale * er * np.log(np.tan((90. + packet.lat) * np.pi / 360.))
    #     tz = packet.alt
    #     t = np.array([tx, ty, tz])

    #     # Use the Euler angles to get the rotation matrix
    #     Rx = USVDataset.rotx(packet.roll)
    #     Ry = USVDataset.roty(packet.pitch)
    #     Rz = USVDataset.rotz(packet.yaw)
    #     R = Rz.dot(Ry.dot(Rx))

    #     # Combine the translation and rotation into a homogeneous transform
    #     return R, t

    @staticmethod
    def transform_from_rot_trans(R, t):
        """Transformation matrix from rotation matrix and translation vector."""
        R = R.reshape(3, 3)
        t = t.reshape(3, 1)
        return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

    # @staticmethod
    # def load_timestamps(data_path):
    #     """Load timestamps from file."""
    #     timestamp_file = os.path.join(data_path, 'oxts', 'timestamps.txt')

    #     timestamps = []
    #     with open(timestamp_file, 'r') as f:
    #         for line in f.readlines():
    #             t = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
    #             timestamps.append(t)
    #     return timestamps


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
    path_data_base = "/home/wisrl/pjw/ai-imu/dataset/sheco_data"
    path_data_save = "../data"
    path_results = "../results"
    path_temp = "../temp"
    epochs = 400
    seq_dim = 6000

    # training, cross-validation and test dataset
    name = ['merged_output1','merged_output2','merged_output3']
    cross_validation_sequences = ['merged_output3']
    test_sequences = ['merged_output3']
    continue_training = False

    # choose what to do
    read_data = 0
    train_filter = 1
    test_filter = 0
    results_filter = 0
    dataset_class = USVDataset
    parameter_class = USVParameters  # 클래스 이름에 맞게 수정


if __name__ == '__main__':
    args = USVArgs()  # USVArgs의 인스턴스 생성
    dataset = USVDataset(args)  # dataset 인스턴스 생성
    launch(args)  # 'args' 객체를 전달
