import numpy as np

from utils_numpy_filter import NUMPYIEKF as IEKF
from utils_inrol_filter import INROLParameters
from utils_inrol_filter import INROLEKF as EKF
import matplotlib
import matplotlib.pyplot as plt
import os 
import pandas as pd
import csv
import numpy as np
import torch
#np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
from utils import prepare_data

from collections import deque, OrderedDict
from pyquaternion import Quaternion
"""
input : raw gps,imu bag(csv) file 
output : ekf csv file

0. compare with original ekf result 
1. ekf result from only imu
2. ekf result with imu and gpu, preprocessed
"""

class ekf_msg : 
    def __init__(self, time, seq, pos, vel, ori) : 
        self.time = int(time)
        self.seq = int(seq)
        self.pos_x = pos[0]
        self.pos_y = pos[1]
        self.pos_z = pos[2]
        self.vel_x = vel[0]
        self.vel_y = vel[1]
        self.vel_z = vel[2]
        self.ori_r = ori[0]
        self.ori_p = ori[1]
        self.ori_y = ori[2]

class EKFrunner :
    
    def __init__(self,imu_col_name, gps_col_name) : 
        self._igps_queue = {}
        self._imu_queue = {}
        #self.imu_stack = deque(maxlen = 8) # 28Hz
        #self.imu_stack = deque(maxlen = 2) # 200Hz
        self.imu_stack = deque(maxlen = 3) # 100Hz
        self.imu_dict = {"index" : 0}       
        for i, n in enumerate(imu_col_name) : 
            self.imu_dict[n] = i+1
        self.gps_dict = {"index" : 0}
        for i, n in enumerate(gps_col_name) : 
            self.gps_dict[n] = i+1
        
    def onIMUdata(self, ekf, msg) : 
        self.imu_stack.append(msg)
        stack_size = self.imu_stack.maxlen 
        if len(self.imu_stack) >= stack_size :
            data_prev = self.imu_stack[0]
            data_curr = self.imu_stack[stack_size-1]
            a = np.array([
                self.imu_stack[stack_size-1][self.imu_dict["field.linear_acceleration.x"]], # linear_acceleration.x
                self.imu_stack[stack_size-1][self.imu_dict["field.linear_acceleration.y"]], # linear_acceleration.y
                self.imu_stack[stack_size-1][self.imu_dict["field.linear_acceleration.z"]]  # linear_accelearation.z
            ])
            w = np.array([
                self.imu_stack[stack_size-1][self.imu_dict["field.angular_velocity.x"]], # angular_velocity.x
                self.imu_stack[stack_size-1][self.imu_dict["field.angular_velocity.y"]], # angular_velocity.y
                self.imu_stack[stack_size-1][self.imu_dict["field.angular_velocity.z"]]  # angular_velocity.z
            ])
            t_prev = data_prev[self.imu_dict["field.header.stamp"]] / 1e9
            t_curr = data_curr[self.imu_dict["field.header.stamp"]] / 1e9
            seq = data_curr[self.imu_dict["field.header.seq"]]
            dt = t_curr - t_prev
            
            ekf.updateIMU(a, w, dt)
            R = ekf._state.orientation.rotation_matrix
            ori = IEKF.to_rpy(R)
            # time same as imu input
            output = ekf_msg(
               time = data_curr[self.imu_dict["field.header.stamp"]],
               seq = seq,
               pos = ekf._state.position,
               vel = ekf._state.velocity,
               ori = ori
            )

            data_prev = self.imu_stack[stack_size-1]
            self.imu_stack.clear()
            self.imu_stack.append(data_prev)
            
            return output
        
        return None
        
    def onGPSdata(self,ekf, data) : 
        timestamp = data[self.gps_dict["field.header.stamp"]] / 1e9
        seq = data[self.gps_dict["field.header.seq"]]
        p_x= data[self.gps_dict["field.pose.pose.position.x"]]
        p_y= data[self.gps_dict["field.pose.pose.position.y"]]
        p_z= data[self.gps_dict["field.pose.pose.position.z"]]
        q_x = data[self.gps_dict["field.pose.pose.orientation.x"]]
        q_y = data[self.gps_dict["field.pose.pose.orientation.y"]]
        q_z = data[self.gps_dict["field.pose.pose.orientation.z"]]
        q_w = data[self.gps_dict["field.pose.pose.orientation.w"]]
        
        q = Quaternion([q_w, q_x, q_y, q_z])
        print("gps: ", p_x, ", ", p_y)
        # covariance in row major order
        Vp = np.array(data[self.gps_dict["field.pose.covariance0"]:]).reshape(6,6)
        V = Vp[0:3, 0:3]
        print("cov: ", V[0,0], V[1,1], V[2,2])
        addr = data[self.gps_dict["field.header.frame_id"]]
        y_p = np.array([p_x, p_y, p_z])
        
        ekf.updateGPS(y_p, q, V, addr, timestamp)
        if ekf._state : 
            print("velocity : ", ekf._state.velocity)
            print("bias_acc : ", ekf._state.bias_acc)
        
            R = ekf._state.orientation.rotation_matrix
            ori = IEKF.to_rpy(R)
            output = ekf_msg(
                time = data[self.gps_dict["field.header.stamp"]],
                seq = seq,
                pos = ekf._state.position,
                vel = ekf._state.velocity,
                ori = ori
            )
            # only output on gps base    
            if addr == "smc_2000" : 
                return output
        
        return None 
        
def main() : 
    data_path = "../dataset/sheco_data"
    
    for exp_num in range(1,4) : 
        print("exp num : ",exp_num)
        imu_name = "imu{}.bag".format(exp_num)
        gps_name = "gt{}.bag".format(exp_num)
        df_imu = pd.read_csv(os.path.join(data_path, imu_name))
        df_gps = pd.read_csv(os.path.join(data_path, gps_name))
  
        # check both imu, gps rows sorted with stamp
        df_imu = df_imu.sort_values('field.header.stamp')
        df_gps = df_gps.sort_values('field.header.stamp')
        
        # run filter until both imu, gps runs out 
        iter_imu = df_imu.itertuples()
        iter_gps = df_gps.itertuples()
        imu_end = False
        gps_end = False
        
        ekf = EKF()
        ekf.filter_parameters = INROLParameters()
        ekf.set_param_attr()

        runner = EKFrunner(df_imu.columns, df_gps.columns)
        # 41 47
        #print(len(df_imu.columns), len(df_gps.columns)) 
        df_ekf = pd.DataFrame(columns = ['%time','field.seq','field.pos_x','field.pos_y','field.pos_z',\
            'field.vel_x','field.vel_y','field.vel_z','field.ori_r','field.ori_p','field.ori_y'])
        imu_flag = True #get new imu row
        gps_flag = True #get new gps row
        while True:
            if not imu_end and imu_flag: 
                try : 
                    imu_row = next(iter_imu)
                except StopIteration : 
                    imu_end = True
            
            if not gps_end and gps_flag : 
                try : 
                    gps_row = next(iter_gps)
                except StopIteration : 
                    gps_end = True
            
            if gps_end and imu_end : 
                break

            imu_stamp = np.inf
            gps_stamp = np.inf
            if not imu_end : 
                imu_stamp = imu_row[runner.imu_dict['field.header.stamp']]
                
            if not gps_end : 
                gps_stamp = gps_row[runner.gps_dict['field.header.stamp']]
            # select oldest row between imu, gps
            #print("imu_stamp :",imu_stamp,", gps_stamp :",gps_stamp)
            if imu_stamp <= gps_stamp : 
                imu_flag = True
                gps_flag = False
                
                ekf_row = runner.onIMUdata(ekf, imu_row)
                if ekf_row : 
                    print("IMU : ",imu_row[runner.imu_dict['field.header.seq']])
            else :
                imu_flag = False
                gps_flag = True
                ekf_row = runner.onGPSdata(ekf, gps_row)
                if ekf_row : 
                    print("GPS : ",gps_row[runner.gps_dict['field.header.seq']])    
            
            # some output can be None
            if ekf_row : 
                df_ekf.loc[len(df_ekf.index)] = ekf_row.__dict__.values()

        # save
        df_ekf = df_ekf.astype({'%time':np.uint64,'field.seq': np.uint64})
        df_ekf.to_csv(os.path.join(data_path, 'ekf_py_{}.csv'.format(exp_num)), index=False)
            
            
if __name__ == "__main__" :
    main()