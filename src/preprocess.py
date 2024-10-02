import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import numpy as np
import torch
import os
import pdb
from collections import defaultdict
import json

def run_sample(datapath, exp_list) :
    #choose ref_step to balance number of num[0], num[1]
        
    #train_dict = {1:[[0, 6801],[6801, 9011]], 2: [[0, 2793], [2793, 9145]]}
    #,3: [[0, 9213]]}
    """
    split by gps 
    
    split exp1 
    0 ~ 6798 (except 6799)  / 6801 ~ 9010
    
    split exp2
    0 ~ 2786 (except 2787) / 2789 ~ 9144
    """
    
    imu_offset = {1: 0, 2: 2, 3:0}
    total = {}
    total_imu = {}
    total_gps = {}
    for exp_num in exp_list : 
        # Load GPS and IMU data
        ekf_df = pd.read_csv(os.path.join(datapath,'sample{}.csv'.format(exp_num)))
        imu_df = pd.read_csv(os.path.join(datapath,'imu{}.bag'.format(exp_num)))
        gps_df = pd.read_csv(os.path.join(datapath,'gt{}.bag'.format(exp_num)))
        print("exp: ",exp_num) 
        
        # 1. pair gps base, rover message
        # if many base or rover message found, use last one for pair
        
        b_mask = np.array((gps_df['field.header.frame_id']=='smc_2000'))
        r_mask = np.array((gps_df['field.header.frame_id']=='smc_plus'))
        
        i = 0
        j = 0
        pair_num = min(sum(b_mask), sum(r_mask))
        tmp_r_t = []
        tmp_b_t = []
        
        # check 
        candidate = []
        missing = []
        while i<len(gps_df) :
            if b_mask[i] ^ b_mask[i+1] : 
                if candidate : 
                    time_ref = gps_df.iloc[i+1,0]
                    #tmp = time_ref - np.array([c[1] for c in candidate])
                    tmp = np.array([c[1] for c in candidate])
                    #print(i, tmp)
                    #assert np.all([t>0 for t in tmp]), "time jump detected"
                    selected_idx= np.argmin(tmp)
                    selected = candidate[selected_idx][0]
                    for k in range(len(candidate)) :
                        if k != selected_idx and b_mask[i]: 
                            missing.append(candidate[k])  
                    candidate = []
                else : 
                    selected = i
                if b_mask[i] : 
                    tmp_b_t.append(selected)
                    tmp_r_t.append(i+1)
                else : 
                    tmp_b_t.append(i+1)
                    tmp_r_t.append(selected)
                i+=2
            else : 
                if not candidate : 
                    candidate.append([i,gps_df.iloc[i,0]])
                candidate.append([i+1, gps_df.iloc[i+1,0]])
                i+=1
        
        gps_time = np.array(gps_df.iloc[:,0])
        ref_gps_t = []        
        for i in range(len(tmp_b_t)) : 
            ref_gps_t.append(min(gps_time[tmp_b_t[i]], gps_time[tmp_r_t[i]]))
        print(np.all(ref_gps_t[:-1] <= ref_gps_t[1:]))
        
        #print(len(tmp_b_t), len(tmp_r_t))
        
        #2. from ekf sample check it was from gps or imu
        # Initialize a variable to keep track of the last used GPS index
        
        ekf_time = np.array(ekf_df.iloc[:,0])
        #ekf_time is sorted
        #print(np.all(ekf_time[:-1] <= ekf_time[1:]))
        
        
        # Assume that no imu, gps input is missing
        
        # if imu_offset is 0, start from imu -> ekf, else gps -> ekf
        all_imu_t = np.array(imu_df.iloc[:,0])
        imu_start_ekf_t = ekf_time[0 if imu_offset[exp_num]==0 else 1]
        imu_start_index = np.searchsorted(all_imu_t, imu_start_ekf_t) -1
        
        print(imu_start_index, imu_start_ekf_t-all_imu_t[imu_start_index])
        from_imu = {}
        dt = imu_start_ekf_t-all_imu_t[imu_start_index]
        if imu_offset[exp_num]==0 : 
            from_imu[0] = [imu_start_index-1, dt]
        else : 
            from_imu[1] = [imu_start_index-1, dt]

        for i in range(imu_start_index,len(all_imu_t)) : 
            if (i-imu_start_index) % 7 == 6 :
                tmp_ekf_idx = np.searchsorted(ekf_time, all_imu_t[i])
                if tmp_ekf_idx >= len(ekf_time) : 
                    break    
                dt = ekf_time[tmp_ekf_idx] - all_imu_t[i]
                from_imu[tmp_ekf_idx]=[i, dt]
        from_imu_= np.array(from_imu.values())

        #print(np.mean(from_imu_[:,2]), np.std(from_imu_[:,2]))
        
        # which gps row is the first gps?
        # when offset is not 0, start from gps
        from_gps = {}
        #gps_count = []
        if imu_offset[exp_num] > 0 :
            from_gps[0] = [-1,-1, 0]
        for i in range(len(tmp_b_t)) : 
            ref_gps_t = min(gps_time[tmp_b_t[i]], gps_time[tmp_r_t[i]])
            rec_ekf_idx = np.searchsorted(ekf_time,ref_gps_t)
            if rec_ekf_idx == len(ekf_time) : 
                break
            else : 
                if rec_ekf_idx == 0 :
                    assert False
                    
                if from_imu.get(rec_ekf_idx) : 
                    rec_ekf_idx += 1
                dt = ekf_time[rec_ekf_idx]-ref_gps_t
                from_gps[rec_ekf_idx] = [tmp_b_t[i],tmp_r_t[i], dt]
                #gps_count.append(tmp_b_t[i])
                #gps_count.append(tmp_r_t[i])
        from_gps_ = np.array(list(from_gps.values()))
        print("gps statistics: ",np.mean(from_gps_[:,2]),np.std(from_gps_[:,2]))
        
        print("missing: ",missing)
        # left ones are also from gps
        
        #for i in missing : 
        #    from_gps[i[0]] = [-1,0]
        
        from_imu_idx = np.array(list(from_imu.keys()))
        from_gps_idx = np.array(list(from_gps.keys()))
        merged= np.unique(np.hstack((from_imu_idx, from_gps_idx)))
        #print(len(merged))
        print("total ekf :", len(ekf_df), "from_imu :", len(from_imu),"from_gps :", len(from_gps),"left :",len(ekf_df)-len(merged))
        
        left = []
        for i in range(len(ekf_df)) : 
            if i not in merged:
                left.append(i)
                
        intersect = np.intersect1d(from_imu_idx, from_gps_idx)
        # for intersected points, choose next ekf for gps ??
        #print(len(intersect), len(left))
        #print(len(np.unique(intersect)),len(np.unique(left)))
        #print(np.all(intersect[:-1] <= intersect[1:]))
        #print(np.all(left[:-1] <= left[1:]))
        
        """
        exp1
        1290 22580 22574 [134256, 2081283] [3400, 5672875]
        1314 22985 22979 [136664, 8747945] [3460, 823541]
        exp2
        308 9280 9247 [55160, 2277625] [1398, 3549863]
        426 12768 12748 [75880, 2207108] [1925, 1459091]
        558 16318 16306 [96957, 7832542] [2463, 3085177]
        642 19101 19081 [113484, 3949209] [2884, 1516978]
        843 24973 24975 [148379, 7734231] [3771, 8182595]
        """
        # intersect_ = intersect+1
        # outlier = []
        # for i in intersect_ : 
        #     if i not in left :
        #         outlier.append(i)
                
        # intersect = np.setdiff1d(intersect, outlier)
        
        outlier = []
        for i in intersect : 
            if i not in left : 
                outlier.append(i)
        
        # for i in intersect : 
        #     tmp = from_gps.pop(i)
        #     from_gps[i+1] = tmp
        
        # outlier
        #pdb.set_trace()
        if exp_num == 1 : 
        #     # 22574, 22979 from gps
            gps_outlier = [22574, 22979]
            for gps_i, i in enumerate(gps_outlier) : 
                from_gps[i] = [missing[gps_i][0],-1, 0]
        elif exp_num == 2 : 
        #     # 9247, 12748, 16306, 19081 from gps
            gps_outlier = [9247, 12748, 16306, 19081]
            for gps_i, i in enumerate(gps_outlier) : 
                from_gps[i] = [missing[gps_i][0],-1, 0]
        #     # 24973 from imu, 24974 from gps, 24975 from imu
            tmp = from_imu[24973]
            dt = ekf_time[24975]-all_imu_t[tmp[0]+7]
            from_imu[24975] = [tmp[0]+7, dt]
            del from_imu[24974]
            #print(from_gps[24974], from_imu[24973], from_imu[24974], from_imu[24975])
        
        from_imu_idx = np.array(list(from_imu.keys()))
        from_gps_idx = np.array(list(from_gps.keys()))
        merged= np.unique(np.hstack((from_imu_idx, from_gps_idx)))
        #print(len(merged))
        print("[After processing] total ekf :", len(ekf_df), "from_imu :", len(from_imu),"from_gps :", len(from_gps),"left :",len(ekf_df)-len(merged))
        
        #ref_step = 3.57*1e7
        threshold = 2*np.mean(from_gps_[:,2])
        
        #threshold = 1*1e6
        print("threshold : ",threshold)
        # for each ekf sample make
        # [ekf index for time, ekf index for gps, imu index for imu]
        num_fix = 0
        processed_idx = []
        for k,v in from_imu.items() : 
            curr_time = ekf_time[k]
            dt_prev = 2*threshold
            dt_next = 2*threshold 
            if k-1 >= 0 and from_gps.get(k-1) : 
                dt_prev = curr_time - ekf_time[k-1]
            
            if k+1 < len(ekf_df) and from_gps.get(k+1):
                dt_next = ekf_time[k+1] - curr_time
            #print(dt_prev, dt_next)
            if dt_prev < threshold : 
                processed_idx.append([k, k-1, v[0]])
                num_fix += 1
            elif dt_next < threshold : 
                processed_idx.append([k, k+1, v[0]])
                num_fix += 1
            else :  
                processed_idx.append([k, k, v[0]])
        processed_idx = np.array(processed_idx)
        print("fixed with gps :", num_fix)
        ekf_time = ekf_time[processed_idx[:,0]]
        processed_dt = ekf_time[1:] - ekf_time[:-1]
        print("processed_dt: ",np.mean(processed_dt), np.std(processed_dt), np.min(processed_dt), np.max(processed_dt),len(processed_dt))
        # fig, ax = plt.subplots()
        # ax.hist(processed_dt, bins=50, density=True, histtype='stepfilled',cumulative=False, color='blue')
        # start, end = ax.get_xlim()
        # ax.xaxis.set_ticks(np.arange(start, end, 1e6))
        # fig.savefig(os.path.join("./plot", "exp{}_dt_fixed.png".format(exp_num)))
        total[exp_num] = processed_idx
        total_gps[exp_num] = from_gps
        total_imu[exp_num] = from_imu
    return total, total_gps, total_imu

def merge_ekf_imu(exp_list, processed_dict, datapath) :      

       
    for exp_num in exp_list : 
        ekf_df = pd.read_csv(os.path.join(datapath,'sample{}.csv'.format(exp_num)))
        imu_df = pd.read_csv(os.path.join(datapath,'imu{}.bag'.format(exp_num))) 
        # Extract the relevant fields
        merged_data = []
        processed_idx = processed_dict[exp_num]
        for i in processed_idx :
            ekf_timestamp_ns = ekf_df.iloc[i[0],0]
            ekf_row = ekf_df.iloc[i[1]]
            imu_row = imu_df.iloc[i[2]]
            
            merged_row = {
                #'%time': datetime.fromtimestamp(ekf_timestamp_ns / 1e9).isoformat(),
                '%time': ekf_timestamp_ns,
                'lat': ekf_row['field.pos_x'],
                'lon': ekf_row['field.pos_y'],
                'alt': ekf_row['field.pos_z'],
                'roll': ekf_row['field.ori_r'],
                'pitch': ekf_row['field.ori_p'],
                'yaw': ekf_row['field.ori_y'],
                'vn': ekf_row['field.vel_x'],
                've': ekf_row['field.vel_y'],
                'vu': ekf_row['field.vel_z'],
                'ax': imu_row['field.linear_acceleration.x'],
                'ay': imu_row['field.linear_acceleration.y'],
                'az': imu_row['field.linear_acceleration.z'],
                'af': imu_row['field.linear_acceleration.x'],  # Duplicates as per instruction
                'al': imu_row['field.linear_acceleration.y'],  # Duplicates as per instruction
                'au': imu_row['field.linear_acceleration.z'],  # Duplicates as per instruction
                'wx': imu_row['field.angular_velocity.x'],
                'wy': imu_row['field.angular_velocity.y'],
                'wz': imu_row['field.angular_velocity.z'],
                'wf': imu_row['field.angular_velocity.x'],  # Duplicates as per instruction
                'wl': imu_row['field.angular_velocity.y'],  # Duplicates as per instruction
                'wu': imu_row['field.angular_velocity.z'],  # Duplicates as per instruction
            }
            #pdb.set_trace()
            merged_data.append(merged_row)
            
            
        print("merged data: ",len(merged_data))
        # Convert merged data to a DataFrame with the desired column order
        column_order = [
            '%time', 'lat', 'lon', 'alt', 'roll', 'pitch', 'yaw', 
            'vn', 've', 'vu', 'ax', 'ay', 'az', 'af', 'al', 'au', 
            'wx', 'wy', 'wz', 'wf', 'wl', 'wu'
        ]

        merged_df = pd.DataFrame(merged_data, columns=column_order)

        # Save the merged data to a new CSV file
        merged_df.to_csv(os.path.join(datapath,'merged_output{}.csv'.format(exp_num)), index=False)

        print("Merged CSV file created successfully!")

        # Save the DataFrame as a pickle file
        with open(os.path.join(datapath,'merged_output{}.p'.format(exp_num)), 'wb') as f:
            pickle.dump(merged_df, f)

        print("Merged CSV file created and saved as a pickle file successfully!")


        # Load the pickle file
        with open(os.path.join(datapath,'merged_output{}.p'.format(exp_num)), 'rb') as f:
            loaded_df = pickle.load(f)

        # Display the first few rows
        print(loaded_df.head())

def merge_final(exp_list, processed_dict, datapath) :      
       
    for exp_num in exp_list : 
        ekf_df = pd.read_csv(os.path.join(datapath,'ekf_final_{}.csv'.format(exp_num)))
        imu_df = pd.read_csv(os.path.join(datapath,'imu{}.bag'.format(exp_num))) 
        # Extract the relevant fields
        merged_data = []
        processed_idx = processed_dict[exp_num]
        for i in processed_idx :
            ekf_timestamp_ns = ekf_df.iloc[i[0],0]
            ekf_row = ekf_df.iloc[i[1]]
            imu_row = imu_df.iloc[i[2]]
            
            merged_row = {
                #'%time': datetime.fromtimestamp(ekf_timestamp_ns / 1e9).isoformat(),
                '%time': ekf_timestamp_ns,
                'lat': ekf_row['field.pos_x'],
                'lon': ekf_row['field.pos_y'],
                'alt': ekf_row['field.pos_z'],
                'roll': ekf_row['field.ori_r'],
                'pitch': ekf_row['field.ori_p'],
                'yaw': ekf_row['field.ori_y'],
                'vn': ekf_row['field.vel_x'],
                've': ekf_row['field.vel_y'],
                'vu': ekf_row['field.vel_z'],
                'ax': imu_row['field.linear_acceleration.x'],
                'ay': imu_row['field.linear_acceleration.y'],
                'az': imu_row['field.linear_acceleration.z'],
                'af': imu_row['field.linear_acceleration.x'],  # Duplicates as per instruction
                'al': imu_row['field.linear_acceleration.y'],  # Duplicates as per instruction
                'au': imu_row['field.linear_acceleration.z'],  # Duplicates as per instruction
                'wx': imu_row['field.angular_velocity.x'],
                'wy': imu_row['field.angular_velocity.y'],
                'wz': imu_row['field.angular_velocity.z'],
                'wf': imu_row['field.angular_velocity.x'],  # Duplicates as per instruction
                'wl': imu_row['field.angular_velocity.y'],  # Duplicates as per instruction
                'wu': imu_row['field.angular_velocity.z'],  # Duplicates as per instruction
                'bax':ekf_row['field.b_acc_x'],
                'bay':ekf_row['field.b_acc_y'],
                'baz':ekf_row['field.b_acc_z'],
                'bgx':ekf_row['field.b_gyr_x'],
                'bgy':ekf_row['field.b_gyr_y'],
                'bgz':ekf_row['field.b_gyr_z'],
                'cov0':ekf_row['field.cov0'],
                'cov1':ekf_row['field.cov1'],
                'cov2':ekf_row['field.cov2'],
                'cov3':ekf_row['field.cov3'],
                'cov4':ekf_row['field.cov4'],
                'cov5':ekf_row['field.cov5'],
                'cov6':ekf_row['field.cov6'],
                'cov7':ekf_row['field.cov7'],
                'cov8':ekf_row['field.cov8'],
            }
            #pdb.set_trace()
            merged_data.append(merged_row)
            
            
        print("merged data: ",len(merged_data))
        # Convert merged data to a DataFrame with the desired column order
        column_order = [
            '%time', 'lat', 'lon', 'alt', 'roll', 'pitch', 'yaw', 
            'vn', 've', 'vu', 'ax', 'ay', 'az', 'af', 'al', 'au', 
            'wx', 'wy', 'wz', 'wf', 'wl', 'wu','bax','bay','baz','bgx','bgy','bgz',
            'cov0', 'cov1', 'cov2', 'cov3', 'cov4', 'cov5', 'cov6', 'cov7', 'cov8'
        ]

        merged_df = pd.DataFrame(merged_data, columns=column_order)

        # Save the merged data to a new CSV file
        merged_df.to_csv(os.path.join(datapath,'merged_output_final{}.csv'.format(exp_num)), index=False)

        print("Merged CSV file created successfully!")

        # Save the DataFrame as a pickle file
        with open(os.path.join(datapath,'merged_output_final{}.p'.format(exp_num)), 'wb') as f:
            pickle.dump(merged_df, f)

        print("Merged CSV file created and saved as a pickle file successfully!")


        # Load the pickle file
        with open(os.path.join(datapath,'merged_output_final{}.p'.format(exp_num)), 'rb') as f:
            loaded_df = pickle.load(f)

        # Display the first few rows
        print(loaded_df.head())

def merge_ekf_py_imu(exp_list,processed_dict,datapath) :      
       
    for exp_num in exp_list : 
        ekf_df = pd.read_csv(os.path.join(datapath,'ekf_py_{}.csv'.format(exp_num)))
        imu_df = pd.read_csv(os.path.join(datapath,'imu{}.bag'.format(exp_num))) 
        # Extract the relevant fields
        merged_data = []
        processed_idx = processed_dict[exp_num]
        for i in processed_idx :
            ekf_timestamp_ns = ekf_df.iloc[i[0],0]
            ekf_row = ekf_df.iloc[i[1]]
            imu_row = imu_df.iloc[i[2]]
            merged_row = {
                #'%time': datetime.fromtimestamp(ekf_timestamp_ns / 1e9).isoformat(),
                '%time': ekf_timestamp_ns,
                'lat': ekf_row['field.pos_x'],
                'lon': ekf_row['field.pos_y'],
                'alt': ekf_row['field.pos_z'],
                'roll': ekf_row['field.ori_r'],
                'pitch': ekf_row['field.ori_p'],
                'yaw': ekf_row['field.ori_y'],
                'vn': ekf_row['field.vel_x'],
                've': ekf_row['field.vel_y'],
                'vu': ekf_row['field.vel_z'],
                'ax': imu_row['field.linear_acceleration.x'],
                'ay': imu_row['field.linear_acceleration.y'],
                'az': imu_row['field.linear_acceleration.z'],
                'af': imu_row['field.linear_acceleration.x'],  # Duplicates as per instruction
                'al': imu_row['field.linear_acceleration.y'],  # Duplicates as per instruction
                'au': imu_row['field.linear_acceleration.z'],  # Duplicates as per instruction
                'wx': imu_row['field.angular_velocity.x'],
                'wy': imu_row['field.angular_velocity.y'],
                'wz': imu_row['field.angular_velocity.z'],
                'wf': imu_row['field.angular_velocity.x'],  # Duplicates as per instruction
                'wl': imu_row['field.angular_velocity.y'],  # Duplicates as per instruction
                'wu': imu_row['field.angular_velocity.z'],  # Duplicates as per instruction
            }
            #pdb.set_trace()
            merged_data.append(merged_row)
            
            
        print("merged data: ",len(merged_data))
        # Convert merged data to a DataFrame with the desired column order
        column_order = [
            '%time', 'lat', 'lon', 'alt', 'roll', 'pitch', 'yaw', 
            'vn', 've', 'vu', 'ax', 'ay', 'az', 'af', 'al', 'au', 
            'wx', 'wy', 'wz', 'wf', 'wl', 'wu'
        ]

        merged_df = pd.DataFrame(merged_data, columns=column_order)

        # Save the merged data to a new CSV file
        merged_df.to_csv(os.path.join(datapath,'merged_output_py_final{}.csv'.format(exp_num)), index=False)

        print("Merged CSV file created successfully!")

        # Save the DataFrame as a pickle file
        with open(os.path.join(datapath,'merged_output_py_final{}.p'.format(exp_num)), 'wb') as f:
            pickle.dump(merged_df, f)

        print("Merged CSV file created and saved as a pickle file successfully!")


        # Load the pickle file
        with open(os.path.join(datapath,'merged_output_py_final{}.p'.format(exp_num)), 'rb') as f:
            loaded_df = pickle.load(f)

        # Display the first few rows
        print(loaded_df.head())


"""
preprocess rosbag generated ekf sample
1. added bias data
2. offset for stopped points?
3. first ekf from gps 
- output : 13, 13, 13
- ext : 13, 13, 13
- ext rev : 13, 13, 12
4. first ekf from imu (after 0.2s interval)
- output : 9, 7, 9
- ext : 9, 7, 9
- ext rev : 9, 7, 9 
"""
def run_ekf(datapath, exp_list, only_imu = False) : 
    start_seq = {1: [3462, 75407], 2: [13922, 281868], 3:[23490, 470123]}
    total = {}
    total_imu = {}
    total_gps = {}
    for exp_num in exp_list : 
        # Load GPS and IMU data
        ekf_df = pd.read_csv(os.path.join(datapath,'ekf_py_{}.csv'.format(exp_num)))
        imu_df = pd.read_csv(os.path.join(datapath,'imu{}.bag'.format(exp_num)))
        gps_df = pd.read_csv(os.path.join(datapath,'gt{}.bag'.format(exp_num)))
        print("exp: ",exp_num) 
        
        #print(len(tmp_b_t), len(tmp_r_t))
        
        #2. from ekf sample check it was from gps or imu
        # Initialize a variable to keep track of the last used GPS index
        
        ekf_time = np.array(ekf_df.iloc[:,0])
        ekf_time = ekf_time[:] - ekf_time[0]
        #ekf_time is sorted
        #print(np.all(ekf_time[:-1] <= ekf_time[1:]))
        
        
        # Assume that no imu, gps input is missing
        
        # if imu_offset is 0, start from imu -> ekf, else gps -> ekf
        from_imu = {}
        from_gps = {}
        imu_offset= start_seq[exp_num][1]
        gps_offset= start_seq[exp_num][0]
        for i in range(len(ekf_df)) : 
            tmp_seq = ekf_df.loc[i, 'field.seq']
            #print(tmp_seq)
            # find index from sequence
            if tmp_seq < 70000 : 
                from_gps[i] = tmp_seq - gps_offset
            else : 
                from_imu[i] = tmp_seq - imu_offset

        #ref_step = 3.57*1e7
        threshold = 1.0 * 1e7
        print("threshold : ",threshold)
        # for each ekf sample make
        # [ekf index for time, ekf index for gps, imu index for imu]
        num_fix = 0
        processed_idx = []
        if not only_imu : 
            for k,v in from_imu.items() : 
                curr_time = ekf_time[k]
                dt_prev = 2*threshold
                dt_next = 2*threshold 
                if k-1 >= 0 and from_gps.get(k-1) : 
                    dt_prev = curr_time - ekf_time[k-1]
                
                if k+1 < len(ekf_df) and from_gps.get(k+1):
                    dt_next = ekf_time[k+1] - curr_time
                #print(dt_prev, dt_next)
                if dt_prev < threshold : 
                    processed_idx.append([k, k-1, v])
                    num_fix += 1
                elif dt_next < threshold : 
                    processed_idx.append([k, k+1, v])
                    num_fix += 1
                else :  
                    processed_idx.append([k, k, v])
        else :
            for k,v in from_imu.items() : 
                processed_idx.append([k, k, v])

        processed_idx = np.array(processed_idx)
        print("fixed with gps :", num_fix)
        ekf_time = ekf_time[processed_idx[:,0]]
        print(processed_idx[:,0])
        #interval = 28 * 7
        interval = 100
        processed_dt = ekf_time[interval:] - ekf_time[:-interval]
        print("time per {} rows".format(interval))
        print("processed_dt: ",np.mean(processed_dt), np.std(processed_dt), np.min(processed_dt), np.max(processed_dt), len(processed_dt))

        #outlier = []
        #for i, dt_ in enumerate(processed_dt) : 
        #    if dt_ > 4.5*1e7 or dt_ < 2.5*1e7 : 
        #        outlier.append(i)
        #print(len(outlier))
        argmin = np.argmin(processed_dt)
        argmax = np.argmax(processed_dt)
        ekf_time_ = np.array(ekf_df.iloc[:,0])
        print(ekf_time[argmin+1]+ekf_time_[0],ekf_time[argmin]+ekf_time_[0],ekf_time[argmax+1]+ekf_time_[0],ekf_time[argmax]+ekf_time_[0])
        fig, ax = plt.subplots()
        ax.hist(processed_dt, bins=50, density=True, histtype='stepfilled',cumulative=False, color='blue')
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 1e6))
        fig.savefig(os.path.join("./plot", "exp{}_dt_final_fixed.png".format(exp_num)))
        total[exp_num] = processed_idx
        total_gps[exp_num] = from_gps
        total_imu[exp_num] = from_imu
    return total, total_gps, total_imu

    

if __name__=="__main__" : 
    data_path = "../dataset/sheco_data"
    exp_list = [1,2,3]

    total, total_gps, total_imu = run_ekf(data_path, exp_list, True)
    #merge_final(exp_list, total, data_path)
    #merge_gps_imu(exp_list,total, total_gps,data_path)
    merge_ekf_py_imu(exp_list, total, data_path)
    # for k,v in total_gps.items() : 
    #     v_ = {}
    #     for k_tmp,v_tmp in v.items() :
    #         v_[int(k_tmp)] = [int(i) for i in v_tmp]
    #     with open("from_gps{}.json".format(k),"w") as json_file : 
    #         json.dump(v_, json_file)
    
    