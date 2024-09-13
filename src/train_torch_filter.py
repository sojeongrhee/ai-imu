import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from termcolor import cprint
from utils_torch_filter import TORCHIEKF
from utils import prepare_data
import copy

max_loss = 2e1
max_grad_norm = 1e0
min_lr = 1e-5
criterion = torch.nn.MSELoss(reduction="sum")
lr_initprocesscov_net = 1e-4
weight_decay_initprocesscov_net = 0e-8
lr_mesnet = {'cov_net': 1e-4,
    'cov_lin': 1e-4,
    }
weight_decay_mesnet = {'cov_net': 1e-8,
    'cov_lin': 1e-8,
    }


def compute_delta_p(Rot, p):
    list_rpe = [[], [], []]  # [idx_0, idx_end, pose_delta_p]

    # sample at 1 Hz
    # Rot = Rot[::10]
    # p = p[::10]
    Rot = Rot[::4]
    p = p[::4]
    #print("Rot, p : ", Rot[0:5], p[0:5])

    # step_size = 10  # every second
    step_size = 7
    distances = np.zeros(p.shape[0])
    dp = p[1:] - p[:-1]  #  this must be ground truth
    print("dp : ", dp, "p shape: ",p.shape[0])
    distances[1:] = dp.norm(dim=1).cumsum(0).numpy()
    # print("distances : ", distances[1:])

    """
        ########################### Important Notice ############################## 
        seq_lengths가 초기처럼 [100, 200, 300, 400, 500, 600, 700, 800]으로 설정되면
        if seq_length + distances[idx_0] > distances[-1]: 조건이 항상 True여서
        for문을 단순히 iterate하므로 list_rpe가 저장되지 않는 문제가 있음
        Notice 1 : 따라서 seq_lengths를 적절히 조정해야 함
        Notice 2 : 1/10 scale로 하면 학습 loop는 돌아가나 loss가 제대로 나오지 않는 문제가 있음
    """
    # seq_lengths = [10, 20, 30, 40, 50, 60, 70, 80]
    seq_lengths = [0.1]
    # seq_lengths = [10, 11, 12, 13, 14, 15, 16, 17]
    k_max = int(Rot.shape[0] / step_size) - 1
    idx_diff = [np.inf]*len(seq_lengths)
    print("k_max : ", k_max)
    for k in range(0, k_max):
        idx_0 = k * step_size
        #print("idx_0 : ", idx_0)
        #print("distances[-1]", distances[-1])
        # import pdb; pdb.set_trace()
        j = 0
        for seq_length in seq_lengths:
            if seq_length + distances[idx_0] > distances[-1]:
                continue
            idx_shift = np.searchsorted(distances[idx_0:], distances[idx_0] + seq_length)
            idx_end = idx_0 + idx_shift
            list_rpe[0].append(idx_0)
            # print("list_rpe[0]", list_rpe[0])
            list_rpe[1].append(idx_end)
            # print("distances : ", distances[idx_0], distances[-1])
            idx_diff[j] = min(idx_shift, idx_diff[j])
            j+=1
    print(idx_diff)
    idxs_0 = list_rpe[0]
    idxs_end = list_rpe[1]
    delta_p = Rot[idxs_0].transpose(-1, -2).matmul(
        ((p[idxs_end] - p[idxs_0]).float()).unsqueeze(-1)).squeeze()
    list_rpe[2] = delta_p
    #print("list_rpe : ", list_rpe[0], list_rpe[1], list_rpe[2])
    return list_rpe


def train_filter(args, dataset):
    iekf = prepare_filter(args, dataset)
    prepare_loss_data(args, dataset)
    # import pdb; pdb.set_trace()
    save_iekf(args, iekf)
    optimizer = set_optimizer(iekf)
    start_time = time.time()
    print("start time : ",start_time)
    for epoch in range(1, args.epochs + 1):
        train_loop(args, dataset, epoch, iekf, optimizer, args.seq_dim)
        save_iekf(args, iekf)
        print("Amount of time spent for {} epoch: {}s\n".format(epoch, int(time.time() - start_time)))
        start_time = time.time()


def prepare_filter(args, dataset):
    iekf = TORCHIEKF()
    # set dataset parameter
    iekf.filter_parameters = args.parameter_class()
    iekf.set_param_attr()
    if type(iekf.g).__module__ == np.__name__:
        iekf.g = torch.from_numpy(iekf.g).double()

    # load model
    if args.continue_training:
        iekf.load(args, dataset)
    iekf.train()
    # init u_loc and u_std
    iekf.get_normalize_u(dataset)
    return iekf


def prepare_loss_data(args, dataset):

    file_delta_p = os.path.join(args.path_temp, 'delta_p.p')
    
    if os.path.isfile(file_delta_p):
        mondict = dataset.load(file_delta_p)
        dataset.list_rpe = mondict['list_rpe']
        dataset.list_rpe_validation = mondict['list_rpe_validation']
        # print("11111111")
        print("dataset_train_filter_keys : ", dataset.datasets_train_filter.keys())
        print("dataset_list_rpe_keys : ", dataset.list_rpe.keys())

        if set(dataset.datasets_train_filter.keys()) <= set(dataset.list_rpe.keys()): 
            return

    # prepare delta_p_gt
    list_rpe = {}
    print("dataset_train_filter_items : ", dataset.datasets_train_filter.items())

    for dataset_name, Ns in dataset.datasets_train_filter.items():
        t, ang_gt, p_gt, v_gt, u = prepare_data(args, dataset, dataset_name, 0)
        p_gt = p_gt.double()
        Rot_gt = torch.zeros(Ns[1], 3, 3)
        for k in range(Ns[1]):
            ang_k = ang_gt[k]
            Rot_gt[k] = TORCHIEKF.from_rpy(ang_k[0], ang_k[1], ang_k[2]).double()
        list_rpe[dataset_name] = compute_delta_p(Rot_gt[:Ns[1]], p_gt[:Ns[1]])

    list_rpe_validation = {}
    for dataset_name, Ns in dataset.datasets_validatation_filter.items():
        t, ang_gt, p_gt, v_gt, u = prepare_data(args, dataset, dataset_name, 0)
        p_gt = p_gt.double()
        Rot_gt = torch.zeros(Ns[1], 3, 3)
        for k in range(Ns[1]):
            ang_k = ang_gt[k]
            Rot_gt[k] = TORCHIEKF.from_rpy(ang_k[0], ang_k[1], ang_k[2]).double()
        list_rpe_validation[dataset_name] = compute_delta_p(Rot_gt[:Ns[1]], p_gt[:Ns[1]])
    
    list_rpe_ = copy.deepcopy(list_rpe)
    dataset.list_rpe = {}
    for dataset_name, rpe in list_rpe_.items():
        if len(rpe[0]) is not 0:
            dataset.list_rpe[dataset_name] = list_rpe[dataset_name]
        else:
            dataset.datasets_train_filter.pop(dataset_name)
            list_rpe.pop(dataset_name)
            cprint("%s has too much dirty data, it's removed from training list" % dataset_name, 'yellow')

    list_rpe_validation_ = copy.deepcopy(list_rpe_validation)
    dataset.list_rpe_validation = {}
    for dataset_name, rpe in list_rpe_validation_.items():
        if len(rpe[0]) is not 0:
            dataset.list_rpe_validation[dataset_name] = list_rpe_validation[dataset_name]
        else:
            dataset.datasets_validatation_filter.pop(dataset_name)
            list_rpe_validation.pop(dataset_name)
            cprint("%s has too much dirty data, it's removed from validation list" % dataset_name, 'yellow')
    mondict = {
        'list_rpe': list_rpe, 'list_rpe_validation': list_rpe_validation,
        }
    dataset.dump(mondict, file_delta_p)


def train_loop(args, dataset, epoch, iekf, optimizer, seq_dim):
    loss_train = 0
    optimizer.zero_grad()
    for i, (dataset_name, Ns) in enumerate(dataset.datasets_train_filter.items()):
        t, ang_gt, p_gt, v_gt, u, N0 = prepare_data_filter(dataset, dataset_name, Ns,
                                                                  iekf, seq_dim)
        #print("t, ang_gt, p_gt, v_gt, u :", t, ang_gt, p_gt, v_gt, u )
        loss = mini_batch_step(dataset, dataset_name, iekf,
                               dataset.list_rpe[dataset_name], t, ang_gt, p_gt, v_gt, u, N0)
        print("loss {} : {} ".format(i, loss))

        if loss is -1 or torch.isnan(loss):
            cprint("{} loss is invalid".format(i), 'yellow')
            continue
        elif loss > max_loss:
            cprint("{} loss is too high {:.5f}".format(i, loss), 'yellow')
            continue
        else:
            loss_train += loss
            cprint("{} loss: {:.5f}".format(i, loss))

    if loss_train == 0: 
        return 
    loss_train.backward()  # loss_train.cuda().backward()  
    g_norm = torch.nn.utils.clip_grad_norm_(iekf.parameters(), max_grad_norm)
    if np.isnan(g_norm) or g_norm > 3*max_grad_norm:
        cprint("gradient norm: {:.5f}".format(g_norm), 'yellow')
        optimizer.zero_grad()

    else:
        optimizer.step()
        optimizer.zero_grad()
        cprint("gradient norm: {:.5f}".format(g_norm))
    print('Train Epoch: {:2d} \tLoss: {:.5f}'.format(epoch, loss_train))
    return loss_train


def save_iekf(args, iekf):
    file_name = os.path.join(args.path_temp, "iekfnets.p")
    torch.save(iekf.state_dict(), file_name)
    print("The IEKF nets are saved in the file " + file_name)


def mini_batch_step(dataset, dataset_name, iekf, list_rpe, t, ang_gt, p_gt, v_gt, u, N0):
    iekf.set_Q()
    measurements_covs = iekf.forward_nets(u)
    #print("measurements cov : ",measurements_covs)
    print(len(t), len(u), len(v_gt),len(p_gt),t.shape[0],ang_gt[0])
    Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf.run(t, u, measurements_covs,
                                                            v_gt, p_gt, t.shape[0],
                                                            ang_gt[0])
    print(len(Rot), len(v), len(p), len(list_rpe[0]),len(list_rpe[1]),len(list_rpe[2]))
    delta_p, delta_p_gt = precompute_lost(Rot, p, list_rpe, N0)
    print("delta_p, delta_p_gt : ", delta_p[:5], delta_p_gt[:5])
    if delta_p is None:
        return -1
    loss = criterion(delta_p, delta_p_gt)
    return loss


def set_optimizer(iekf):
    param_list = [{'params': iekf.initprocesscov_net.parameters(),
                           'lr': lr_initprocesscov_net,
                           'weight_decay': weight_decay_initprocesscov_net}]
    for key, value in lr_mesnet.items():
        param_list.append({'params': getattr(iekf.mes_net, key).parameters(),
                           'lr': value,
                           'weight_decay': weight_decay_mesnet[key]
                           })
    optimizer = torch.optim.Adam(param_list)
    return optimizer


def prepare_data_filter(dataset, dataset_name, Ns, iekf, seq_dim):
    # get data with trainable instant
    t, ang_gt, p_gt, v_gt,  u = dataset.get_data(dataset_name)
    t = t[Ns[0]: Ns[1]]
    ang_gt = ang_gt[Ns[0]: Ns[1]]
    p_gt = p_gt[Ns[0]: Ns[1]] - p_gt[Ns[0]]
    v_gt = v_gt[Ns[0]: Ns[1]]
    u = u[Ns[0]: Ns[1]]

    # subsample data
    N0, N = get_start_and_end(seq_dim, u)
    t = t[N0: N].double()
    ang_gt = ang_gt[N0: N].double()
    p_gt = (p_gt[N0: N] - p_gt[N0]).double()
    v_gt = v_gt[N0: N].double()
    u = u[N0: N].double()

    # add noise
    if iekf.mes_net.training:
        u = dataset.add_noise(u)

    return t, ang_gt, p_gt, v_gt, u, N0


def get_start_and_end(seq_dim, u):
    if seq_dim is None:
        N0 = 0
        N = u.shape[0]
    else: # training sequence
        #N0 = 10 * int(np.random.randint(0, (u.shape[0] - seq_dim)/10))
        N0 = 7 * int(np.random.randint(0, (u.shape[0] - seq_dim)/7))
        N = N0 + seq_dim
    return N0, N


def precompute_lost(Rot, p, list_rpe, N0):
    N = p.shape[0]
    # Rot_10_Hz = Rot[::10]
    # p_10_Hz = p[::10]
    print("N : ", N ,"N0 : ",N0)
    #idxs_0 = torch.Tensor(list_rpe[0]).clone().long() - int(N0 / 10)
    #idxs_end = torch.Tensor(list_rpe[1]).clone().long() - int(N0 / 10)
    Rot_7_Hz = Rot[::4]
    p_7_Hz = p[::4]
    idxs_0 = torch.Tensor(list_rpe[0]).clone().long() - int(N0 / 7)
    idxs_end = torch.Tensor(list_rpe[1]).clone().long() - int(N0 / 7)
    delta_p_gt = list_rpe[2]
    idxs = torch.ones(idxs_0.shape[0], dtype=torch.bool)
    #idxs[:] = True
    print(torch.sum(idxs_0 < 0))
    print(torch.sum(idxs_end >= int(N/7)))
    idxs[idxs_0 < 0] = False
    idxs[idxs_end >= int(N / 7)] = False
    print("clip length: ", torch.sum(idxs), len(idxs))
    delta_p_gt = delta_p_gt[idxs]
    idxs_end_bis = idxs_end[idxs]
    idxs_0_bis = idxs_0[idxs]
    print("precompute lost", len(delta_p_gt), len(idxs_end_bis), len(idxs_0_bis))
    if len(idxs_0_bis) is 0: 
        return None, None     
    else:
        delta_p = Rot_7_Hz[idxs_0_bis].transpose(-1, -2).matmul(
        (p_7_Hz[idxs_end_bis] - p_7_Hz[idxs_0_bis]).unsqueeze(-1)).squeeze()
        distance = delta_p_gt.norm(dim=1).unsqueeze(-1)
        return delta_p.double() / distance.double(), delta_p_gt.double() / distance.double() 
