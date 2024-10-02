import numpy as np
from pyquaternion import Quaternion
from utils_numpy_filter import NUMPYIEKF as IEKF
from collections import OrderedDict
from scipy.linalg import cho_factor, cho_solve


class igps_state_t : 
    def __init__(self, pos, ori, addr) : 
        # vector3d 
        self.position = pos
        # quaternion
        self.orientation  = ori
        # string 
        self.address = addr 

class ekf_state : 
    def __init__(self, pos, vel, ori, b_acc, b_omega, init_cov) : 
        self.position = pos # vector3d
        self.velocity = vel # vector3d
        self.orientation = ori # quaternion
        self.bias_acc = b_acc # vector3d
        self.bias_gyr = b_omega # vector3d
        self.covariance = init_cov # P

class INROLParameters(IEKF.Parameters) : 
    
    g = np.array([0, 0, -9.81])
    cov_omega = 0.005 #gyr_noise_stddev = 0.005
    cov_acc = 0.05 #acc_noise_stddev = 0.05 
    cov_b_omega = 1e-4 #gyr_bias_drift = 1e-4
    cov_b_omega_decay = 100 #gyr_bias_exponential_decay = 100
    cov_b_acc = 1e-4 #acc_bias_drift = 1e-4
    cov_b_acc_decay = 1e-6 #acc_bias_exponential_decay = 1e-6
    cov_vel_dis = 1e-4 #velocity_discretization_noise_stddev = 1e-4
    
    
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

    #1 sec
    n_normalize_rot = 28
    #10 sec
    n_normalize_rot_c_i = 280

    P_dim = 15
    Q_dim = 15
    # sheco
    Pr_base = np.array([0.459, 0.449, 0.140])
    Pr_rover = np.array([0.459, -0.449, 0.140]) 
    def __init__(self, **kwargs):
        super(INROLParameters, self).__init__(**kwargs)  
        self.set_param_attr()

    def set_param_attr(self):
        attr_list = [a for a in dir(INROLParameters) if  
                     not a.startswith('__') and not callable(getattr(INROLParameters, a))] 
        for attr in attr_list:
            setattr(self, attr, getattr(INROLParameters, attr)) 
    
    
class INROLEKF : 
    def __init__(self, parameter_class=None) : 
        # variables to initialize with `filter_parameters`
        self.g = None
        self.cov_omega = None
        self.cov_acc = None
        self.cov_b_omega = None
        self.cov_b_acc = None
        self.cov_Rot_c_i = None
        self.cov_t_c_i = None
        self.cov_lat = None
        self.cov_up = None
        self.cov_b_omega0 = None
        self.cov_b_acc0 = None
        self.cov_Rot0 = None
        self.cov_v0 = None
        self.cov_Rot_c_i0 = None
        self.cov_t_c_i0 = None
        self.Q = None
        self.Q_dim = None
        self.n_normalize_rot = None
        self.n_normalize_rot_c_i = None
        self.P_dim = None
        
        # only sheco
        self.Pr_base = None
        self.Pr_rover = None
        self.cov_b_omega_decay = None
        self.cov_b_acc_decay = None
        self.cov_vel_dis = None
        self.verbose = None
        
        if parameter_class is None : 
            filter_parameters = INROLParameters()
        else:
            raise NotImplementedError
            
        self.filter_parameters = filter_parameters
        self.set_param_attr()
        
        self._state = None 
        #self._last_update =None
        self._measurement_queue = {}
        
    def set_param_attr(self) : 
        attr_list = [a for a in dir(self.filter_parameters) if not a.startswith('__')
                     and not callable(getattr(self.filter_parameters, a))]
        for attr in attr_list:
            setattr(self, attr, getattr(self.filter_parameters, attr))
        
        self.Q = np.zeros((self.Q_dim, self.Q_dim))
        self.Q[:3,:3] = (self.cov_vel_dis**2)*np.eye(3)
        self.Q[3:6,3:6] = (self.cov_acc**2)*np.eye(3)
        self.Q[6:9,6:9] =(self.cov_omega**2)*np.eye(3)
        self.Q[9:12,9:12] = (self.cov_b_acc**2)*np.eye(3)
        self.Q[12:15,12:15] = (self.cov_b_omega**2)*np.eye(3)
        
    def init_covariance(self) : 
        P = np.zeros((self.P_dim, self.P_dim))
        P[:3,:3] = 0.01*np.eye(3)    #self.cov_pos0
        P[3:6,3:6] = 0.1*np.eye(3)   #self.cov_vel0
        P[6:9,6:9] = 1.0*np.eye(3)   #self.cov_ori0
        P[9:12,9:12] = 1.0*np.eye(3) #self.cov_b_acc0
        P[12:15,12:15] = 1e-2*np.eye(3) #self.cov_b_gyr0
        return P        
        
    def update_measurement_queue(self, p, q, t, address):
        self._measurement_queue[t] = igps_state_t(
            pos = p,
            ori = q,
            addr = address
        )

        while len(self._measurement_queue) > 2 : 
            self._measurement_queue.popitem(last=False)
        
    def init_ekf_state(self, p, q, t, address) : 
        self.update_measurement_queue(p, q, t, address)
        #print(self._measurement_queue)
        #if len(self._measurement_queue) < 2 : 
        #    return None
    
        #assert self._state is None, "ekf state already initialized"
        # update_measurement_queue 

        initial_covariance = self.init_covariance()
        R = q.rotation_matrix
        if address == "smc_2000" : 
            pos = p - R.dot(self.Pr_base)
        elif address == "smc_plus" : 
            pos = p - R.dot(self.Pr_rover)
        else : 
            raise NotImplementedError
        
        return ekf_state(pos = pos, vel = np.zeros(3), ori = q, \
                b_acc = np.zeros(3), b_omega = np.zeros(3), init_cov = initial_covariance)
    
    # update on IMU input
    def updateIMU(self, a, w, dt) : 
        # self._state should be initialized
        if self._state is None : 
            return 
        
        assert type(self._state) == ekf_state, "ekf state uninitialized"
        
        a_hat = a - self._state.bias_acc
        w_hat = w - self._state.bias_gyr
        print("bias acc :",self._state.bias_acc,"bias gyr : ",self._state.bias_gyr)
        #print("a hat:", a_hat," w hat  : ",w_hat)
        # -self.g 
        R = self._state.orientation.rotation_matrix
        #print("rotation matrix: ",R)
        acceleration = R.dot(a_hat) - self.g
        #print("acceleration:", acceleration,",dt : ",dt)
        self._state.position += self._state.velocity * dt + acceleration* dt * dt / 2
        self._state.velocity += acceleration * dt 
        self._state.orientation = Quaternion(matrix=R.dot(IEKF.so3exp(w_hat * dt)))
        
        eta_a = self.cov_b_acc_decay
        eta_w = self.cov_b_omega_decay
        self._state.bias_acc *= self.compute_decay(eta_a*dt)
        self._state.bias_gyr *= self.compute_decay(eta_w*dt)
        
        Fx = self.make_propagation_error_jacobian(a_hat, w_hat, dt)
        Fi = self.make_propagation_noise_jacobian(dt)
        Q = self.make_propagation_noise_covariance()
        #print("Fx : ", Fx)
        self._state.covariance = Fx.dot(self._state.covariance).dot(Fx.T) + Fi.dot(Q).dot(Fi.T)
        #print("IMU cov :",self._state.covariance)
    # update on GPS input
    # y_p : vector3d, q : quaternion, V : matrix3d, 
    def updateGPS(self, y_p, q, V, address, t) : 
        if self._state is None : 
            # initialize self._state
            self._state = self.init_ekf_state(y_p, q, t, address)
            print("ekf state initialized")

        if self._state is None : 
            return 

        assert type(self._state) == ekf_state, "ekf state uninitialized"
        
        #self._last_update = t
        # qw = self._state.orientation.w
        # qx = self._state.orientation.x
        # qy = self._state.orientation.y
        # qz = self._state.orientation.z
        # theta = np.arcsin(2*qx*qy + 2*qz*qw)
        #print("y_p : ",y_p)
        #print("address : ",address)
        
        H = self.make_measurement_jacobian(address)
        R = self._state.orientation.rotation_matrix
        if address == "smc_2000" : 
            hx= self._state.position + R.dot(self.Pr_base)
        elif address == "smc_plus" : 
            hx= self._state.position + R.dot(self.Pr_rover)
        else : 
            raise NotImplementedError
        #print("hx : ", hx)
        delta_p = y_p - hx
        #print("residual: ", delta_p)
        #print("covariance : ",self._state.covariance)
        #print("H : ", H)
        S = H.dot(self._state.covariance).dot(H.T) + V
        #print("S : ", S)
        r = delta_p 
        # inverse of S
        #S_inv = np.linalg.solve(S,np.eye(3))
        S_chol = cho_factor(S)
        S_inv = cho_solve(S_chol, np.eye(3))
        K = (self._state.covariance.dot(H.T)).dot(S_inv)
        #print("K: ",K)
        #print(K.dot(r))
        self.update_innovation(K.dot(r))
        #print(K.dot(S).dot(K.T))
        self._state.covariance -= K.dot(S).dot(K.T)
        
        eigenvalues, eigenvectors = np.linalg.eigh(self._state.covariance)
        eig_safe = np.maximum(eigenvalues,1e-9)
        #print("eigen_value :", eig_safe)
        #print("eigen vectors :", eigenvectors)
        self._state.covariance = eigenvectors.dot(np.diag(eig_safe)).dot(np.linalg.inv(eigenvectors))

        
    # addr : base / rover, th : quaternion 
    def make_measurement_jacobian(self, addr) : 
        H = np.zeros((3,15))
        H[0:3, 0:3] = np.eye(3)
        R = (self._state.orientation).rotation_matrix
        if addr == "smc_2000" : 
            H[0:3, 6:9] = - R.dot(IEKF.skew(self.Pr_base))
        elif addr == "smc_plus" : 
            H[0:3, 6:9] = - R.dot(IEKF.skew(self.Pr_rover))
        else : 
            raise NotImplementedError    

        return H
    
    def make_propagation_error_jacobian(self, a_hat, w_hat, dt) : 
        I = np.eye(3)
        R = self._state.orientation.rotation_matrix
        R_w = IEKF.so3exp(w_hat*dt)
        #print("dt ",dt)
        eta_a = self.cov_b_acc_decay
        eta_w = self.cov_b_omega_decay
        Fx = np.eye(15)
        Fx[0:3, 3:6] = I*dt
        Fx[0:3, 6:9] = -0.5*(R.dot(IEKF.skew(a_hat))) * dt * dt
        Fx[0:3, 9:12] = -0.5*R*dt*dt
        Fx[3:6, 6:9] = -(R.dot(IEKF.skew(a_hat)))*dt
        Fx[3:6, 9:12] = -R*dt
        Fx[6:9, 6:9] = R_w.T
        Fx[6:9, 12:15] = -I*dt
        Fx[9:12, 9:12] = self.compute_decay(eta_a*dt) * I
        Fx[12:15, 12:15] = self.compute_decay(eta_w*dt) * I
        #print("decay ", self.compute_decay(eta_a*dt),self.compute_decay(eta_w*dt))
        return Fx
        
    def make_propagation_noise_jacobian(self, dt) : 
        I = np.eye(3)
        #R = self._state.orientation.rotation_matrix
        Fi = np.zeros((15,15))
        Fi[0:3, 0:3] = I*dt
        Fi[0:3, 3:6] = I*dt
        Fi[3:6, 3:6] = I*dt
        Fi[6:9, 6:9] = I*dt
        Fi[9:12, 9:12] = I*np.sqrt(dt)
        Fi[12:15, 12:15] = I*np.sqrt(dt)
        return Fi
    
    def make_propagation_noise_covariance(self) : 
        I = np.eye(3)
        n_v = self.cov_vel_dis
        n_a = self.cov_acc
        n_w = self.cov_omega
        w_a = self.cov_b_acc
        w_w = self.cov_b_omega
        
        Q = np.zeros((15,15))
        Q[0:3, 0:3] = I * n_v * n_v
        Q[3:6, 3:6] = I * n_a * n_a
        Q[6:9, 6:9] = I * n_w * n_w
        Q[9:12, 9:12] = I * w_a * w_a
        Q[12:15, 12:15] = I * w_w * w_w
        return Q
    
    # delta_x is np.array 15x1
    def update_innovation(self, delta_x) : 
        delta_p = delta_x[0:3]
        delta_v = delta_x[3:6]
        delta_theta = delta_x[6:9]
        delta_b_a = delta_x[9:12]
        delta_b_w = delta_x[12:15]
        
        self._state.position += delta_p
        self._state.velocity += delta_v
        R = self._state.orientation.rotation_matrix
        self._state.orientation = Quaternion(matrix=R.dot(IEKF.so3exp(delta_theta)))
        #print("delta_b_a :",delta_b_a)
        self._state.bias_acc += delta_b_a
        self._state.bias_gyr += delta_b_w
    
    @staticmethod
    # input q is quaternion
    def log(q) : 
        return q.angle * q.axis
    
    @staticmethod
    def compute_decay(z) : 
        return np.exp(-min(10.0, max(0.0, z)))