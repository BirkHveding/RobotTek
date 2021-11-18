import sympy as sp
import numpy as np

def Ry_sym(theta):
    ct = sp.cos(theta)
    st = sp.sin(theta)
    R = sp.Matrix([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0, ct]])
    return R

def Rx_sym(theta):
    ct = sp.cos(theta)
    st = sp.sin(theta)
    R = sp.Matrix([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]])
    return R 

def skew(v):
    return sp.Matrix([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
                    
def exp3(omega, theta):
    omega = skew(omega)
    R = sp.eye(3) + sp.sin(theta) * omega + (1 - sp.cos(theta)) * omega * omega
    return R

def exp6(twist, theta):
    omega = skew(twist[:3])
    v = sp.Matrix(twist[3:])
    T = sp.eye(4)
    T[:3,:3] = exp3(twist[:3], theta)
    T[:3,3] = (sp.eye(3) * theta + (1 - sp.cos(theta)) * omega +
              (theta-sp.sin(theta)) * omega * omega) * v
    return T

def Ad(T):
    AdT = sp.zeros(6)
    R = sp.Matrix(T[:3, :3])
    AdT[:3, :3] = R
    AdT[3:, 3:] = R
    AdT[3:, :3] = skew(T[:3, 3]) * R
    return AdT

def calc_v(omega_mat, q_mat):
    #omega_mat and q_mat of type matrix with q_i & omega_i as columns
    #Returns v_mat in same type/format
    assert len(omega_mat) == len(q_mat)
    
    n_joints = omega_mat.shape[1] 
    v_mat = sp.zeros(3, n_joints)      

    for i in range(n_joints):
        v_mat[:,i] = (-skew(omega_mat.col(i)) * q_mat.col(i))
    return v_mat

def Slist_maker(omega_mat, q_mat): #omega_mat and q_mat of type matrix with q_i & omega_i as columns
    #Returns v_mat in same type/format
    v_mat = calc_v(omega_mat, q_mat)    
    n_joints = omega_mat.shape[1]
    Slist = sp.zeros(6, n_joints)
    
    for i in range(n_joints):
        Slist[:3,i] = omega_mat[:,i]
        Slist[3:,i] = v_mat[:,i]
    return Slist

def Js_maker(Slist, theta_list):
    n_joints = Slist.shape[1]
    Js = sp.zeros(6, n_joints)

    for i in range(n_joints-1, -1, -1):
        if i==0: # legger til denne for å få Js[0] = S_sb[0], står i boka
            Js[:,i] = Slist[:,i]
                    
        else:
            T = exp6(Slist[:,i-1], theta_list[i-1])

            for j in range( i-2, -1, -1):
                T = exp6(Slist[:,j], theta_list[j]) * T

        Js[:,i] = Ad(T) * Slist[:,i]
        

    Js.simplify()
    return Js

def Jb_maker(Blist, theta_list):
    n_joints = Blist.shape[1] - 1
    Jb = sp.zeros(6, 6)
    print(n_joints)
    
    Jb[:,n_joints] = Blist[:,n_joints] # Jb[n] = B[n]
    
    for i in range(n_joints-1, -1, -1):
        T = sp.eye(4)#exp6(Blist[:,i], -theta_list[i])
        print("i",i)
        for j in range( i+1, n_joints+1):
            T = sp.exp6(Blist[:,j], -theta_list[j]) * T
            print("j",j)
        Jb[:,i] = Ad(T) * Blist[:,i]
        print("\n")
    
    return Jb

jointLimits = np.array([[-180, 180], [-190, 45], [-120, 156], [-180, 180], [-90, 90], [-180, 180]]) #Assuming joint 5 has limits [-90, 90]
def applyJointLim(jointLimits, thetas):
    ''' Check if Inverse Kinematics solution (thetas) is within jointlimits\n 
    PARAMETERS:
    Jointlimits, numpy 2D array with lower and upper limits in deg\n
    Jointangles, numpy 1D array in rad
    RETURNS: Boolean true or false, if false a print message with the offending link is printed '''
    jointLimits = np.deg2rad(jointLimits) 
    thetas %= 2*np.pi #Post processing: all thetas in [0,2*pi)

    for i, theta in enumerate(thetas):
        if (jointLimits[i][1] < theta) | (theta < jointLimits[i][0]):
            print("Joint number: {} is outside limits. Theta: {} | limits: {}".format(i+1, np.rad2deg(float(theta)), np.rad2deg(jointLimits[i])))
            return False
    return True


#____DH-functions____

def rotX(alfa_im1):
    Rx = sp.eye(4)
    Rx[1,1] =    sp.cos(alfa_im1)
    Rx[1,2] =   -sp.sin(alfa_im1)
    Rx[2,1] =    sp.sin(alfa_im1)
    Rx[2,2] =    sp.cos(alfa_im1)
    return Rx

def rotZ(alfa_im1):
    Rz = sp.eye(4)
    Rz[0,0] =    sp.cos(alfa_im1)
    Rz[0,1] =   -sp.sin(alfa_im1)
    Rz[1,0] =    sp.sin(alfa_im1)
    Rz[1,1] =    sp.cos(alfa_im1)
    return Rz

def transX(a_im1):
    trA = sp.eye(4)
    trA[0,3] =  a_im1
    return trA

def transZ(d_i):
    trA = sp.eye(4)
    trA[2,3] =  d_i
    return trA

def PsFromTsd(T_sd):
    #Finner Ps fra T_sd
    #T_sd gir konfigurasjonen vi vil ha end-effector framen, B, i.
    #B, og derav også M, er lik som i DH
    #s er plassert nederst på roboten med positiv z oppover, altså ikke som i DH. Bør kanskje endres til å være lik DH 
    P_d = np.array([0,0,80,1])
    P_s = T_sd@P_d
    return P_s