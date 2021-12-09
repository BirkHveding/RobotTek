import sympy as sp
import numpy as np
from sympy.physics.mechanics import dynamicsymbols, mechanics_printing
mechanics_printing()

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
    assert len(omega_mat) == len(q_mat)
    
    n_joints = omega_mat.shape[1] 
    v_mat = sp.zeros(3, n_joints)      

    for i in range(n_joints):
        v_mat[:,i] = (-skew(omega_mat.col(i)) * q_mat.col(i))
    return v_mat

def Slist_maker(omega_mat, q_mat): 
    """
    Calculates Slist from omega's and q's\n
    PARAMETERS:
    omega_mat & q_mat: of type sp.Matrix with q_i & omega_i as columns
    RETURNS: Slist as sp.Matrix with Si as coloumns
    """
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

th1, th2, th3, th4, th5, th6 = dynamicsymbols('theta_1, theta_2, theta_3, theta_4, theta_5, theta_6')
def Jb_maker6x6(Blist): #Foreløpig for 6x6, fikset!
    Jb = sp.zeros(6,6)
    Jb[:,5] =  Blist[:,5]
    Jb[:,4] =  Ad(exp6(-Blist[:,5], th6)) * Blist[:,4]
    Jb[:,3] =  Ad(exp6(-Blist[:,5], th6) * exp6(-Blist[:,4], th5)) * Blist[:,3]
    Jb[:,2] =  Ad(exp6(-Blist[:,5], th6) * exp6(-Blist[:,4], th5) * exp6(-Blist[:,3], th4)) * Blist[:,2]
    Jb[:,1] =  Ad(exp6(-Blist[:,5], th6) * exp6(-Blist[:,4], th5) * exp6(-Blist[:,3], th4) * exp6(-Blist[:,2], th3)) * Blist[:,1]
    Jb[:,0] =  Ad(exp6(-Blist[:,5], th6) * exp6(-Blist[:,4], th5) * exp6(-Blist[:,3], th4) * exp6(-Blist[:,2], th3) * exp6(-Blist[:,1], th2)) * Blist[:,0]
    
    return Jb



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

def A1_sym(th_i, d_i):
    ct = sp.cos(th_i)
    st = sp.sin(th_i)
    A1 = sp.Matrix([[ct, -st, 0.0, 0.0], [st, ct, 0.0, 0.0], [0.0, 0.0, 1, d_i], [0.0, 0.0, 0.0, 1]])
    return A1
def A2_sym(al_i, a_i):
    ca = sp.cos(al_i)
    sa = sp.sin(al_i)
    A2 = sp.Matrix([[1, 0.0, 0.0, a_i], [0.0, ca, -sa, 0.0], [0.0, sa, ca, 0.0], [0.0, 0.0, 0.0, 1]])
    return A2

def T_from_sic(config):
    n = len(config.col(0))
    Alist = [sp.zeros(4,4)]*n
    Tlist = [sp.zeros(4,4)]*n
    for i in range(n):
        al_i = config[i,1]
        a_i = config[i,0]
        d_i = config[i,2]
        th_i = config[i,3]
        Alist[i] = A1_sym(th_i, d_i) * A2_sym(al_i, a_i)

        T = sp.eye(4)
        for j in range(i+1):
            T = T * Alist[j]
        Tlist[i] = T
    return Tlist

#IK functions
def ps_from_Tsd(T_sd):
    P_d = np.array([-80,0,0,1])
    P_s = T_sd@P_d
    return P_s


jointLimits = np.array([[-180, 180], [-190, 45], [-120, 156], [-180, 180], [-90, 90], [-180, 180]]) #Assuming joint 5 has limits [-90, 90]
def apply_joint_lim(jointLimits, thetas):
    ''' Check if Inverse Kinematics solution (thetas) is within jointlimits\n 
    PARAMETERS:
    Jointlimits, numpy 2D array with lower and upper limits in deg\n
    Jointangles, numpy 1D array in rad
    RETURNS: Boolean true or false, if false a print message with the offending link is printed '''
    jointLimits = np.deg2rad(jointLimits) 
    #thetas %= 2*np.pi #Post processing: all thetas in [0,2*pi)

    #for theta, i in enumerate(thetas):
    #    if jointLimits[i][1] < theta < jointLimits[i][0]:
    #        print("Joint number: ", i+1, "is not within the limits")
    #        return False
    #return True
    for i in range(0,len(thetas)):
        if jointLimits[i,1] < thetas[i] or thetas[i] < jointLimits[i,0]:
            print("Joint number: ", i+1, "is not within the limits")
            return False
    return True


def agilus_theta_23(T_sd):
    """
    Calculates theta 2 and 3 of the agilus 6R robot
    PARAMTERS:
    T_sd: The desired end effector pose
    RETURNS: floats, Thetas 2 and 3 for both elbow up and elbow down solutions.
    """

    Ps = ps_from_Tsd(T_sd)
    P2 = np.array([Ps[0],Ps[1],Ps[2]-400]) # The same as Ps, but now relative to joint 2. Needed to do the following trigonometrics

    # Define the edges of the constructed triangle:
    a = np.sqrt(420**2+35**2)
    c = 455
    b = np.sqrt((np.sqrt(P2[0]**2+P2[1]**2)-25)**2 + P2[2]**2)

    # Calculate the four angles needed:
    psi = np.arccos(420/a)
    phi = sp.atan2(P2[2], sp.sqrt(P2[0]**2 + P2[1]**2)-25)
    alpha = np.arccos((b**2+c**2-a**2)/(2*b*c))
    beta = np.arccos((a**2+c**2-b**2)/(2*a*c))

    # Calculate the elbow up and elbow down solutions of theta2 and theta3
    theta2_down =  -(phi - alpha)
    theta3_down =  -(np.pi - beta - psi)
    theta2_up = -(alpha + phi)
    theta3_up = np.pi - (beta - psi)

    return float(sp.N(theta2_up)), float(sp.N(theta3_up)), float(sp.N(theta2_down)), float(sp.N(theta3_down))



def euler_nx_y_nx(R):
    """
    Calculates the Euler angles for rotations about (-x)y(-x)
    PARAMETERS:
    R: The desired rotation
    RETURNS:
    float, Three angles
    """
    theta_x1 = -sp.atan2(R[1,0], -R[2,0])
    theta_y = sp.atan2(sp.sqrt(1-R[0,0]**2), R[0,0])
    theta_x2 = -sp.atan2(R[0,1], R[0,2])

    return float(sp.N(theta_x1)), float(sp.N(theta_y)), float(sp.N(theta_x2))


def agilus_analytical_IK(Slist,M,T_sd):
    """
    Computes the analytical inverse kinematics of the Agilus 6R robot.
    PARAMETERS:
    M: The home configuration
    Slist: An array with screw axes as columns
    Tsd: The desired end-effector pose
    RETURNS: two float arrays of joint values, elbow up and elbow down.
    """
    thetas_up = [0,0,0,0,0,0]
    thetas_down = [0,0,0,0,0,0]
    Ps = ps_from_Tsd(T_sd)

    # Theta 1
    thetas_up[0] = float(sp.N(-sp.atan2(Ps[1],Ps[0]))) # Minus sign since the axis of rotation is defined as -z.
    thetas_down[0] = thetas_up[0]

    # Thetas 2,3
    thetas_up[1], thetas_up[2], thetas_down[1], thetas_down[2] = agilus_theta_23(T_sd)

    # Thetas 4,5,6
     # Elbow down:
    T1 = exp6(Slist[:,0], -thetas_down[0])
    T2 = exp6(Slist[:,1], -thetas_down[1])
    T3 = exp6(Slist[:,2], -thetas_down[2])
    R_down = (T3@T2@T1@T_sd@np.linalg.inv(M))  # The remaining rotation needed, defined in s
    thetas_down[3], thetas_down[4], thetas_down[5] = euler_nx_y_nx(R_down)

     # Elbow up:
    T1 = exp6(Slist[:,0], -thetas_up[0])
    T2 = exp6(Slist[:,1], -thetas_up[1])
    T3 = exp6(Slist[:,2], -thetas_up[2])
    R_up = (T3@T2@T1@T_sd@np.linalg.inv(M))
    thetas_up[3], thetas_up[4], thetas_up[5] = euler_nx_y_nx(R_up)

    return thetas_up, thetas_down
