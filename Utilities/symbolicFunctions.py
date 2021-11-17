import sympy as sp

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
