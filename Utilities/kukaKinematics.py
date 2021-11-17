############################
#Kinematics for Kuka robot, includes:
# M
# Mlist - consisting of M for each joint as if it was the end-effector joint
# Slist and S1-S6

import sympy as sp
import numpy as np

def skew(v):
    return sp.Matrix([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])

def Slist_maker(omega_mat, q_mat): #omega_mat and q_mat of type matrix with q_i & omega_i as columns
    #Returns v_mat in same type/format
    v_mat = calc_v(omega_mat, q_mat)    
    n_joints = omega_mat.shape[1]
    Slist = sp.zeros(6, n_joints)
    
    for i in range(n_joints):
        Slist[:3,i] = omega_mat[:,i]
        Slist[3:,i] = v_mat[:,i]
    return Slist

def calc_v(omega_mat, q_mat):
    #omega_mat and q_mat of type matrix with q_i & omega_i as columns
    #Returns v_mat in same type/format
    assert len(omega_mat) == len(q_mat)
    
    n_joints = omega_mat.shape[1] 
    v_mat = sp.zeros(3, n_joints)      

    for i in range(n_joints):
        v_mat[:,i] = (-skew(omega_mat.col(i)) * q_mat.col(i))
    return v_mat



M1=sp.Matrix([[0, 1, 0, 0],
             [1, 0, 0, 0],
             [0, 0, -1, 200],
             [0, 0, 0, 1]])

M2=sp.Matrix([[1, 0, 0, 25],
             [0, 0, 1, 0],
             [0, -1, 0, 400],
             [0, 0, 0, 1]])

M3=sp.Matrix([[1, 0, 0, 455+25],
             [0, 0, 1, 0],
             [0, -1, 0, 400],
             [0, 0, 0, 1]])
M4=sp.Matrix([[0, 0, -1, 455+25+200], #420
             [0, 1, 0, 0],
             [1, 0, 0, 400+35],
             [0, 0, 0, 1]])
M5=sp.Matrix([[1, 0, 0, 455+25+420],
             [0, 0, 1, 0],
             [0, -1, 0, 400+35],
             [0, 0, 0, 1]])
M6=sp.Matrix([[0, 0, -1, 455+25+420+50], ##OBS lagt til 50 for å se endeffector
             [0, 1, 0, 0],
             [1, 0, 0, 400+35],
             [0, 0, 0, 1]])
Mlist = np.array([M1,M2,M3,M4,M5,M6], dtype=float)

om = sp.zeros(3,6)
om1 = om[:, 0] = M1[:3, 2]
om2 = om[:, 1] = M2[:3, 2]
om3 = om[:, 2] = M3[:3, 2]
om4 = om[:, 3] = M4[:3, 2]
om5 = om[:, 4] = M5[:3, 2]
om6 = om[:, 5] = M6[:3, 2]
q = sp.zeros(3,6)
q1 = q[:,0] = M1[:3, 3]
q2 = q[:,1] = M2[:3, 3]
q3 = q[:,2] = M3[:3, 3]
q4 = q[:,3] = M4[:3, 3]
q5 = q[:,4] = M5[:3, 3]
q6 = q[:,5] = M6[:3, 3]

Slist = Slist_maker(om,q)
S1 = Slist[:,0]
S2 = Slist[:,1]
S3 = Slist[:,2]
S4 = Slist[:,3]
S5 = Slist[:,4]
S6 = Slist[:,5]