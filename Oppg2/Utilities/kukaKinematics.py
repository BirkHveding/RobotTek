import sympy as sp
import numpy as np
from symbolicFunctions import Slist_maker


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
M4=sp.Matrix([[0, 0, -1, 455+25+420],
             [0, 1, 0, 0],
             [1, 0, 0, 400],
             [0, 0, 0, 1]])
M5=sp.Matrix([[1, 0, 0, 455+25+420],
             [0, 0, 1, 0],
             [0, -1, 0, 400],
             [0, 0, 0, 1]])
M6=sp.Matrix([[0, 0, -1, 455+25+420+50], ##OBS lagt til 50 for å se endeffector
             [0, 1, 0, 0],
             [1, 0, 0, 400],#400
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

S_list = Slist_maker(om,q)

