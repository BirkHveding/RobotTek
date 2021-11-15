import sympy as *

def skew(v):
    return Matrix([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])

def exp6(twist, theta):
    omega = skew(twist[:3])
    v = Matrix(twist[3:])
    T = eye(4)
    T[:3, :3] = exp3(twist[:3], theta)
    T[:3, 3] = (eye(3) * theta + (1 - cos(theta)) * omega +
                (theta - sin(theta)) * omega * omega) * v
    return T

def exp3(omega, theta):
    omega = skew(omega)
    R = eye(3) + sin(theta) * omega + (1 - cos(theta)) * omega * omega
    return R

def Ad(T):
    AdT = zeros(6)
    R = Matrix(T[:3, :3])
    AdT[:3, :3] = R
    AdT[3:, 3:] = R
    AdT[3:, :3] = skew(T[:3, 3]) * R
    return AdT

def JsMaker(Slist, theta_list):
    n_joints = Slist.shape[1]
    Js = zeros(6, n_joints)
    
    for i in range(n_joints-1, -1, -1):
        T = exp6(Slist[:,i-1], theta_list[i-1])
        
        for j in range( i-2, -1, -1):
            T = exp6(Slist[:,j], theta_list[j]) * T
          
        Js[:,i] = Ad(T) * Slist[:,i]
    Js.simplify()
    return Js

Mi = sp.Matrix([[sp.eye(4)]*6])

for i in range(6):
    if i == 1:
        Mi[:,4*i:4*(i+1)] = rotX(config[i,0]) * transX(config[i,1]) * transZ(config[i,2]) * rotZ(-sp.pi/2) # We compansate for the rotation of -pi/2 done when finding the D-H parameters 
    else:
        Mi[:,4*i:4*(i+1)] = rotX(config[i,0]) * transX(config[i,1]) * transZ(config[i,2])


Ai = sp.Matrix([[0,-1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]) # This is a given matrix due to revolute joints

S_sp = sp.zeros(6)
for i in range(6):
    dot_sum = sp.eye(4)
    for n in range(i,-1,-1):
        dot_sum = Mi[:,4*n:4*(n+1)] * dot_sum
    S_skew = dot_sum * Ai * sp.Inverse(dot_sum)
    S_sp[0,i] = S_skew[2,1] 
    S_sp[1,i] = S_skew[0,2] 
    S_sp[2,i] = S_skew[1,0] 
    S_sp[3,i] = S_skew[0,3] 
    S_sp[4,i] = S_skew[1,3] 
    S_sp[5,i] = S_skew[2,3] 