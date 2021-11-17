import open3d as o3d
import numpy as np
import modern_robotics as mr
from open3d.web_visualizer import draw
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

class Robot:
    #Parameters: 
    # Mlist: Pose of all joints in zero-config as homogenous transformation
    # link_orient: orientation of link in next joints frame (including ground to Link1) ex: ['z', '-z', 'x', 'x', 'z','x']
    def __init__(self, Mlist, link_orient='x'):
        self.robotObjects = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=75)]
        self.current_config = Mlist
        self.Mlist = Mlist
        self.num_joints = len(Mlist)
        self.num_links = len(Mlist)
        self.length_links = self.findLinkLengths(Mlist)
        self.link_orient = ['x']*(self.num_links+1) if link_orient == 'x' else link_orient #link attached to preceeding joints x-axis by default

        self.joints = []  # Of class Joint
        self.links = []  # Of class Link
        self.make_robot_objects()  # create all objects of robot (links, frames, joints)
        self.update_mesh_list()
        self.__transform(Mlist) # Transforms all objects from {s} to zero-config

# Calculates link lengths based on M
    def findLinkLengths(self, Mlist):
        linkLengths = np.zeros(self.num_links)

        for i in range(self.num_links):
            p = Mlist[i][:3, 3]
            p_pre = np.array([0, 0, 0]) if i == 0 else Mlist[i-1][:3, 3] #1. link from ground ([0,0,0]) to 1. joint
            linkLengths[i] = np.linalg.norm(p_pre-p)
            linkLengths[linkLengths <= 0] = 0.1 # cant have zero-length links because of transformation logic
        return linkLengths

# creates all o3d-objects of the robot in {s}
    def make_robot_objects(self):
        for i in range(self.num_joints):
            self.joints.append(Joint())
        for i in range(self.num_links):
            self.links.append(Link(self.length_links[i], self.link_orient[i]))

    def update_mesh_list(self):
        for Joint in self.joints:
            self.robotObjects.append(Joint.joint)
            self.robotObjects.append(Joint.coord)
        for Link in self.links:
            self.robotObjects.append(Link.link)

    def allToOrigin(self):  # Sends all objects to Origin
        T_origin = []
        for T in self.current_config:
            T_origin.append(mr.TransInv(T))
        self.__transform(T_origin)
        return

    def transform(self, Slist, thetas):
        self.allToOrigin()
        T_list = []  # List to fill with T01,T02,T03...
        T = np.eye(4)
        for i in range(len(thetas)):
            T = T @ exp6(Slist[:, i], thetas[i])
            T_list.append(T*self.Mlist[i])
        self.__transform(T_list)
        self.current_config = T_list
        return

    # Moves all objects from {s} to config given by T_list
    def __transform(self, T_list):  # Private member function
        for i, J in enumerate(self.joints):
            J.transform(T_list[i])
        for i, L in enumerate(self.links):
            T_links = np.concatenate(([np.eye(4)], T_list[:-1])) #Transform links after joint
            L.transform(T_links[i])

    def draw_robot(self):  # Draws all o3d objects in robotObjects list
        draw(self.robotObjects)

#______________Joint Class_______________#
class Joint(Robot):
    def __init__(self):
        self.joint = o3d.geometry.TriangleMesh.create_cylinder(
            radius=10, height=30)
        self.coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25)
        self.set_colour()

    def set_colour(self, colour=[0, 1, 0]):
        self.joint.paint_uniform_color(colour)

    def transform(self, T):
        self.joint = self.joint.transform(T)
        self.coord = self.coord.transform(T)

#_______________Link Class_________________#
class Link(Robot):
    def __init__(self, lenght, orient):
        self.lenght = lenght

        if (orient == 'x'):  # Defines link direction from preceeding joint
            self.link = o3d.geometry.TriangleMesh.create_cylinder(radius=1, height=self.lenght).rotate(
                Ry_sym(np.pi/2)).translate(np.array([self.lenght/2, 0, 0]))
        elif (orient == 'y'):
            self.link = o3d.geometry.TriangleMesh.create_cylinder(radius=1, height=self.lenght).rotate(
                Rx_sym(-np.pi/2)).translate(np.array([0, self.lenght/2, 0]))
        elif (orient == 'z'):
            self.link = o3d.geometry.TriangleMesh.create_cylinder(
                radius=1, height=self.lenght).translate(np.array([0, 0, self.lenght/2]))
        elif (orient == '-z'):
            self.link = o3d.geometry.TriangleMesh.create_cylinder(
                radius=1, height=self.lenght).translate(np.array([0, 0, -self.lenght/2]))
        self.set_colour()

    def set_colour(self, colour=[0, 0, 1]):
        self.link.paint_uniform_color(colour)

    def transform(self, T):
        self.link = self.link.transform(T)