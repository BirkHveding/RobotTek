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
    T[:3, :3] = exp3(twist[:3], theta)
    T[:3, 3] = (sp.eye(3) * theta + (1 - sp.cos(theta)) * omega +
                (theta-sp.sin(theta)) * omega * omega) * v
    return T


class Robot:
    '''
    ## Parameters:\n
    Mlist: Pose of all joints in zero-config as homogenous transformation\n
    link_orient: list, axis of preceeding joints frame to attach links (including link from ground to joint1) ex: ['z', '-z', 'x', 'x', 'z','x']\n
    endEffectorOffset: T, offset of endeffector from last link, given as a Homogeneous transformation matrix. Default no offset
    '''

    def __init__(self, Mlist, link_orient='x', endEffectorOffset=sp.Matrix(sp.eye(4))):
        self.robotObjects = [o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=75)]  # Initated with {s}-frame coord-frame
        self.current_config = Mlist  # list of T's giving pose for all joints
        self.Mlist = Mlist
        self.num_joints = len(Mlist)
        self.num_links = len(Mlist)
        self.length_links = self.findLinkLengths()
        self.link_orient = ['x']*(self.num_links +
                                  1) if link_orient == 'x' else link_orient  # link attached to preceeding joints x-axis by default
        self.Tne = endEffectorOffset

        self.joints = []  # Elements of class Joint
        self.links = []  # Elements of class Link
        self.__make_robot_objects()  # create all objects of robot (links, frames, joints)
        self.update_mesh_list() # Update robotObject list (only objects in list will be drawn)
        self.__transform(Mlist) # Transforms all objects from {s} to zero-config

# Calculates link lengths based on M
    def findLinkLengths(self):
        linkLengths = np.zeros(self.num_links)

        for i in range(self.num_links):
            p = self.Mlist[i][:3, 3]
            # 1. link from ground ([0,0,0]) to 1. joint
            p_pre = np.array([0, 0, 0]) if i == 0 else self.Mlist[i-1][:3, 3]
            linkLengths[i] = np.linalg.norm(p_pre-p)
            # cant have zero-length links because of transformation logic
            linkLengths[linkLengths <= 0] = 0.1
        return linkLengths

    # creates all o3d-objects of the robot in {s}
    def __make_robot_objects(self):
        for i in range(self.num_joints):
            self.joints.append(Joint())
        for i in range(self.num_links):
            self.links.append(Link(self.length_links[i], self.link_orient[i]))

        #Creates endeffector frame if Te != Tn
        if self.Tne != sp.Matrix(sp.eye(4)): # True if endeffector-offset is given
            self.endEffectorObject = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=50)
            self.robotObjects.append(self.endEffectorObject)
        else:
            self.endEffectorObject = False

    def update_mesh_list(self):
        for Joint in self.joints:
            self.robotObjects.append(Joint.joint)
            self.robotObjects.append(Joint.coord)
        for Link in self.links:
            self.robotObjects.append(Link.link)
    
    # Sends all objects to Origin
    def allToOrigin(self):  
        T_origin = []
        for T in self.current_config:
            T_origin.append(mr.TransInv(T))
        self.__transform(T_origin)
        return

    def transform(self, Slist, thetas): 
        self.allToOrigin() # o3d interpret transforms as relative to current pose of object
        T_list = []  # List to fill with T01,T02,T03...
        T = np.eye(4)
        for i in range(len(thetas)):
            T = T @ exp6(Slist[:, i], thetas[i])
            T_list.append(T*self.Mlist[i])
        self.__transform(T_list)
        self.current_config = T_list
        return

    # Moves all objects from {s} to config given by T_list
    def __transform(self, T_list): 
        # Displace endeffector
        if self.endEffectorObject:
            # self.endEffectorObject.transform(mr.TransInv(self.current_config[-1]*self.Tne)) #T = Tse^-1
            self.endEffectorObject.transform(
                T_list[-1]*self.Tne)

        for i, J in enumerate(self.joints):
            J.transform(T_list[i])
        for i, L in enumerate(self.links):
            # Displace links with T of preceeding joint
            T_links = np.concatenate(([np.eye(4)], T_list[:-1]))
            L.transform(T_links[i])
            

    # Draws all o3d objects in robotObjects list
    def draw_robot(self, method = 1):
        '''method 1: Draws in Jupyter cell-output
        method 2: Draws in own window'''
        if method == 1:
            draw(self.robotObjects)
        elif method == 2:
            o3d.visualization.draw_geometries(self.robotObjects)


#______________Joint Class_______________#


class Joint(Robot):
    def __init__(self):
        self.joint = o3d.geometry.TriangleMesh.create_cylinder(
            radius=15, height=30)
        self.coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60)
        self.set_colour()

    def set_colour(self, colour=[0.4, 0.4, 0.4]):
        self.joint.paint_uniform_color(colour)

    def transform(self, T):
        self.joint = self.joint.transform(T)
        self.coord = self.coord.transform(T)

#_______________Link Class_________________#


class Link(Robot):
    def __init__(self, lenght, orient):
        self.lenght = lenght

        if (orient == 'x'):  # Defines link direction from preceeding joint
            self.link = o3d.geometry.TriangleMesh.create_cylinder(radius=2, height=self.lenght).rotate(
                Ry_sym(np.pi/2)).translate(np.array([self.lenght/2, 0, 0]))
        elif (orient == 'y'):
            self.link = o3d.geometry.TriangleMesh.create_cylinder(radius=2, height=self.lenght).rotate(
                Rx_sym(-np.pi/2)).translate(np.array([0, self.lenght/2, 0]))
        elif (orient == 'z'):
            self.link = o3d.geometry.TriangleMesh.create_cylinder(
                radius=2, height=self.lenght).translate(np.array([0, 0, self.lenght/2]))
        elif (orient == '-z'):
            self.link = o3d.geometry.TriangleMesh.create_cylinder(
                radius=2, height=self.lenght).translate(np.array([0, 0, -self.lenght/2]))
        self.set_colour()

    def set_colour(self, colour=[0, 0, 0]):
        self.link.paint_uniform_color(colour)

    def transform(self, T):
        self.link = self.link.transform(T)
