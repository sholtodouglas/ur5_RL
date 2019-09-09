
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import math
import gym
import sys
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from itertools import chain
from collections import deque

import random
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import functools
import time
import itertools
from cntrl import *
from pyRobotiqGripper import *
real = False

def setup_sisbot(p, uid):
    # controlJoints = ["shoulder_pan_joint","shoulder_lift_joint",
    #                  "elbow_joint", "wrist_1_joint",
    #                  "wrist_2_joint", "wrist_3_joint",
    #                  "robotiq_85_left_knuckle_joint"]
    controlJoints = ["shoulder_pan_joint","shoulder_lift_joint",
                     "elbow_joint", "wrist_1_joint",
                     "wrist_2_joint", "wrist_3_joint", 'left_gripper_motor', 'right_gripper_motor']

    jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    numJoints = p.getNumJoints(uid)
    jointInfo = namedtuple("jointInfo", 
                           ["id","name","type","lowerLimit","upperLimit","maxForce","maxVelocity","controllable"])
    joints = AttrDict()
    for i in range(numJoints):
        info = p.getJointInfo(uid, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        jointLowerLimit = info[8]
        jointUpperLimit = info[9]
        jointMaxForce = info[10]
        jointMaxVelocity = info[11]
        controllable = True if jointName in controlJoints else False
        info = jointInfo(jointID,jointName,jointType,jointLowerLimit,
                         jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
        if info.type=="REVOLUTE": # set revolute joint to static
            p.setJointMotorControl2(uid, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        joints[info.name] = info
    controlRobotiqC2 = False
    mimicParentName = False
    return joints, controlRobotiqC2, controlJoints, mimicParentName



class ur5:

    def __init__(self,p, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01, vr = False, three_D = True):

        self._p =p
        self.three_D = three_D
        if three_D:
            self.reset_pos = [0.22192452542166125, -1.6382572664293273, -1.7993126794035792, -1.3204855586878317, 1.5692558868410538, 0.26404106438702235, 0.012000000479180678, -0.0120000004745803]
            self.robotUrdfPath = os.path.dirname(os.path.abspath(__file__))+ "/urdf/real_arm.urdf" 
        else:
            self.reset_pos = [0.015103069759782975, 1.8509778040375553, -1.7288175651068152, 0.6191207734395402, 1.5939804808300704, 1.5599556847373455, 0.01200060652908279, -0.011999410708707916]
            self.robotUrdfPath = os.path.dirname(os.path.abspath(__file__))+ "/urdf/2D_arm.urdf" 


        
        self.robotStartPos = [0.0,0.0,0.0]
        self.robotStartOrn = self._p.getQuaternionFromEuler([1.885,1.786,0.132])

        self.xin = self.robotStartPos[0]
        self.yin = self.robotStartPos[1]

        self.zin = self.robotStartPos[2]
        self.lastJointAngle = None
        self.active = False
        if real:
            self.s = init_socket()

            if True:
                self.grip=RobotiqGripper("COM8")
                #grip.resetActivate()
                self.grip.reset()
                #grip.printInfo()
                self.grip.activate()
                #grip.printInfo()
                #grip.calibrate()




        self.reset()
        self.timeout = 0

    def reset(self):
        
        print("----------------------------------------")
        print("Loading robot from {}".format(self.robotUrdfPath))
        self.uid = self._p.loadURDF(os.path.join(os.getcwd(),self.robotUrdfPath), self.robotStartPos, self.robotStartOrn, 
                             flags=self._p.URDF_USE_INERTIA_FROM_FILE)
        self.joints, self.controlRobotiqC2, self.controlJoints, self.mimicParentName = setup_sisbot(self._p, self.uid)
        self.endEffectorIndex = 7 # ee_link
        self.numJoints = self._p.getNumJoints(self.uid)
        self.active_joint_ids = []
        for i, name in enumerate(self.controlJoints):
            joint = self.joints[name]
            self.active_joint_ids.append(joint.id)



    def getActionDimension(self):
        # if (self.useInverseKinematics):
        #     return len(self.motorIndices)
        return 8  # position x,y,z and ori quat and finger angle
    def stateDimension(self):
        return len(self.state())

    def setPosition(self, pos, quat):

        self._p.resetBasePositionAndOrientation(self.uid,pos,
                                          quat)

    # motor commands is 8 dimensional, one for each joint. 
    def setJointPose(self,motorCommands):

        for i, name in enumerate(self.controlJoints):
            joint = self.joints[name]
            self._p.resetJointState(self.uid, joint.id, motorCommands[i])


    def resetJointPoses(self):
                # move to this ideal init point
        if real:
            self.active = False
            self.setJointPose()
            self.active = True
            movej(self.s,self.reset_pos,a=0.01,v=0.05,t=10.0)
            time.sleep(10)
        else:
            self.setJointPose(self.reset_pos)
        self.lastJointAngle = self.reset_pos


    def state(self):
        observation = []
        state = self._p.getLinkState(self.uid, self.endEffectorIndex, computeLinkVelocity = 1)
        #print('state',state)
        pos = state[0]
        orn = state[1]
        pos_vel = state[-2]
        orn_vel = state[-1]

        observation.extend(list(orn))
        observation.extend(list(pos_vel))
        observation.extend(list(orn_vel))

        joint_states = self._p.getJointStates(self.uid, self.active_joint_ids)
        
        joint_positions = list()
        joint_velocities = list()
        

        for joint in joint_states:
            
            joint_positions.append(joint[0])
            joint_velocities.append(joint[1])
            
        
  
        return {'joint_positions': joint_positions, 'joint_velocities':joint_velocities, 'pos': pos, 'orn':orn,
                                        'gripper':[joint_positions[-2]*25],'pos_vel':pos_vel, 'orn_vel':orn_vel }



    def action(self, motorCommands):
        #print(motorCommands)
        
        poses = []
        indexes = []
        forces = []


        # if self.lastJointAngle == None:
        #     self.lastJointAngle =  motorCommands[0:6]

        # rel_a = np.array(motorCommands[0:6]) - np.array(self.lastJointAngle) 

        # clipped_a = np.clip(rel_a, -0.1, 0.1)
        # motorCommands[0:6] = list(clipped_a+self.lastJointAngle)
        # self.lastJointAngle =  motorCommands[0:6]

        for i, name in enumerate(self.controlJoints):
            joint = self.joints[name]
            if not self.three_D:
                if i in [1,2,3,6]:

                    poses.append(motorCommands[i]*10)
                    indexes.append(joint.id)
                    forces.append(joint.maxForce)
                else:
                    poses.append(0)
                    indexes.append(joint.id)
            else:
                poses.append(motorCommands[i])
                indexes.append(joint.id)
                forces.append(joint.maxForce)

        l = len(poses)
        positionGains = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.03, 0.03]
        
        if self.three_D:
            self._p.setJointMotorControlArray(self.uid, indexes, self._p.POSITION_CONTROL, targetPositions=poses, targetVelocities =[0.0]*l, 
        #holy shit this is so much faster in arrayform!
        else:
            # well this is just a bit shit hey - if we want this to work, need to fix the positions of 0,4,5,7.
            # huh. Fuck 2D, straight to threeD?
            self._p.setJointMotorControlArray(self.uid, indexes, self._p.VELOCITY_CONTROL, forces = np.array(poses))

        if real and self.active:
            
            if time.time() > self.timeout+0.05:
                servoj(self.s,poses[0:6],a=0,v=0,t=0.05, gain = 100, lookahead_time = 0.05)
                self.timeout = time.time()


                grip_angle =  max(0, min(255,int(poses[6]*255/0.04)))  # range 0 - 0.04
                self.grip.goTo(grip_angle)


 





    

    def move_to(self, position_delta, mode = 'abs', noise = False, clip = False):

        #at the moment UR5 only absolute
        #TODO Don't have this so limiting?
        position_delta = np.clip(position_delta, [-0.14, -0.6, -0.1, -1, -1, -1, -1, 0], [1, 1, 1, 1, 1, 1, 1, 1])

        x = position_delta[0]
        y = position_delta[1]
        z = position_delta[2]
        
        orn = position_delta[3:7]
        finger_angle = position_delta[7]

        # define our limits. 
        z = max(0.14, min(0.7,z))
        x = max(-0.3, min(0.7,x))
        y =max(-0.6, min(0.6,y))
        if real:
            
            x = max(-0.25, min(0.3,x))
            y =max(-0.4, min(0.4,y))


        jointPose = list(self._p.calculateInverseKinematics(self.uid, self.endEffectorIndex, [x,y,z], orn))

        # print(jointPose)
        # print(self.state()[:len(self.controlJoints)]) ## get the current joint positions
        
        jointPose[7] = -finger_angle/25 
        jointPose[6] = finger_angle/25
        
        self.action(jointPose)
        #print(jointPose)
        return jointPose

