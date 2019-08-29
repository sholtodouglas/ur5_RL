# #
# physicsClient =self._p.connect(p.GUI) #p.direct for non GUI version
#self._p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
#self._p.setGravity(0,0,-10)
# planeId =self._p.loadURDF("plane.urdf")
# cubeStartPos = [0,0,4]
# cubeStartOrientation =self._p.getQuaternionFromEuler([0,0,0])
# boxId =self._p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)
#self._p.stepSimulation()
# cubePos, cubeOrn =self._p.getBasePositionAndOrientation(boxId)

# while cubePos[2] > 2:
# 	p.stepSimulation()
# 	cubePos, cubeOrn =self._p.getBasePositionAndOrientation(boxId)
# print(cubePos,cubeOrn)
#self._p.disconnect()

import os, inspect
import time

from tqdm import tqdm
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import click
import math 
import gym
import sys
from gym import spaces
from gym.utils import seeding 
import numpy as np
import time
import pybullet as p
from pybullet_utils import bullet_client
from itertools import chain

import random
import pybullet_data

from kuka import kuka
from ur5 import ur5
import sys
from scenes import * # where our loading stuff in functions are held


viewMatrix =p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition = [0,0,0], distance = 0.3, yaw = 90, pitch = -90, roll = 0, upAxisIndex = 2) 
projectionMatrix =p.computeProjectionMatrixFOV(fov = 120,aspect = 1,nearVal = 0.01,farVal = 10)

image_renderer = p.ER_BULLET_HARDWARE_OPENGL # if the rendering throws errors, use ER_TINY_RENDERER, but its hella slow cause its cpu not gpu.

# def setup_controllable_camera(p):
#     p.addUserDebugParameter("Camera Zoom", -15, 15, 2)
#     p.addUserDebugParameter("Camera Pan", -360, 360, 30)
#     p.addUserDebugParameter("Camera Tilt", -360, 360, -40.5 )
#     p.addUserDebugParameter("Camera X", -10, 10,0)
#     p.addUserDebugParameter("Camera Y", -10, 10,0)
#     p.addUserDebugParameter("Camera Z", -10, 10,0)


# def update_camera(p):

#     p.resetDebugVisualizerCamera(p.readUserDebugParameter(0),
#                                      p.readUserDebugParameter(1),
#                                      p.readUserDebugParameter(2),
#                                      [p.readUserDebugParameter(3),
#                                       p.readUserDebugParameter(4),
#                                       p.readUserDebugParameter(5)])

class RingBuffer:
    def __init__(self, size):
        self.data = [np.zeros(7) for i in range(0,size)]

    def append(self, x):
        self.data.pop(0)
        self.data.append(x)

    def get(self):
        return np.mean(self.data, axis = 0)



class ur5Env(gym.GoalEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=1,
                 isEnableSelfCollision=True,
                 renders=False,
                 arm = 'ur5',
                 vr = False,
                 pos_cntrl=True,
                 ag_only_self=True,
                 state_arm_pose=True,
                 only_xyz = True,
                 num_objects = 0,
                 relative=False):
        #pos_cntrl is whether we control the motors or we control the position of 
        # head and gripper. 
        # ag_only_self is whether we want to have the objects as the achieved goal
        # or the tip position as the achieved goal
        # state arm pose is whether we return the motor positions.
        # only xyz is whether we can modify the ori as well.
        self._timeStep = 1. / 240.
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._envStepCounter = 0
        self._renders = renders
        self._vr = vr
        self.terminated = 0
        self._p = p
        self.TARG_LIMIT = 0.25
        self.pos_cntrl = pos_cntrl
        self.ag_only_self = ag_only_self
        self.state_arm_pose = state_arm_pose
        self.only_xyz = only_xyz
        self.physics_client_active = 0
        self.relative = relative
        self._seed()
        self.action_buffer = RingBuffer(10)
        self.roving_goal = False
        if pos_cntrl:
            action_dim = 8
        else:
            raise NotImplementedError
            action_dim = 7 # however many motors there are? 





        if self.state_arm_pose:
            # + 3 + 1 +3  +4 + 8,
            obs_dim = 19
        else:
            # size 3 + 1 + 3
            obs_dim = 7 # xyz, quat, gripper
            if not self.only_xyz:
                obs_dim += 7

        if ag_only_self:
            goal_dim = 3
            
        else:

            goal_dim = 3*num_objects # xyz,quat for each object.
            obs_dim += 7*num_objects


        # actions are xyz space, quaternion, gripper. 
        act_high = np.array([1,1,1,1,1,1,1,1]) 
        act_low = np.array([-1,-1,-1,-1,-1,-1,-1,0]) 
        self.action_space = spaces.Box(act_low, act_high) 
        high_obs = np.inf * np.ones([obs_dim])
        high_goal = np.inf * np.ones([goal_dim])
        
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-high_goal, high_goal),
            achieved_goal=spaces.Box(-high_goal, high_goal),
            observation=spaces.Box(-high_obs, high_obs),
        ))
        
        

    def reset(self, arm='ur5'):
        
        if self.physics_client_active == 0:
            print('Initialising Env.')
            # if self._renders:
            #     cid =self._p.connect(p.SHARED_MEMORY)
                
            #     if (cid < 0):
            #         cid =self._p.connect(p.GUI)
            #     if self._vr:
            #         #p.resetSimulation()
            #                         #disable rendering during loading makes it much faster
            #        self._p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            #    self._p.setRealTimeSimulation(1)
            # else:
            #    self._p.connect(p.DIRECT)
            if self._renders:
                self._p = bullet_client.BulletClient(connection_mode=p.GUI)
                #setup_controllable_camera(self._p)
            else:
                self._p = bullet_client.BulletClient(connection_mode=p.DIRECT)


            self.physics_client_active = 1

            self._seed()
            self._arm_str = arm
            
            if self._vr:
               self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)
               self._p.setRealTimeSimulation(1)
            else:
               self._p.setTimeStep(self._timeStep)
            

            self.terminated = 0
            #p.resetSimulation()
            self._p.setPhysicsEngineParameter(numSolverIterations=150)
            
            print(self._p)
            if self.ag_only_self:
                self.objects = basic_scene(self._p)
            else:
                self.objects = one_block_scene(self._p)
            

            self._p.setGravity(0, 0, -10)
            if self._arm_str == 'rbx1':
                self._arm = rbx1(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
            elif self._arm_str == 'kuka':
                self._arm = kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep, vr = self._vr)
            else:
                self._arm = load_arm_dim_up(self._p,'ur5',dim='Z')
            
            sphereRadius = 0.03
            mass = 1
            visualShapeId = 2
            colSphereId = self._p.createCollisionShape(self._p.GEOM_SPHERE,radius=sphereRadius)
            
            if self._renders:
                self._p.resetDebugVisualizerCamera(0.75,90,-45,[0,0,0])
                self.goal = self._p.createMultiBody(mass,colSphereId,1,[1,1,1.4])
                collisionFilterGroup = 0
                collisionFilterMask = 0
                print('the moment')
                self._p.setCollisionFilterGroupMask(self.goal, -1, collisionFilterGroup, collisionFilterMask)
                self.goal_cid = self._p.createConstraint(self.goal,-1,-1,-1,self._p.JOINT_FIXED,[1,1,1.4],[0,0,0],[0,0,0],[0,0,0,1])
        else:
            print('Resetting')


        self._envStepCounter = 0
        self.reset_goal_pos()
        self._arm.resetJointPoses()
        arm_obs = self._arm.state()
        self.default_ori = arm_obs['orn']


        return self.getSceneObservation()

    def __del__(self):
       self._p.disconnect()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def render(self, mode):
            if (mode=="human"):
                self._renders = True
                return np.array([])
            if mode == 'rgb_array':
                raise NotImplementedError

    def reset_goal_pos(self, goal_pos = None):
            
            if goal_pos is None: 
                goal_x  = self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT)
                goal_y  = self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT)
                if self.ag_only_self:
                        goal_z = self.np_random.uniform(low=0.2, high=self.TARG_LIMIT)
                else:
                    goal_z = self.np_random.uniform(low=0.05, high=0.05)
                goal_pos = [goal_x,goal_y,goal_z]

            self.goal_pos = goal_pos

            # try:
            #     self._p.removeUserDebugItem(self.goal)
            # except:
            #     pass

            # self.goal = self._p.addUserDebugText("o", goal_pos,
            #        textColorRGB=[0, 0, 1],
            #        textSize=1)

            if self._renders:
                self._p.resetBasePositionAndOrientation(self.goal, goal_pos, [0,0,0,1])
                self._p.changeConstraint(self.goal_cid,goal_pos, maxForce = 100)

    def reset_objects(self, positions=None):

        for idx, o in enumerate(self.objects):
            if positions is None: 
                new_x  = self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT)
                new_y  = self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT)
                new_z = self.np_random.uniform(low=0.1, high=self.TARG_LIMIT)
                new_pos = [new_x,new_y,new_z]
            else:
                new_pos = positions[idx]

            self._p.resetBasePositionAndOrientation(o, new_pos[0:3], new_pos[3:7])

    def initialize_start_pos(self,start_state):
        # the only time we will be initializeing this is if we have the joint positions.
        #  according to state arm pose
        self._arm.setJointPose(start_state[11:19])
        obj_list= []
        for o in range(0,len(self.objects)):
            obj_list.append(start_state[19+o*7:26+o*7])
        self.reset_objects(obj_list) # one object





    def getSceneObservation(self):
        arm_obs = self._arm.state()

        scene_obs = get_scene_observation(self._p,self.objects)


        if self.state_arm_pose:
            #  + 3 + 1 +3  +4 + 8, ////  +3  +8,
            observation =  list(arm_obs['pos']) +   list(arm_obs['gripper']) + list(arm_obs['pos_vel'])+ list(arm_obs['orn']) + list(arm_obs['joint_positions']) #+ list(arm_obs['orn_vel'])#+ list(arm_obs['joint_velocities'])
        else:
            # size 3 + 1 + 3
            observation = list(arm_obs['pos']) + list(arm_obs['gripper']) + list(arm_obs['pos_vel'])

        #top_down_img =self._p.getCameraImage(500, 500, viewMatrix,projectionMatrix, shadow=0,renderer=image_renderer)
        #grip_img = self.gripper_camera(self._observation)
        if self.ag_only_self:
            achieved_goal = np.array(arm_obs['pos'])
        else:
            index = 0
            for o in self.objects:
                achieved_goal = np.array(scene_obs[index:index+3]) # note ag is only pos of first object doesn't includes orienation of scene obs.
                index+= 7 #to skip the orientation. at the moment we only care about ag in terms of position
            observation += list(scene_obs) # but observation gets the full obs

        goal  = np.array(self.goal_pos)


        
        return {
                'observation': np.array(observation).copy().astype('float32'),
                'achieved_goal': achieved_goal.copy().astype('float32'),
                'desired_goal':  goal.copy().astype('float32'),
            }

    #moves motors to desired pos
    def step(self, action):
        #if self._renders:
            #update_camera(self._p)

        action = np.array(action)
        # self.action_buffer.append(action[0:7])
        if self.relative:
            action[0:7] = action[0:7]**3
            action = list(action)

            observation = self._arm.state()
            # action is xyz positon, orietnation quaternion, gripper closedness.
            commanded_ori = list(action[3:7])
            action[0:3] = list(np.array(action[0:3]) + np.array(observation['pos']))
            commanded_ori = list((np.array(commanded_ori) + np.array(observation['orn'])))
            action = action[0:3] + commanded_ori + [action[7]]


        if self.pos_cntrl:
            
            if self.only_xyz: # we want to make sure the orientation here is always down
                action[3:7] = self.default_ori
            
            self._arm.move_to(action)
        else:
            self._arm.action(action)
        
        for i in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            
            self._envStepCounter += 1

        obs = self.getSceneObservation()
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])

        if self.roving_goal:
            
            if self._envStepCounter % 60 == 0:
                    print('rove')
                    self.reset_goal_pos()

        return obs, reward, False, {}



    def render(self, mode='human', close=False):
            if (mode=="human"):
                self._renders = True
                return np.array([])
            if mode == 'rgb_array':
                raise NotImplementedError


    def compute_reward(self, achieved_goal, desired_goal, info = None):

        distance = np.sum(abs(achieved_goal-desired_goal))
        
        if distance < 0.03:
            reward  = 50.0
        else:
            reward = 0.0
        
        
        return reward

    def activate_roving_goal(self):
        self.roving_goal = True

    def gripper_camera(self,obs): 
        # Center of mass position and orientation (of link-7)
        pos = obs[-7:-4] 
        ori = obs[-4:] # last 4
        # rotation = list(p.getEulerFromQuaternion(ori))
        # rotation[2] = 0
        # ori =self._p.getQuaternionFromEuler(rotation)

        rot_matrix =self._p.getMatrixFromQuaternion(ori)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # Initial vectors
        init_camera_vector = (1, 0, 0) # z-axis
        init_up_vector = (0, 1, 0) # y-axis
        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix_gripper =self._p.computeViewMatrix(pos, pos + 0.1 * camera_vector, up_vector)
        img =self._p.getCameraImage(200, 200, view_matrix_gripper, projectionMatrix,shadow=0, flags =self._p.ER_NO_SEGMENTATION_MASK, renderer=image_renderer)



class ur5Env_objects(ur5Env):
    def __init__(self,
                 renders=False,
                 ag_only_self=False,
                 ):
        super().__init__(renders = renders, ag_only_self = ag_only_self, num_objects=1)

class ur5Env_reacher_relative(ur5Env):
    def __init__(self,
                 renders=False,
                 ag_only_self=True,
                 ):
        super().__init__(renders = renders, ag_only_self = ag_only_self, relative=True)
    

##############################################################################################################




def move_in_xyz(environment, arm, abs_rel):

    motorsIds = []

    dv = 0.1
    abs_distance =  1.0
    observation = environment._arm.state()
    xyz = observation['pos']

    ori =p.getEulerFromQuaternion(observation['orn'][0:4])
    original_ori   = observation['orn'][0:4]

    if abs_rel == 'abs': 
        print(arm)

        if arm == 'ur5':
            xin = xyz[0]
            yin = xyz[1]
            zin = xyz[2]
            rin = ori[0]
            pitchin = ori[1]
            yawin = ori[2]
        else:
            xin = 0.537
            yin = 0.0
            zin = 0.5
            rin = math.pi/2
            pitchin = -math.pi/2
            yawin = 0

        motorsIds.append(environment._p.addUserDebugParameter("X", -abs_distance, abs_distance, xin))
        motorsIds.append(environment._p.addUserDebugParameter("Y", -abs_distance, abs_distance, yin))
        motorsIds.append(environment._p.addUserDebugParameter("Z", -abs_distance, abs_distance, zin))
        motorsIds.append(environment._p.addUserDebugParameter("roll", -math.pi, math.pi, rin,))
        motorsIds.append(environment._p.addUserDebugParameter("pitch", -math.pi, math.pi, pitchin))
        motorsIds.append(environment._p.addUserDebugParameter("yaw", -math.pi, math.pi, yawin))

    else:
        environment.relative = True
        motorsIds.append(environment._p.addUserDebugParameter("dX", -dv, dv, 0))
        motorsIds.append(environment._p.addUserDebugParameter("dY", -dv, dv, 0))
        motorsIds.append(environment._p.addUserDebugParameter("dZ", -dv, dv, 0))
        motorsIds.append(environment._p.addUserDebugParameter("roll", -dv, dv, 0))
        motorsIds.append(environment._p.addUserDebugParameter("pitch", -dv, dv, 0))
        motorsIds.append(environment._p.addUserDebugParameter("yaw", -dv, dv, 0))
    motorsIds.append(environment._p.addUserDebugParameter("fingerAngle", 0, 1.5, .1))

    done = False
    while (not done):


        action = []

        for motorId in motorsIds:

            action.append(environment._p.readUserDebugParameter(motorId))


        #update_camera(environment._p)
        
        #environment._p.addUserDebugLine(environment._arm.endEffectorPos, [0, 0, 0], [1, 0, 0], 5)

        # action is xyz positon, orietnation quaternion, gripper closedness.

        action = action[0:3] + list(p.getQuaternionFromEuler(action[3:6])) + [action[6]]


        state, reward, done, info = environment.step(action)
        obs = environment.getSceneObservation()

##############################################################################################################





def setup_controllable_motors(environment, arm):
    

    possible_range = 3.2  # some seem to go to 3, 2.5 is a good rule of thumb to limit range.
    motorsIds = []

    for tests in range(0, environment._arm.numJoints):  # motors

        jointInfo =self._p.getJointInfo(environment._arm.uid, tests)
        #print(jointInfo)
        qIndex = jointInfo[3]

        if arm == 'kuka':
            if qIndex > -1 and jointInfo[0] != 7:
        
                motorsIds.append(environment._p.addUserDebugParameter("Motor" + str(tests),
                                                              -possible_range,
                                                              possible_range,
                                                              0.0))
        else:
            motorsIds.append(environment._p.addUserDebugParameter("Motor" + str(tests),
                                                              -possible_range,
                                                              possible_range,
                                                              0.0))

    return motorsIds



def send_commands_to_motor(environment, motorIds):

    done = False


    while (not done):
        action = []

        for motorId in motorIds:
            action.append(environment._p.readUserDebugParameter(motorId))
        
        state, reward, done, info = environment.step(action)
        obs = environment.getSceneObservation()
        #update_camera(environment._p)


    environment.terminated = 1

def control_individual_motors(environment, arm):
    environment.pos_cntrl = False
    motorIds = setup_controllable_motors(environment, arm)
    send_commands_to_motor(environment, motorIds)


###################################################################################################
def make_dir(string):
    try:
        os.makedirs(string)
    except FileExistsError:
        pass # directory already exists



#####################################################################################

def str_to_bool(string):
    if str(string).lower() == "true":
            string = True
    elif str(string).lower() == "false":
            string = False

    return string



def launch(mode, arm, abs_rel, render):
    print(arm)
    
    #environment = ur5Env(renders=str_to_bool(render), arm = arm)
    environment = ur5Env_objects(renders=str_to_bool(render))
    environment.reset()

    print(mode)
    if mode == 'xyz':
            move_in_xyz(environment, arm, abs_rel)
    else:
        environment._arm.active = True

        control_individual_motors(environment, arm)



@click.command()
@click.option('--mode', type=str, default='xyz', help='motor: control individual motors, xyz: control xyz/rpw of gripper, demos: collect automated demos')
@click.option('--abs_rel', type=str, default='abs', help='absolute or relative positioning, abs doesnt really work with rbx1 yet')
@click.option('--arm', type=str, default='ur5', help='rbx1 or kuka')
@click.option('--render', type=bool, default=True, help='rendering')



def main(**kwargs):
    launch(**kwargs)

if __name__ == "__main__":
    main()