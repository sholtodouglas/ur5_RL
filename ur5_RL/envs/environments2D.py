# #
# physicsClient = p.connect(p.GUI) #p.direct for non GUI version
# p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
# p.setGravity(0,0,-10)
# planeId = p.loadURDF("plane.urdf")
# cubeStartPos = [0,0,4]
# cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
# boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)
# p.stepSimulation()
# cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)

# while cubePos[2] > 2:
# 	p.stepSimulation()
# 	cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
# print(cubePos,cubeOrn)
# p.disconnect()

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
from itertools import chain
from pybullet_utils import bullet_client

import random
import pybullet_data

from kuka import kuka
from ur5 import ur5
import sys
from scenes import * # where our loading stuff in functions are held


viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition = [0,0,0], distance = 0.3, yaw = 90, pitch = -90, roll = 0, upAxisIndex = 2) 
projectionMatrix = p.computeProjectionMatrixFOV(fov = 120,aspect = 1,nearVal = 0.01,farVal = 10)

image_renderer = p.ER_BULLET_HARDWARE_OPENGL # if the rendering throws errors, use ER_TINY_RENDERER, but its hella slow cause its cpu not gpu.

def setup_controllable_camera(p):
    p.addUserDebugParameter("Camera Zoom", -15, 15, 3)
    p.addUserDebugParameter("Camera Pan", -360, 360, 0)
    p.addUserDebugParameter("Camera Tilt", -360, 360, -90.5 )
    p.addUserDebugParameter("Camera X", -10, 10,0)
    p.addUserDebugParameter("Camera Y", -10, 10,0)
    p.addUserDebugParameter("Camera Z", -10, 10,0)


def update_camera(p):

    p.resetDebugVisualizerCamera(p.readUserDebugParameter(0),
                                     p.readUserDebugParameter(1),
                                     p.readUserDebugParameter(2),
                                     [p.readUserDebugParameter(3),
                                      p.readUserDebugParameter(4),
                                      p.readUserDebugParameter(5)])


class ur5Env_2D(gym.GoalEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=3,
                 isEnableSelfCollision=True,
                 renders=False,
                 arm = 'ur5',
                 vr = False,
                 pos_cntrl=False,
                 ag_only_self=True,
                 state_arm_pose=False,
                 only_xyz = True,
                 num_objects = 0):
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
        self.TARG_LIMIT = 0.85
        self.pos_cntrl = pos_cntrl
        self.ag_only_self = ag_only_self
        self.state_arm_pose = state_arm_pose
        self.only_xyz = only_xyz
        self.physics_client_active = 0

        self._seed()

        if pos_cntrl:
            action_dim = 4
        else:
            action_dim = 4 # however many motors there are? 
        if ag_only_self:
            goal_dim = 2
            obs_dim = 11 # motor poses 1,2,3, gripper, x y of head
        else:
            
            goal_dim = 2*num_objects # xyz,quat for each object. 
            obs_dim = 11+ 2*num_objects

        # actions are xyz space, quaternion, gripper.
        pi = math.pi 
        act_high = np.array([math.pi,math.pi,math.pi,1]) 
        act_low = np.array([-math.pi,-math.pi,-math.pi,0]) 
        self.action_space = spaces.Box(act_low, act_high) 
        high_obs = np.inf * np.ones([obs_dim])
        high_goal = np.inf * np.ones([goal_dim])
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-high_goal, high_goal),
            achieved_goal=spaces.Box(-high_goal, high_goal),
            observation=spaces.Box(-high_obs, high_obs),
        ))
        
        self.relevant_indices = [1,2,3,6] # of the arm motor pos/vels. 

        
        

    def reset(self, arm='ur5'):
        
        if self.physics_client_active == 0:
            print('Initialising Env.')
            # if self._renders:
            #     cid = p.connect(p.SHARED_MEMORY)
            #     if (cid < 0):
            #         cid = p.connect(p.GUI)
            #     if self._vr:
            #         #p.resetSimulation()
            #                         #disable rendering during loading makes it much faster
            #         p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            #     p.setRealTimeSimulation(1)
            # else:
            #     p.connect(p.DIRECT)
            if self._renders:
                self._p = bullet_client.BulletClient(connection_mode=p.GUI)
                setup_controllable_camera(self._p)
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
            #self._p.resetSimulation()
            self._p.setPhysicsEngineParameter(numSolverIterations=150)
            
            
                
                
            if self.ag_only_self:
                self.objects = scene_2D(self._p)
            else:
                self.objects = scene_2D_object(self._p)

            self._p.setGravity(0, 0, -10)
            if self._arm_str == 'rbx1':
                self._arm = rbx1(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
            elif self._arm_str == 'kuka':
                self._arm = kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep, vr = self._vr)
            else:
                self._arm = load_arm_dim_up(self._p,'ur5',dim='2D')
            
            sphereRadius = 0.03
            mass = 1
            visualShapeId = 2
            colSphereId = self._p.createCollisionShape(self._p.GEOM_SPHERE,radius=sphereRadius)
            
            self.goal = self._p.createMultiBody(mass,colSphereId,1,[1,1,1.4])
            # self.mass = [p.loadURDF((os.path.join(urdfRoot,"sphere2.urdf")), 0,0.0,1.0,1.00000,0.707107,0.000000,0.707107)]
            collisionFilterGroup = 0
            collisionFilterMask = 0
            self._p.setCollisionFilterGroupMask(self.goal, -1, collisionFilterGroup, collisionFilterMask)
            self.goal_cid = self._p.createConstraint(self.goal,-1,-1,-1,self._p.JOINT_FIXED,[1,1,1.4],[0,0,0],[0,0,0],[0,0,0,1])
        else:
            print('Resetting')


        self._envStepCounter = 0
        self.reset_goal_pos()
        self.reset_objects(self.objects)
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
                
                goal_pos = [goal_x,goal_y,0.1]

            self.goal_pos = goal_pos[0:2]
            self._p.resetBasePositionAndOrientation(self.goal, goal_pos, [0,0,0,1])
            self._p.changeConstraint(self.goal_cid,goal_pos, maxForce = 100)



    def reset_objects(self, objects, positions=None):
        
        for idx, o in enumerate(objects):
            if positions is None: 
                new_x  = self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT)
                new_y  = self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT)
                new_z = self.np_random.uniform(low=0.1, high=self.TARG_LIMIT)
                new_pos = [new_x,new_y,new_z]
            else:
                new_pos = positions[idx][0:3]

            self._p.resetBasePositionAndOrientation(o[0], new_pos, [0,0,0,1])


    def getSceneObservation(self):
        arm_obs = self._arm.state()

        scene_obs = get_scene_observation(self._p,self.objects)

        # convert all this to 2D as relevant.
        arm_obs['joint_positions'] = [arm_obs['joint_positions'][i] for i in self.relevant_indices]
        arm_obs['joint_velocities'] = [arm_obs['joint_velocities'][i] for i in self.relevant_indices]
        arm_obs['pos'] = list(arm_obs['pos'][0:2]) 


        
        observation = list(arm_obs['joint_positions']) + list(arm_obs['joint_velocities']) + list(arm_obs['pos'][0:2]) + list(arm_obs['gripper'])

        #top_down_img = p.getCameraImage(500, 500, viewMatrix,projectionMatrix, shadow=0,renderer=image_renderer)
        #grip_img = gripper_camera(self._observation)
        if self.ag_only_self:
            achieved_goal = np.array(arm_obs['pos'][0:2])
        else:
            observation += list(scene_obs[0:2])
            achieved_goal = np.array(list(scene_obs[0:2])) # note is only pos of first object doesn't includes orienation of scene obs.

        goal  = np.array(self.goal_pos)
        return {
                'observation': np.array(observation).copy().astype('float32'),
                'achieved_goal': achieved_goal.copy().astype('float32'),
                'desired_goal':  goal.copy().astype('float32'),
            }

    #moves motors to desired pos
    def step(self, action):
        # in 2D, we have a 4 dimensional action space. 
        # in 2D, we only want motor control not pos control - too hard!
        if self._renders:
            update_camera(self._p)
        if self.pos_cntrl:
            
            action[2] = 0.187
            
            self._arm.move_to(action)
        else:
            constraints=  self._arm.reset_pos
            # thus, assuming it comes in in 4D.
            new_action = [0,0,0,0,0,0,0,0]
            new_action[0] = constraints[0]
            new_action[1:4] = action[0:3]
            new_action[4] =  constraints[4]
            new_action[5] = constraints[5]
            new_action[6] = action[3]/5
            new_action[7] = -action[3]/5
            self._arm.action(new_action)
        
        for i in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            
            self._envStepCounter += 1
        
        obs = self.getSceneObservation()
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])
        
        return obs, reward, False, {}

    def _termination(self):
        if (self.terminated):
            return True

    def render(self, mode='human', close=False):
            if (mode=="human"):
                self._renders = True

                return np.array([])
            if mode == 'rgb_array':
                raise NotImplementedError


    def compute_reward(self, achieved_goal, desired_goal, info = None):

        distance = np.sum(abs(achieved_goal-desired_goal))
        if distance < 0.1:
            reward  = 1
        else:
            reward = 0 
        
        return reward

    def gripper_camera(self, obs): 
        # Center of mass position and orientation (of link-7)
        pos = obs[-7:-4] 
        ori = obs[-4:] # last 4
        # rotation = list(p.getEulerFromQuaternion(ori))
        # rotation[2] = 0
        # ori = p.getQuaternionFromEuler(rotation)

        rot_matrix = self._p.getMatrixFromQuaternion(ori)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # Initial vectors
        init_camera_vector = (1, 0, 0) # z-axis
        init_up_vector = (0, 1, 0) # y-axis
        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix_gripper = self._p.computeViewMatrix(pos, pos + 0.1 * camera_vector, up_vector)
        img = self._p.getCameraImage(200, 200, view_matrix_gripper, projectionMatrix,shadow=0, flags = self._p.ER_NO_SEGMENTATION_MASK, renderer=image_renderer)

        

##############################################################################################################




# a version where reward is determined by the obstacle. 
# need a sub class so we can load it as a openai gym env.
class ur5Env_2D_objects(ur5Env_2D):
    def __init__(self,
                 renders=False,
                 ag_only_self=False,
                 ):
        super().__init__(renders = renders, ag_only_self = ag_only_self, num_objects = 1)

##############################################################################################################

def move_in_xyz(environment, arm, abs_rel):

    motorsIds = []

    dv = 0.01
    abs_distance =  1.0
    observation = environment._arm.state()
    xyz = observation['pos']

    ori = p.getEulerFromQuaternion(observation['orn'])
    original_ori   = observation['orn']
    
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
        motorsIds.append(environment._p.addUserDebugParameter("dX", -dv, dv, 0))
        motorsIds.append(environment._p.addUserDebugParameter("dY", -dv, dv, 0))
        motorsIds.append(environment._p.addUserDebugParameter("dZ", -dv, dv, 0))
        if arm == 'rbx1':
            motorsIds.append(environment._p.addUserDebugParameter("wrist_rotation", -0.1, 0.1, 0))
            motorsIds.append(environment._p.addUserDebugParameter("wrist_flexsion", -0.1, 0.1, 0))
        else:
            motorsIds.append(environment._p.addUserDebugParameter("roll", -dv, dv, 0))
            motorsIds.append(environment._p.addUserDebugParameter("pitch", -dv, dv, 0))
            motorsIds.append(environment._p.addUserDebugParameter("yaw", -dv, dv, 0))
    motorsIds.append(environment._p.addUserDebugParameter("fingerAngle", 0, 1.5, .3))

    done = False
    while (not done):

        action = []

        for motorId in motorsIds:
            # print(environment._p.readUserDebugParameter(motorId))
            action.append(environment._p.readUserDebugParameter(motorId))


        update_camera(environment._p)
        
        #environment._p.addUserDebugLine(environment._arm.endEffectorPos, [0, 0, 0], [1, 0, 0], 5)
        
        # action is xyz positon, orietnation quaternion, gripper closedness. 
        action = action[0:3] + list(p.getQuaternionFromEuler(action[3:6])) + [action[6]]
        state, reward, done, info = environment.step(action)
        obs = environment.getSceneObservation()

##############################################################################################################







def setup_controllable_motors(environment, arm):
    

    possible_range = 3.2  # some seem to go to 3, 2.5 is a good rule of thumb to limit range.
    motorsIds = []

    observation = environment._arm.state()
    angles = observation['joint_positions']

    for tests in range(0, environment._arm.numJoints-2):  # motors

        jointInfo = environment._p.getJointInfo(environment._arm.uid, tests)
        #print(jointInfo)
        qIndex = jointInfo[3]

        if tests in [1,2,3]: # only motors that count in 2D
            motorsIds.append(environment._p.addUserDebugParameter("Motor" + str(tests),
                                                              -possible_range,
                                                              possible_range,
                                                              angles[tests]))
        if tests == 6:
            motorsIds.append(environment._p.addUserDebugParameter("Motor" + str(tests),
                                                              -0,
                                                              1,
                                                              angles[tests]))


    return motorsIds





def send_commands_to_motor(environment, motorIds):

    done = False


    while (not done):
        action = []

        for motorId in motorIds:
            action.append(environment._p.readUserDebugParameter(motorId))
        
        state, reward, done, info = environment.step(action)
        obs = environment.getSceneObservation()
        update_camera(environment._p)


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
    
    environment = ur5Env_2D_objects(renders=str_to_bool(render))
    environment.reset()
    

    print(mode)
    if mode == 'xyz':
            move_in_xyz(environment, arm, abs_rel)
    else:
        environment._arm.active = True

        control_individual_motors(environment, arm)



@click.command()
@click.option('--mode', type=str, default='motor', help='motor: control individual motors, xyz: control xyz/rpw of gripper, demos: collect automated demos')
@click.option('--abs_rel', type=str, default='abs', help='absolute or relative positioning, abs doesnt really work with rbx1 yet')
@click.option('--arm', type=str, default='ur5', help='rbx1 or kuka')
@click.option('--render', type=bool, default=True, help='rendering')



def main(**kwargs):
    launch(**kwargs)

if __name__ == "__main__":
    main()



        