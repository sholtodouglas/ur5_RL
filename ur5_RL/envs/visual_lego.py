# #
# physicsClient =self._p.connect(p.GUI) #p.direct for non GUI version
# self._p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
# self._p.setGravity(0,0,-10)
# planeId =self._p.loadURDF("plane.urdf")
# cubeStartPos = [0,0,4]
# cubeStartOrientation =self._p.getQuaternionFromEuler([0,0,0])
# boxId =self._p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)
# self._p.stepSimulation()
# cubePos, cubeOrn =self._p.getBasePositionAndOrientation(boxId)

# while cubePos[2] > 2:
# 	p.stepSimulation()
# 	cubePos, cubeOrn =self._p.getBasePositionAndOrientation(boxId)
# print(cubePos,cubeOrn)
# self._p.disconnect()

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
from scenes import *  # where our loading stuff in functions are held


image_renderer = p.ER_BULLET_HARDWARE_OPENGL  # if the rendering throws errors, use ER_TINY_RENDERER, but its hella slow cause its cpu not gpu.

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

viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.00, 0, 0], distance=1.2, yaw=90, pitch=-90,
                                                 roll=0, upAxisIndex=2)
viewMatrix_1 = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0], distance=0.4, yaw=135, pitch=-35,
                                                   roll=0, upAxisIndex=2)

projectionMatrix = p.computeProjectionMatrixFOV(fov=25, aspect=1, nearVal=0.01, farVal=10)


def gripper_camera(obs):
    # Center of mass position and orientation (of link-7)
    pos = obs[0:3]
    ori = obs[7:11]  # last 4
    # rotation = list(p.getEulerFromQuaternion(ori))
    # rotation[2] = 0
    # ori = p.getQuaternionFromEuler(rotation)

    rot_matrix = p.getMatrixFromQuaternion(ori)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    # Initial vectors
    init_camera_vector = (1, 0, 0)  # z-axis
    init_up_vector = (0, 1, 0)  # y-axis
    # Rotated vectors
    camera_vector = rot_matrix.dot(init_camera_vector)
    up_vector = rot_matrix.dot(init_up_vector)
    view_matrix_gripper = p.computeViewMatrix(pos, pos + 0.1 * camera_vector, up_vector)
    img = p.getCameraImage(200, 200, view_matrix_gripper, projectionMatrix, shadow=0, flags=p.ER_NO_SEGMENTATION_MASK,
                           renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return img


class ur5Env(gym.GoalEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=10,
                 isEnableSelfCollision=True,
                 renders=False,
                 arm='ur5',
                 vr=False,
                 pos_cntrl=True,
                 ag_only_self=True,
                 state_arm_pose=True,
                 only_xyz=False,
                 num_objects=0,
                 relative=False,
                 only_xyzr=True,
                 only_xy=False,
                 reward_scaling=1,
                 pointmass_test=False,
                 curriculum_learn=False,
                 tools=False):
        # pos_cntrl is whether we control the motors or we control the position of
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
        self.only_xyzr = only_xyzr
        self.only_xy = only_xy
        self.physics_client_active = 0
        self.relative = relative
        self._seed()
        self.roving_goal = False
        self.reward_scaling = reward_scaling
        self.pointmass_test = pointmass_test
        self.curriculum_learn = curriculum_learn
        self.tools = tools

        if pos_cntrl:
            action_dim = 8
        else:
            raise NotImplementedError
            action_dim = 7  # however many motors there are?

        if self.state_arm_pose:
            # + 3 + 1 +3  +4 + 8,
            obs_dim = 19
        else:
            # size 3 + 1 + 3
            obs_dim = 7  # xyz, quat, gripper


        if ag_only_self:
            goal_dim = 3

        else:
            if self.tools:
                goal_dim = 3
                obs_dim += 7 * num_objects
            else:
                goal_dim = 3 * num_objects  # xyz,quat for each object.
                obs_dim += 7 * num_objects

        # actions are xyz space, quaternion, gripper.
        act_high = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        act_low = np.array([-1, -1, -1, -1, -1, -1, -1, 0])
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
            print('***********************', self._renders)

            if self._renders:
                self._p = bullet_client.BulletClient(connection_mode=p.GUI)

                # setup_controllable_camera(self._p)
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
            # p.resetSimulation()
            self._p.setPhysicsEngineParameter(numSolverIterations=150)


            self.objects =  lego_scene(self._p)

            self._p.setGravity(0, 0, -10)
            if self._arm_str == 'rbx1':
                self._arm = rbx1(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
            elif self._arm_str == 'kuka':
                self._arm = kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep, vr=self._vr)
            else:
                self._arm = load_arm_dim_up(self._p, 'ur5', dim='Z')

            if self._renders:
                self._p.resetDebugVisualizerCamera(1.25, 90, -45, [0, 0, 0])

            self._arm.resetJointPoses()
            arm_obs = self._arm.state()
            self.default_ori = arm_obs['orn']


            ##################################################################################################

        else:
            pass

        self._envStepCounter = 0
        self.goal_pos = np.array([1,2,3])
        self.reset_objects()
        self.reset_goal_pos()
        self._arm.resetJointPoses()

        if self.pointmass_test:
            self._p.resetBasePositionAndOrientation(self.mass, [0, 0, 0.05], [0, 0, 0, 1])
            self._p.changeConstraint(self.mass_cid, [0, 0, 0.05], maxForce=100)
        self.reset_pos  =np.array([-0.25,0,0.5])
        done = False
        while not done:
            done = self.go_to_point(self.reset_pos, self.default_ori, gripper = 1) #reset

        return self.get_viz_obs()

    def __del__(self):
        self._p.disconnect()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode):
        if (mode == "human"):
            self._renders = True
            return np.array([])
        if mode == 'rgb_array':
            raise NotImplementedError

    def reset_goal_pos(self, goal_pos=None):
        return 0



    def reset_objects(self, positions=None):

        for idx, o in enumerate(self.objects):

            if positions is None:

                new_x = self.np_random.uniform(low=-self.TARG_LIMIT / 2, high=self.TARG_LIMIT / 2)
                new_y = self.np_random.uniform(low=-self.TARG_LIMIT / 2, high=self.TARG_LIMIT / 2)
                new_z = self.np_random.uniform(low=0.05, high=0.05)
                new_pos = [new_x,new_y,new_z, 0,0,0,1]
            else:
                new_pos = positions[idx]
            self._p.resetBasePositionAndOrientation(o[0], new_pos[0:3], new_pos[3:7])

    def initialize_start_pos(self, start_state):
        # the only time we will be initializeing this is if we have the joint positions.
        #  according to state arm pose
        for i in range(0, 100):  # for some reason needs this to make sure
            self._arm.setJointPose(start_state[11:19])
            obj_list = []
            for o in range(0, len(self.objects)):
                obj_list.append(start_state[19 + o * 7:26 + o * 7])

            self.reset_objects(obj_list)  # one object

    def getSceneObservation(self):
        arm_obs = self._arm.state()

        scene_obs = get_scene_observation(self._p, self.objects)

        observation = list(arm_obs['pos']) + list(arm_obs['gripper']) + list(arm_obs['pos_vel']) + list(arm_obs['orn']) + list(arm_obs['joint_positions'])  # + list(arm_obs['orn_vel'])#+ list(arm_obs['joint_velocities'])

        top_down_img =self._p.getCameraImage(64, 64, viewMatrix,projectionMatrix, shadow=0,renderer=image_renderer)

        achieved_goal = np.array(arm_obs['pos'])
        goal = np.array(self.goal_pos)

        return {
            'observation': np.array(observation).copy().astype('float32'),
            'achieved_goal': achieved_goal.copy().astype('float32'),
            'desired_goal': goal.copy().astype('float32'),
        }

    # moves motors to desired pos


    def go_to_point(self,position, ori, gripper = 0, slow_factor = 8):


        gripperPos = self._arm.state()['pos']
        self._p.addUserDebugLine(position, gripperPos, lifeTime = 1.0)
        object_rel_pos = position - gripperPos  # lastObs['observation'][6:9]
        action = [0, 0, 0, 0, 0, 0, 0, 0]
        action[3:7] =  ori
        object_oriented_goal = object_rel_pos.copy()/slow_factor + gripperPos
        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]# * 6

        action[len(action) - 1] = gripper  # 0 for open, 1 for closed
        self.step_sim(action)
        done = False

        if np.linalg.norm(self._arm.state()['pos']-position) <= 0.005:

            done = True
        return done

# 0.3 and 0.125

    def grasp_primitive(self, point, angle = 0):
        time_limit = 100
        done = False
        cnt = 0
        ori = p.getQuaternionFromEuler(np.array(p.getEulerFromQuaternion(self.default_ori)) + np.array([angle*math.pi/180.0, 0, 0]))
        while not done and cnt < time_limit:

            done = self.go_to_point(point + np.array([0,0,0.3]), ori, gripper=0, slow_factor =1) # go above the point
            cnt += 1

        done = False

        cnt = 0
        while not done and cnt < time_limit:

            done = self.go_to_point(point + np.array([0, 0, 0.135]), ori, gripper=0, slow_factor = 8) # descend on the point
            cnt += 1
        done = False
        for i in range(0,20):
            self.go_to_point(point + np.array([0, 0, 0.135]), ori, gripper=1, slow_factor = 8) # close the gripper
        cnt = 0
        while not done and cnt < time_limit:
            done = self.go_to_point(np.array([-0.25,0,0.5]),ori, gripper = 1, slow_factor = 12) #take object up
            cnt += 1


    def push_primitive(self, point, angle = 0):

        time_limit = 100
        done = False
        cnt = 0
        ori = p.getQuaternionFromEuler(
            np.array(p.getEulerFromQuaternion(self.default_ori)) + np.array([angle * math.pi / 180.0, 0, 0]))



    def step_sim(self, action):


        # if self._renders:
        # update_camera(self._p)
        #top_down_img = self._p.getCameraImage(500, 500, viewMatrix, projectionMatrix, shadow=0, renderer=image_renderer)

        action = np.array(action).copy()
        # new_x,new_y = action[0], action[1]

        if self.pos_cntrl:


            #action[3:7] = self.default_ori


            self._arm.move_to(action)
        else:
            self._arm.action(action)

        for i in range(self._actionRepeat):
            self._p.stepSimulation()
            # if self._renders:
            # time.sleep(self._timeStep/self._actionRepeat)


    def get_viz_obs(self):
        top_down_img = self._p.getCameraImage(128, 128, viewMatrix, projectionMatrix, shadow=0, renderer=image_renderer)
        depth = top_down_img[3]
        image = ((top_down_img[2][:,:,0:3]).astype(np.float32)/255, (depth-np.mean(depth))*500)
        goal = np.array(self.goal_pos)
        obs = {
            'observation': image,
            'achieved_goal': image,
            'desired_goal': goal.copy().astype('float32'),
        }

        return obs

    def step(self, action, angle = 0):
        self.grasp_primitive(action[0:3], action[3])

        self._envStepCounter += 1




        reward = 0
        completed = 0
        for o in self.objects:
            pos, orn = self._p.getBasePositionAndOrientation(o[0])

            if pos[2] > 0.3:
                self._p.resetBasePositionAndOrientation(o[0], [0.7+np.random.random()*0.3, 0.7+np.random.random()*0.2, 0.05], [0.1, 0.05, 0.01, 1])
                
                reward  += 1

            pos, orn = self._p.getBasePositionAndOrientation(o[0]) # i.e check if its been reset
            if pos[1] > 0.7:
                completed += 1


        #reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal']) * self.reward_scaling

        success = 0 if reward < 1 else 1  # assuming negative rewards
        done = False
        if completed == len(self.objects):

            done = True

        obs = self.get_viz_obs()

        return obs, reward, done, {'is_success': success}

    def render(self, mode='human', close=False):
        if (mode == "human"):
            self._renders = True
            return np.array([])
        if mode == 'rgb_array':
            raise NotImplementedError

    def compute_reward(self, achieved_goal, desired_goal, info=None):

        if len(achieved_goal.shape) > 1:
            # vectorized
            distance = np.sum(abs(achieved_goal - desired_goal), axis=1)
        else:  # single examples
            distance = np.sum(abs(achieved_goal - desired_goal))

        return -(distance > 0.05).astype(np.float32)

        # if positive
        # return (dist<0.3).astype(np.float32)

    def activate_roving_goal(self):
        self.roving_goal = True

    def gripper_camera(self, obs):
        # Center of mass position and orientation (of link-7)
        pos = obs[-7:-4]
        ori = obs[-4:]  # last 4
        # rotation = list(p.getEulerFromQuaternion(ori))
        # rotation[2] = 0
        # ori =self._p.getQuaternionFromEuler(rotation)

        rot_matrix = self._p.getMatrixFromQuaternion(ori)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # Initial vectors
        init_camera_vector = (1, 0, 0)  # z-axis
        init_up_vector = (0, 1, 0)  # y-axis
        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix_gripper = self._p.computeViewMatrix(pos, pos + 0.1 * camera_vector, up_vector)
        img = self._p.getCameraImage(200, 200, view_matrix_gripper, projectionMatrix, shadow=0,
                                     flags=self._p.ER_NO_SEGMENTATION_MASK, renderer=image_renderer)


class ur5Env_lego(ur5Env):
    def __init__(self,
                 renders=False,
                 ag_only_self=False,
                 relative=False,
                 curriculum_learn=False
                 ):
        super().__init__(curriculum_learn=curriculum_learn, renders=renders, ag_only_self=ag_only_self, num_objects=4,
                         relative=relative)






##############################################################################################################


def move_in_xyz(environment, arm, abs_rel):
    print('dacing')
    motorsIds = []

    dv = 0.1
    abs_distance = 1.0
    observation = environment._arm.state()
    xyz = observation['pos']

    ori = p.getEulerFromQuaternion(observation['orn'][0:4])
    original_ori = observation['orn'][0:4]

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
            rin = math.pi / 2
            pitchin = -math.pi / 2
            yawin = 0

        motorsIds.append(environment._p.addUserDebugParameter("X", -abs_distance, abs_distance, xin))
        motorsIds.append(environment._p.addUserDebugParameter("Y", -abs_distance, abs_distance, yin))
        motorsIds.append(environment._p.addUserDebugParameter("Z", -abs_distance, abs_distance, zin))
        motorsIds.append(environment._p.addUserDebugParameter("roll", -math.pi, math.pi, rin, ))
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
    gg = environment._p.addUserDebugParameter("grasper", 0, 1.5, 0)

    done = False

    while (not done):

        action = []

        for motorId in motorsIds:
            action.append(environment._p.readUserDebugParameter(motorId))

        if environment._p.readUserDebugParameter(gg) > 1:
            environment.grasp_primitive(np.array([0,0,0]))

        # update_camera(environment._p)

        # environment._p.addUserDebugLine(environment._arm.endEffectorPos, [0, 0, 0], [1, 0, 0], 5)

        # action is xyz positon, orietnation quaternion, gripper closedness.

        action = action[0:3] + list(p.getQuaternionFromEuler(action[3:6])) + [action[6]]
        observation = environment._arm.state()

        ori = p.getEulerFromQuaternion(observation['orn'][0:4])
        print(np.array(ori)*180/math.pi)


        environment.step_sim(action)



##############################################################################################################


def setup_controllable_motors(environment, arm):
    possible_range = 3.2  # some seem to go to 3, 2.5 is a good rule of thumb to limit range.
    motorsIds = []

    for tests in range(0, environment._arm.numJoints):  # motors

        jointInfo = self._p.getJointInfo(environment._arm.uid, tests)
        # print(jointInfo)
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
        # update_camera(environment._p)

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
        pass  # directory already exists


#####################################################################################

def str_to_bool(string):
    if str(string).lower() == "true":
        string = True
    elif str(string).lower() == "false":
        string = False

    return string


def launch(mode, arm, abs_rel, render):
    print(arm)

    # environment = ur5Env(renders=str_to_bool(render), relative=  False, only_xy = True, pointmass_test=True)
    # environment = ur5Env_objects(renders=str_to_bool(render), relative=  False)
    environment = ur5Env_lego(renders=True, relative=False)

    environment.reset()

    print(mode)
    if mode == 'xyz':
        move_in_xyz(environment, arm, abs_rel)
    else:
        environment._arm.active = True

        control_individual_motors(environment, arm)


@click.command()
@click.option('--mode', type=str, default='xyz',
              help='motor: control individual motors, xyz: control xyz/rpw of gripper, demos: collect automated demos')
@click.option('--abs_rel', type=str, default='abs',
              help='absolute or relative positioning, abs doesnt really work with rbx1 yet')
@click.option('--arm', type=str, default='ur5', help='rbx1 or kuka')
@click.option('--render', type=bool, default=True, help='rendering')
def main(**kwargs):
    launch(**kwargs)


if __name__ == "__main__":
    main()



