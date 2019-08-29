import pybullet as p
from environments import *
import os
import math

import socket

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server



def move_in_xyz(environment, arm, abs_rel,s):
    motorsIds = []

    dv = 0.01
    abs_distance = 1.0
    observation = environment._arm.state()
    xyz = observation['pos']

    ori = p.getEulerFromQuaternion(observation['orn'][0:4])
    original_ori = observation['orn'][0:4]
    print(original_ori)
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

        # update_camera(environment._p)

        # environment._p.addUserDebugLine(environment._arm.endEffectorPos, [0, 0, 0], [1, 0, 0], 5)

        # action is xyz positon, orietnation quaternion, gripper closedness.
        action = action[0:3] + list(p.getQuaternionFromEuler(action[3:6])) + [action[6]]
        state, reward, done, info = environment.step(action)
        obs = environment.getSceneObservation()

        try:
            s.sendall(b'Hello, world')
            data = s.recv(1024)
            print('Received', repr(data))
        except:
            print("Connection Failed")





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


def launch( arm, abs_rel, render):
    print(arm)
    # environment = ur5Env(renders=str_to_bool(render), arm = arm)
    environment = ur5Env_objects(renders=str_to_bool(render))
    environment.reset()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
        except:
            print("conneciton refused")
        move_in_xyz(environment, arm, abs_rel,s)








@click.command()
@click.option('--abs_rel', type=str, default='abs',
              help='absolute or relative positioning, abs doesnt really work with rbx1 yet')
@click.option('--arm', type=str, default='ur5', help='rbx1 or kuka')
@click.option('--render', type=bool, default=True, help='rendering')
def main(**kwargs):
    launch(**kwargs)


if __name__ == "__main__":
    main()