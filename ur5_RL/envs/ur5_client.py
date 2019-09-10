import pybullet as p
from environments import *
import os
import math
import pickle
import socket
import time
HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition = [0,0,0], distance = 0.4, yaw = 45, pitch = -35, roll = 0, upAxisIndex = 2) 
viewMatrix_1 = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition = [0,0,0], distance = 0.4, yaw = 135, pitch = -35, roll = 0, upAxisIndex = 2) 

projectionMatrix = p.computeProjectionMatrixFOV(fov = 120,aspect = 1,nearVal = 0.01,farVal = 10)


def gripper_camera(obs):
    # Center of mass position and orientation (of link-7)
    pos = obs[0:3] 
    ori = obs[7:11] # last 4
    # rotation = list(p.getEulerFromQuaternion(ori))
    # rotation[2] = 0
    # ori = p.getQuaternionFromEuler(rotation)

    rot_matrix = p.getMatrixFromQuaternion(ori)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    # Initial vectors
    init_camera_vector = (1, 0, 0) # z-axis
    init_up_vector = (0, 1, 0) # y-axis
    # Rotated vectors
    camera_vector = rot_matrix.dot(init_camera_vector)
    up_vector = rot_matrix.dot(init_up_vector)
    view_matrix_gripper = p.computeViewMatrix(pos, pos + 0.1 * camera_vector, up_vector)
    img = p.getCameraImage(200, 200, view_matrix_gripper, projectionMatrix,shadow=0, flags = p.ER_NO_SEGMENTATION_MASK, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return img


def move_in_xyz(environment, arm, abs_rel,s, record, play_sequence):
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
    actions = []
    observations = []
    step = 0
    while (not done):

        action = []

        for motorId in motorsIds:
            # print(environment._p.readUserDebugParameter(motorId))
            action.append(environment._p.readUserDebugParameter(motorId))

        # update_camera(environment._p)

        # environment._p.addUserDebugLine(environment._arm.endEffectorPos, [0, 0, 0], [1, 0, 0], 5)

        # action is xyz positon, orietnation quaternion, gripper closedness.
        #action = action[0:3] + list(p.getQuaternionFromEuler(action[3:6])) + [action[6]]
        

        try:
            #print('sending')
            s.sendall(b'Hello, world')
            #print('recieving')
            data = s.recv(1024)
            action = pickle.loads(data)
            print(action[0:3])
            #print('Received', action )
            state, reward, done, info = environment.step(action)
            grip_img = gripper_camera(state['observation'])
            #time.sleep(0.01)
            if record:
                #print(time.time())

                actions.append(action)
                observations.append(np.array(state['observation']))
                step = step + 1
                if step % 100 == 0:

                    np.save(play_sequence+'/observations',np.array(observations))
                    np.save(play_sequence+'/actions', np.array(actions))

        except Exception as e:
            print(e)
            print("Connection Failed")

        




###################################################################################################
######################################
def make_dir(string):
    try:
        print('trying')
        os.makedirs(string)
        print('wtf')
    except FileExistsError:
        print('FileExistsError')
        
def make_new_play_data_folder():
    traj_count = 0
    make_dir('play_data')
    print('***********')
    for i in next(os.walk('play_data'))[1]:
        if 'set' in i:
            traj_count += 1 # count the number of previous trajectories
    dir_string = os.getcwd()+'/play_data/set_'+str(traj_count)
    make_dir(dir_string)
    return dir_string


#####################################################################################

def str_to_bool(string):
    if str(string).lower() == "true":
        string = True
    elif str(string).lower() == "false":
        string = False

    return string


def launch( arm, abs_rel, record):

    if record:
        play_sequence = make_new_play_data_folder() #do this because we don't want the data corrupted by a weird crossover between different trajectories. 
    else:
        play_sequence = None

    print(arm)
    # environment = ur5Env(renders=str_to_bool(render), arm = arm)
    environment = ur5Env_objects(renders=True)
    environment.reset()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
        except:
            print("conneciton refused")
        move_in_xyz(environment, arm, abs_rel,s, record, play_sequence)







@click.command()
@click.option('--abs_rel', type=str, default='abs',
              help='absolute or relative positioning, abs doesnt really work with rbx1 yet')
@click.option('--arm', type=str, default='ur5', help='rbx1 or kuka')
@click.option('--record', type=bool, default=True, help='recording')
def main(**kwargs):
    launch(**kwargs)


if __name__ == "__main__":
    main()