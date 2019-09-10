import os, inspect
import time
import traceback
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

import random
import pybullet_data

import sys
import imageio
import time
import socket
import pickle

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)


#VR Event Indices for interacting with the controller
CONTROLLER = 0
POSITION = 1
ORIENTATION = 2
ANALOG=3
BUTTONS=6
BUTTON_2 = (6,2) # indices



def basic_vr_env(s):
    last_button_2_push = time.time()
    gravity = -10
    cid = p.connect(p.SHARED_MEMORY)
    print(cid)
    print("Connected to shared memory")
    if (cid<0):
        p.connect(p.GUI)
    p.resetSimulation()
    
    #p.setGravity(0,0,-0.01) 

    p.setVRCameraState([0,0,-1.0],p.getQuaternionFromEuler([0,0,0]))
    
    p.resetDebugVisualizerCamera(0.75,45,-45,[0,0,0])
    past_ori = np.array([None])
    print('VR Setup Done')

    s.bind((HOST, PORT))
    s.listen()
    print('Ready to accept connection')
    conn, addr = s.accept()
    xyz_offset = np.array([0,0,0])
    with conn:
        print('Connected by', addr)
        while (1):
           
           


           events = p.getVREvents()

           try:

            e = events[0]
            
            ori = np.array(list(e[ORIENTATION]))
            # check if flipped!
            if past_ori.all() != None:
                if (np.sign(ori) == -np.sign(past_ori)).all():
                    
                    ori = -ori
                    #print('triggered!', ori)
            past_ori = ori

            if e[BUTTON_2[0]][BUTTON_2[1]] >=1 and (time.time() >= last_button_2_push+1):
                print("offsetting")
                last_button_2_push = time.time()
                xyz_offset = np.array(e[POSITION])
        
            pos = np.array(list(e[POSITION]))-xyz_offset
            action = np.array(list(pos) + list(ori) + [e[ANALOG]]) 
            print(action)
            #action = np.array(list(e[POSITION]) + list(e[ORIENTATION]) + [e[ANALOG]]) 
            #print(action)
            
            #print(time.time())
            #if the button is pressed and its been a second since the last recording
            

            #     pass
                # another button to work with if we want

            data = conn.recv(1024)
            
            if not data:
              print('breaking')
              break
            action = pickle.dumps(action)
            
            conn.sendall(action)
            
           except Exception as e:
            pass
            print(e)
            # print(traceback.format_exc())
        p.disconnect()


def launch(arm, record):
    # This will be a server that sends the latest commanded position whenever the client is ready to recieve it.
# This means we can run a really lightweight VR setup, just the controller, sending commands to a separate simulation.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        basic_vr_env(s)
        
            
            
                

    

@click.command()
@click.option('--arm', type=str, default='ur5', help='rbx1 or kuka or ur5')
@click.option('--record', type=bool, default=False, help='True or False')

def main(**kwargs):
    launch(**kwargs)

if __name__ == "__main__":
    main()

