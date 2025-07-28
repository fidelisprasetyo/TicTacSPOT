#### Encapsulates all robot functions

import time
import bosdyn.geometry
import numpy as np
from bosdyn.client.robot_command import RobotCommandBuilder, blocking_stand, block_until_arm_arrives
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2

def robot_power_off(robot):
    robot.power_off(cut_immediately=False, timeout_sec=20)
    assert not robot.is_powered_on(), 'Robot power off failed.'
    robot.logger.info('Robot safely powered off.')

def robot_stand(command_client, _body_height = 0.0):
    stand_command = RobotCommandBuilder.synchro_stand_command(body_height = _body_height)
    cmd_id = command_client.robot_command(stand_command)
    blocking_stand(command_client, cmd_id)
    time.sleep(1)


def robot_velocity_move(command_client, duration = 0.5, x = 0.0, y = 0.0, rot = 0.0):
    cmd = RobotCommandBuilder.synchro_velocity_command(v_x = x, v_y = y, v_rot = rot)
    command_client.robot_command(command = cmd, end_time_secs = time.time() + duration)
    time.sleep(1)
    
def robot_adjust_roll(command_client, _roll, _body_height = 0.0):
    footprint_R_body = bosdyn.geometry.EulerZXY(yaw=0.0, roll=_roll, pitch=0.0)
    cmd = RobotCommandBuilder.synchro_stand_command(footprint_R_body=footprint_R_body, body_height=_body_height)
    command_client.robot_command(command = cmd)
    time.sleep(1)

def robot_trajectory_move(command_client, x, y, yaw, frame_name = 'vision', timeout_sec = 5):
    cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x = x, 
        goal_y = y, 
        goal_heading = yaw, 
        frame_name = frame_name,
        body_height = 0.0, 
        locomotion_hint = spot_command_pb2.HINT_AUTO)
                
    command_client.robot_command(command = cmd, end_time_secs = time.time() + timeout_sec)
    time.sleep(timeout_sec)

def robot_arm_stow(command_client, timeout_sec = 3.0):
    stow = RobotCommandBuilder.arm_stow_command()
    stow_command_id = command_client.robot_command(stow)
    block_until_arm_arrives(command_client, stow_command_id, timeout_sec)

def robot_arm_pose(command_client, world_point, frame_name = 'vision'):
    arm_command = RobotCommandBuilder.arm_pose_command(
        frame_name = frame_name,
        x = world_point[0][0],
        y = world_point[0][1],
        z = world_point[0][2],
        qw = np.cos(world_point[1] / 2),
        qx = 0.0,
        qy = 0.0,
        qz = np.sin(world_point[1] / 2),
        seconds=2.0
    )
    follow_arm_command = RobotCommandBuilder.follow_arm_command()
    # gripper_command = RobotCommandBuilder.claw_gripper_open_command()
    synchro_command = RobotCommandBuilder.build_synchro_command(arm_command, follow_arm_command)
    command_client.robot_command(synchro_command)
    time.sleep(10)