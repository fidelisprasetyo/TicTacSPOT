import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import math

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.api import trajectory_pb2, geometry_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b, get_vision_tform_body
from bosdyn.client.math_helpers import Quat
from bosdyn.client.image import ImageClient, pixel_to_camera_space
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand, block_until_arm_arrives
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.util import seconds_to_duration
from bosdyn.api import image_pb2

from contour import *

from PIL import Image
import io

#### Some configs (px) #####
TOLERANCE = 100
TARGET_AREA = 2500
############################


def robot_move(command_client, _time = 0.5, x = 0.0, y = 0.0, rot = 0.0):
    cmd = RobotCommandBuilder.synchro_velocity_command(v_x = x, v_y = y, v_rot = rot)
    command_client.robot_command(command = cmd, end_time_secs = time.time() + _time)
    time.sleep(1)

def robot_roll(command_client, _roll):
    footprint_R_body = bosdyn.geometry.EulerZXY(yaw=0.0, roll=_roll, pitch=0.0)
    cmd = RobotCommandBuilder.synchro_stand_command(footprint_R_body=footprint_R_body)
    command_client.robot_command(command = cmd)
    time.sleep(1)


def get_world_coordinates(grid, image_responses, robot_state_client):

    grid_x = grid[0][0]
    grid_y = grid[0][1]
    grid_depth = get_depth(image_responses, grid_x, grid_y)

    if grid_depth == 0:
        print("we got zero depth!!")
        return None

    cam_coords = pixel_to_camera_space(image_responses[1], grid_x, grid_y, depth=grid_depth)
    approach_dir = np.array(cam_coords) / np.linalg.norm(cam_coords)
    offset_cam_coords = cam_coords - approach_dir * 0.30
    T_world_cam = get_a_tform_b(image_responses[1].shot.transforms_snapshot, "odom", "left_fisheye")
    world_point = T_world_cam.transform_point(*offset_cam_coords)

    robot_state = robot_state_client.get_robot_state()
    robot_rt_world = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)

    target_vector = np.array([world_point[0] - robot_rt_world.x, world_point[1] - robot_rt_world.y,0])
    target_direction = target_vector / np.linalg.norm(target_vector)
    zhat = [0.0, 0.0, 1.0]
    yhat = np.cross(zhat, target_direction)
    mat = np.array([target_direction, yhat, zhat]).T
    angle_desired = Quat.from_matrix(mat).to_yaw()

    return (world_point, angle_desired)

def get_depth(image_responses, x, y):
    cv_depth = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint16)
    cv_depth = cv_depth.reshape(image_responses[0].shot.image.rows, image_responses[0].shot.image.cols)

    depth = cv_depth[y, x]/1000

    return depth

def main():
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-j', '--jpeg-quality-percent', help='JPEG quality percentage (0-100)',
                        type=int, default=50)
    options = parser.parse_args()

    sdk = bosdyn.client.create_standard_sdk('TicTacSPOT')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    
    ### main

    assert not robot.is_estopped()
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)


    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.logger.info('Powering on robot... This may take several seconds.')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')

        image_client = robot.ensure_client(ImageClient.default_service_name)

        # stand up!
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client)
        time.sleep(1)

        roll = 0.0
        board_found = False
        while True:
            image_responses = image_client.get_image_from_sources(["left_depth_in_visual_frame", "left_fisheye_image"])

            cv_visual = cv2.imdecode(np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8), -1)

            img_res = (image_responses[1].shot.image.cols, image_responses[1].shot.image.rows)
            gray_frame = cv_visual
            board_grids = get_board_grids(gray_frame)

            draw_board_centers(gray_frame, board_grids)
            cv2.imshow("Tictacspot", cv2.resize(gray_frame, (640, 480)))

            grid_count = len(board_grids)
            if not board_found:
                if grid_count == 9:
                    grid_11 = get_grid(board_grids, 1, 1)
                    if is_x_aligned(grid_11, TOLERANCE, img_res[0]/2):
                        if is_y_aligned(grid_11, TOLERANCE, img_res[1]/2 - 40):
                            print(f"[Detected grids: {grid_count}], Board is found and aligned!")
                            board_found = True
                        else:
                            print(f"[Detected grids: {grid_count}], Board found: aligning...")
                            if roll == 0.4:
                                roll = 0.0
                            else:
                                roll = roll + 0.1
                            robot_roll(command_client, roll)
                    else:
                        print(f"[Detected grids: {grid_count}], Board found: aligning...")
                        robot_move(command_client, rot = 0.5)
                elif grid_count >= 3 and grid_count < 9:
                    print(f"[Detected grids: {grid_count}], Board is partially found!")
                    grid_01 = get_grid(board_grids, 0, 1)

                    if is_x_aligned(grid_01, TOLERANCE, img_res[0]/2):
                        if roll == 0.4:
                            roll = 0.0
                        else:    
                            roll = roll + 0.1
                        robot_roll(command_client, roll)
                    else:
                        print(f"[Detected grids: {grid_count}], Partial found: aligning...")
                        robot_move(command_client, rot = 0.5)
                else:
                    print("No board is found, try rotate")
                    robot_move(command_client, rot = 0.5)

            
            if board_found:
                grids = [get_grid(board_grids, 0,0), get_grid(board_grids, 0,1), get_grid(board_grids, 0,2)]
                world_points = []
                for grid in grids:
                    world_points.append(get_world_coordinates(grid, image_responses, robot_state_client))

                #blocking_stand(command_client)
                for world_point in world_points:
                    # gaze_command = RobotCommandBuilder.arm_gaze_command(x = world_point[0][0], y = world_point[0][1], z = world_point[0][2], frame_name="odom")
                    arm_command = RobotCommandBuilder.arm_pose_command(
                        frame_name="odom",
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

                # grid_01_area = get_grid(board_grids, 0, 1)[1]
                # area_diff = TARGET_AREA - grid_01_area
                # if area_diff > 300:
                #     print(f"Area difference: {area_diff}, moving closer...")
                #     robot_move(command_client, y=-0.2)
                # elif area_diff < -300:
                #     robot_move(command_client, y=0.2)
                #     print(f"Area difference: {area_diff}, moving farther...")
                # else:
                #     if grid_count == 9:
                #         print("PERFECT")
                #     else:
                #         board_found = False

            #Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    

    # ### for testing purposes
    # image_client = robot.ensure_client(ImageClient.default_service_name)
    # test_list = []
    # while True:
    #     image_responses = image_client.get_image_from_sources(["left_depth_in_visual_frame", "left_fisheye_image"])

    #     cv_depth = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint16)
    #     cv_depth = cv_depth.reshape(image_responses[0].shot.image.rows, image_responses[0].shot.image.cols)
    #     cv_visual = cv2.imdecode(np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8), -1)
    #     visual_rgb = cv_visual if len(cv_visual.shape) == 3 else cv2.cvtColor(
    #     cv_visual, cv2.COLOR_GRAY2RGB)
    #     min_val = np.min(cv_depth)
    #     max_val = np.max(cv_depth)
    #     depth_range = max_val - min_val
    #     depth8 = (255.0 / depth_range * (cv_depth - min_val)).astype('uint8')
    #     depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
    #     depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)
    #     out = cv2.addWeighted(visual_rgb, 0.5, depth_color, 0.5, 0)

    #     # board_girds = get_board_grids(cv_visual)
    #     # grid_11 = get_grid(board_girds, 1, 1)

    #     # depth_mm = cv_depth[grid_11[0][1], grid_11[0][0]]
    #     # depth_m = depth_mm / 1000.0


    #     # draw_board_centers(cv_visual, board_girds)
    #     cv2.imshow("Tictacspot", out)
    
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()