import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import math

import boardInput
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

from contour import *
from robot_control import *

#### Some configs (px) #####
TOLERANCE = 50
GRID_Y_OFFSET = -15    
ARM_BOARD_DISTANCE = 0.3
BODY_HEIGHT = 0.3
ROTATION_VELOCITY = 0.5
############################

def get_robot_coordinates(robot_state_client, roll):
    """Returns the current coordinates of Spot in the world frame."""
    robot_state = get_vision_tform_body(robot_state_client.get_robot_state().kinematic_state.transforms_snapshot)
    return {
        'x': robot_state.x,
        'y': robot_state.y,
        'z': robot_state.z,
        'yaw': robot_state.rot.to_yaw(),
        'roll': roll
    }

def calc_world_coordinate(image_responses, robot_state_client, x, y):
    """Returns the world coordinate of a pixel on camera frame and quaternion"""
    grid_depth = get_depth(image_responses, x, y)
    y = y + GRID_Y_OFFSET

    if grid_depth == None:
        return None

    cam_coords = pixel_to_camera_space(image_responses[1], x, y, depth=grid_depth)
    approach_dir = np.array(cam_coords) / np.linalg.norm(cam_coords)
    offset_cam_coords = cam_coords - approach_dir * ARM_BOARD_DISTANCE
    T_world_cam = get_a_tform_b(image_responses[1].shot.transforms_snapshot, "vision", "left_fisheye")
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
    """Returns the depth of a pixel in camera frame"""
    cv_depth = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint16)
    cv_depth = cv_depth.reshape(image_responses[0].shot.image.rows, image_responses[0].shot.image.cols)
    depth = cv_depth[y, x]/1000
    return None if depth == 0 else depth

def sort_board_grids(board_grids):
    """Map the grids into tictactoe board"""
    sorted_grids = {"00" : get_grid(board_grids, 0,0), "01" : get_grid(board_grids, 0,1), "02" : get_grid(board_grids, 0,2),
                    "10" : get_grid(board_grids, 1,0), "11" : get_grid(board_grids, 1,1), "12" : get_grid(board_grids, 1,2),
                    "20" : get_grid(board_grids, 2,0), "21" : get_grid(board_grids, 2,1), "22" : get_grid(board_grids, 2,2)}
    return sorted_grids

def get_world_grids(image_responses, robot_state_client, board_grids):
    """Returns the world coordinates of each grid of the game board"""
    sorted_grids = sort_board_grids(board_grids)

    world_points = {}
    for grid_key in sorted_grids:
        grid_px = compute_center(sorted_grids[grid_key])
        grid_x = grid_px[0]
        grid_y = grid_px[1]
        world_points[grid_key] = calc_world_coordinate(image_responses, robot_state_client, grid_x, grid_y)
    return world_points

def is_game_board_valid(world_grids):
    """Check all 9 grids in world coordinate"""
    return all(grid_point is not None for grid_point in world_grids.values())
        
def find_board(command_client, robot_state_client, image_client, timeout_time = 15):
    roll = 0.0
    rot_v = ROTATION_VELOCITY
    initial_coord = None
    world_grids = {}
    prev_count = 99

    start_time = time.time()
    current_time = time.time()

    while current_time - start_time < timeout_time:
        image_responses = image_client.get_image_from_sources(["left_depth_in_visual_frame", "left_fisheye_image"])
        gray_frame = cv2.imdecode(np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8), -1)
        bin_frame = convert_to_bin(gray_frame)
        img_res = (image_responses[1].shot.image.cols, image_responses[1].shot.image.rows)
        board_grids = get_board_grids(gray_frame)

        draw_board_centers(bin_frame, board_grids)
        cv2.imshow("Tictacspot", cv2.resize(bin_frame, (640, 480)))

        grid_count = len(board_grids)
        if grid_count == 9:
            grid_11 = compute_center(get_grid(board_grids, 1, 1))
            if is_x_aligned(grid_11, TOLERANCE, img_res[0]/2):
                if is_y_aligned(grid_11, TOLERANCE, img_res[1]/2):
                    print(f"[Detected grids: {grid_count}], Board is found and aligned!")
                    initial_coord = get_robot_coordinates(robot_state_client, roll)
                    world_grids = get_world_grids(image_responses, robot_state_client, board_grids)
                    if is_game_board_valid(world_grids):
                        return world_grids, initial_coord
                    else:
                        print("Failed to validate game board!")
                        break
                else:
                    print(f"[Detected grids: {grid_count}], Board found: aligning...")
                    roll = roll + 0.1
                    robot_adjust_roll(command_client, roll, BODY_HEIGHT)
            else:
                print(f"[Detected grids: {grid_count}], Board found: aligning...")
                # if prev_count > grid_count:
                #     rot_v = -rot_v
                #     print("Turn back!")
                # prev_count = grid_count
                robot_velocity_move(command_client, rot = rot_v)

        elif grid_count >= 3 and grid_count < 9:
            print(f"[Detected grids: {grid_count}], Board is partially found!")
            grid_01 = compute_center(get_grid(board_grids, 0, 1))
            if is_x_aligned(grid_01, TOLERANCE, img_res[0]/2):
                roll = roll + 0.1
                robot_adjust_roll(command_client, roll, BODY_HEIGHT)
            else:
                print(f"[Detected grids: {grid_count}], Partial found: aligning...")
                # if prev_count > grid_count:
                #     rot_v = -rot_v
                #     print("Turn back!")
                # prev_count = grid_count
                robot_velocity_move(command_client, rot = rot_v)
        else:
            print("No board is found, try rotate")
            # if prev_count > grid_count:
            #     rot_v = -rot_v
            #     print("Turn back!")
            # prev_count = grid_count
            robot_velocity_move(command_client, rot = rot_v)
        current_time = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return None, None

def check_board_state(command_client, image_client, initial_coord):
    robot_trajectory_move(command_client, initial_coord['x'], initial_coord['y'], initial_coord['yaw'])
    robot_adjust_roll(command_client, initial_coord['roll'], BODY_HEIGHT)
    image_responses = image_client.get_image_from_sources(["left_fisheye_image"])
    frame = cv2.imdecode(np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint8), -1)
    board_grids = sort_board_grids(get_board_grids(frame))
    circles = find_circles(frame)
    for circle in circles:
        for grid in board_grids:
            if is_px_inside_contour(board_grids[grid], circle[0], circle[1]):
                print("O is in grid " + str(grid))

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
    
    # main
    assert not robot.is_estopped()
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)


    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.logger.info('Powering on robot... This may take several seconds.')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')

        image_client = robot.ensure_client(ImageClient.default_service_name)
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)

        robot_stand(command_client, BODY_HEIGHT)
        world_grids, initial_coord = find_board(command_client, robot_state_client, image_client)

        if (world_grids, initial_coord) == (None, None):
            robot.power_off(cut_immediately=False, timeout_sec=20)
            assert not robot.is_powered_on(), 'Robot power off failed.'
            robot.logger.info('Robot safely powered off.')
            return

        check_board_state(command_client, image_client, initial_coord)
                            

        robot_arm_stow(command_client)
        robot_trajectory_move(command_client, initial_coord['x'], initial_coord['y'], initial_coord['yaw'])
        robot_stand(command_client)

        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), 'Robot power off failed.'
        robot.logger.info('Robot safely powered off.')


    # ### for testing purposes
    # image_client = robot.ensure_client(ImageClient.default_service_name)
    # test_list = []
    # while True:
    #     image_responses = image_client.get_image_from_sources(["left_depth_in_visual_frame", "left_fisheye_image"])

    #     cv_visual = cv2.imdecode(np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8), -1)
    #     find_circles(cv_visual)
    #     cv2.imshow("Tictacspot", cv_visual)
    
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()