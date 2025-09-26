import time
import cv2
import bosdyn.geometry
import numpy as np
from bosdyn.api import arm_command_pb2, robot_command_pb2, geometry_pb2, trajectory_pb2
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, get_a_tform_b
from bosdyn.client.image import ImageClient, pixel_to_camera_space
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand, block_until_arm_arrives
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.math_helpers import Quat
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2

from contour import *
from fetch_only_pickup import PICK_UP_STATE_SUCCESS, PICK_UP_STATE_FAIL, PICK_UP_STATE_NOT_FOUND
import fetch_only_pickup as fetch

ARM_BOARD_OFFSET_DISTANCE = 0.25
TOLERANCE = 50
BODY_HEIGHT = 0.3
FORCE_THRESHOLD = 15
Y_BOTTOM_OFFSET = 10
Y_MIDDLE_OFFSET = 12
Y_TOP_OFFSET = 15

class TicTacSpot:

    def __init__(self, robot, options):
        self.robot = robot
        self.options = options
        self.image_client = robot.ensure_client(ImageClient.default_service_name)
        self.command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self.robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

        self.initial_coord = {}
        self.best_view_roll = None
        self.average_grid_area = None
        self.board_world_coord = None
        self.virtual_board = None
        self.board_outline = None
        self.visual_frame = None

        _image_responses = self.image_client.get_image_from_sources(["left_fisheye_image"])
        self.cx, self.cy = (_image_responses[0].shot.image.cols/2, _image_responses[0].shot.image.rows/2)


    ### Tictacspot Main Methods
    
    def find_board(self, timeout_time = 30):
        """Finds the tic-tac-toe game board."""
        start_time = time.time()
        current_time = time.time()
        is_board_found = False
        prev_grid_count = 0
        rot_v = 0.5
        roll = 0.0

        print("Finding tic-tac-toe game board...")
        while current_time - start_time < timeout_time:
            image_responses = self.image_client.get_image_from_sources(["left_depth_in_visual_frame", "left_fisheye_image"])
            gray_frame = cv2.imdecode(np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8), -1)
            bin_frame = convert_to_bin(gray_frame)
            board_grids, board_outline = get_board_grids(gray_frame)
            
            ### --- for visualization & debugging ----
            visual_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            draw_board_centers(visual_frame, board_grids)
            draw_board(visual_frame, board_outline, color = (0,0,255))
            self.visual_frame = visual_frame
            cv2.imshow('Tictacspot', visual_frame)
            cv2.imshow('Tictacspot_bin', bin_frame)
            cv2.waitKey(1)
            ### --------------------------------------

            grid_count = len(board_grids)
            if grid_count == 9:
                grid_11 = get_grid(board_grids, 1, 1)   # middle grid pixel coords
                _, (x, y) = grid_11
                if is_x_aligned(x, self.cx, TOLERANCE):
                    if is_y_aligned(y, self.cy, TOLERANCE):
                        print(f"[Detected grids: {grid_count}], Board is found and aligned!")

                        self.initial_coord = self.get_robot_coordinates()
                        self.best_view_roll = roll
                        self.board_outline = board_outline
                        self.average_grid_area = self._grid_avg_area(board_grids)
                        self.virtual_board = self._sort_board_grids(board_grids)
                        self.board_world_coord = self._get_board_world_coords(image_responses, self.virtual_board, ARM_BOARD_OFFSET_DISTANCE)
                        
                        occupancy_grid = self.get_board_occupancy()
                        if not self._is_board_empty(occupancy_grid):
                            print("Board is not empty, clear the board first!")
                            continue

                        if self.board_world_coord is not None and self.board_outline is not None:
                            is_board_found = True
                            self.stand()
                            print("Board is found and is mapped successfully. Game start!")
                        break
                    else:
                        print(f"[Detected grids: {grid_count}], Board is found: aligning the board vertically...")
                        roll += 0.1
                        self.adjust_roll(roll, duration = .5, body_height = BODY_HEIGHT)
                else:
                    print(f"[Detected grids: {grid_count}], Board is found: aligning the board horizontally")
                    self.velocity_move(duration = .5, rot = .5)
                    time.sleep(1)

            elif grid_count >= 3 and grid_count < 9:
                _, (x, y) = get_grid(board_grids, 2,1)
                if is_x_aligned(x, self.cx, TOLERANCE):
                    print(f"[Detected grids: {grid_count}], Board is partially found: aligning vertically...")
                    roll += 0.1
                    self.adjust_roll(roll, duration = .5, body_height = BODY_HEIGHT)
                else:
                    print(f"[Detected grids: {grid_count}], Board is partically found: aligning horizontally")
                    if prev_grid_count > grid_count:
                       rot_v = -rot_v
                       print(f"[Detected grids: {grid_count}], Rotate back!")
                    self.velocity_move(duration=.5, rot=rot_v)
                    time.sleep(1)
            else:
                print("No board is found, try rotate")
                self.roll = 0.0
                self.velocity_move(duration = 1, rot = 1)
                time.sleep(1)

            prev_grid_count = grid_count
            current_time = time.time()

        if not is_board_found:
            print("Failed to find the board, powering off...")
            self.power_off()

    def pick_up(self, timeout_time = 60):
        """Picks up the X-pieces"""
        self.stand()
        start_time = time.time()
        current_time = time.time()
        temp_coord = None
        while current_time - start_time < timeout_time:
            pick_up_status = fetch.pick_up(self.options, self.robot)
            if pick_up_status == PICK_UP_STATE_SUCCESS:
                return
            elif pick_up_status == PICK_UP_STATE_FAIL:
                if temp_coord is None:
                    self.go_to_initial()
                else:
                    self.trajectory_move(temp_coord['x'], temp_coord['y'], temp_coord['yaw'])
            elif pick_up_status == PICK_UP_STATE_NOT_FOUND:
                self.velocity_move(duration = 0.5, rot = 1)
                temp_coord = self.get_robot_coordinates()
            current_time = time.time()

        print("Timed out. Failed to find/ pick up X piece. Spot cannot continue to play the game.")
        self.power_off()
        
    def place(self, row, col):
        self.trajectory_move(self.initial_coord['x'], self.initial_coord['y'], self.initial_coord['yaw'] + np.pi/2)
        time.sleep(.5)
        self.arm_pose(self.board_world_coord[row][col])
        time.sleep(1)

        while self._is_pushing_under_threshold(FORCE_THRESHOLD):
            self.arm_push(duration = .25, v_r= .3)
            time.sleep(.25)

        print("Touched the board! Release the grip")
        self.gripper_open()
        self.arm_push(1, -1)
        time.sleep(1)

        self.arm_stow()
        self.gripper_close()
        time.sleep(.5)

    def get_board_occupancy(self):
        """Returns a 2d matrix of occupied grids. Also re-aligns the virtual board position if Spot shifts from the initial position."""
        
        virtual_board = []
        cur_board_outline = None

        self.go_to_initial()
        self.adjust_roll(self.best_view_roll, duration = 1, body_height=BODY_HEIGHT)

        
        while cur_board_outline is None:
            image_response = self.image_client.get_image_from_sources(["left_fisheye_image"])
            gray_frame = cv2.imdecode(np.frombuffer(image_response[0].shot.image.data, dtype=np.uint8), -1)
            bin_frame = convert_to_bin(gray_frame)
            cur_board_outline = get_board_outline(gray_frame, approx_area = cv2.contourArea(self.board_outline))

            ### --- for visualization & debugging ----
            visual_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            draw_board(visual_frame, self.board_outline, color = (0, 0, 255))
            cv2.imshow('Tictacspot', visual_frame)
            cv2.imshow('Tictacspot_bin', bin_frame)
            cv2.waitKey(1)
            ### --------------------------------------
        
            
        prev_outline = np.array(self.board_outline, dtype=np.float32)
        cur_outline = np.array(cur_board_outline, dtype=np.float32)
        H, _ = cv2.findHomography(prev_outline, cur_outline)

        for row_idx, row in enumerate(self.virtual_board):
            for col_idx, grid in enumerate(row):
                contour, _ = grid
                contour = np.array(contour, dtype=np.float32).reshape(-1, 1, 2)
                contour = cv2.perspectiveTransform(contour, H)
                contour = contour.astype(np.int32)
                virtual_board.append(contour)  ## unsorted, it works but ocassionally the transformation will mess up the ordering

                ### --- for visualization & debugging ----
                draw_board(visual_frame, contour, color= (255,0,0))
                ### --------------------------------------

        virtual_board = self._sort_board_grids(virtual_board) ## re-sort the transformed contours

        occupancy_grid = np.ones((3,3))
        blobs = detect_blobs(gray_frame, self.average_grid_area)
        for row_idx, row in enumerate(virtual_board):
            for col_idx, grid in enumerate(row):
                contour, _ = grid
                for blob in blobs:
                    (x,y) = blob.pt
                    if is_px_inside_contour(contour, x, y):
                        occupancy_grid[row_idx][col_idx] = 0
                        break
        
        ### --- for visualization & debugging ----
        draw_board(visual_frame, cur_board_outline, color = (255, 0, 0))
        draw_board(visual_frame, self.board_outline, color = (0, 0, 255))
        visual_frame = cv2.drawKeypoints(visual_frame, blobs, np.array([]), (255,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('Tictacspot', visual_frame)
        cv2.imshow('Tictacspot_bin', bin_frame)
        cv2.waitKey(1)
        self.visual_frame = visual_frame
        ### --------------------------------------

        print("Occupancy grid:\n", occupancy_grid)
        return occupancy_grid


    ### Pixel processing methods

    def _calc_world_coordinate(self, image_responses, x, y, depth_val, offset_distance, y_offset):
        """Returns the world coordinate of a pixel on camera frame and quaternion"""

        y -= y_offset
        cam_coords = pixel_to_camera_space(image_responses[1], x, y, depth=depth_val)
        approach_dir = np.array(cam_coords) / np.linalg.norm(cam_coords)
        offset_cam_coords = cam_coords - approach_dir * offset_distance
        T_world_cam = get_a_tform_b(image_responses[1].shot.transforms_snapshot, "vision", "left_fisheye")
        world_point = T_world_cam.transform_point(*offset_cam_coords)

        robot_state = self.robot_state_client.get_robot_state()
        robot_rt_world = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)

        target_vector = np.array([world_point[0] - robot_rt_world.x, world_point[1] - robot_rt_world.y,0])
        target_direction = target_vector / np.linalg.norm(target_vector)
        zhat = [0.0, 0.0, 1.0]
        yhat = np.cross(zhat, target_direction)
        mat = np.array([target_direction, yhat, zhat]).T
        angle_desired = Quat.from_matrix(mat).to_yaw()

        return (world_point, angle_desired)
        
    def _get_board_world_coords(self, image_responses, board_matrix, offset_distance):
        """Returns the world coordinates of each grid of the game board"""
        world_points = [[None for _ in range(3)] for _ in range(3)]
        depths = np.zeros((3,3))

        for row_idx, row in enumerate(board_matrix):
            for col_idx, grid in enumerate(row):
                _, (x,y) = grid
                depths[row_idx][col_idx] = self._get_depth(image_responses, x, y)

        depths = self._fill_missing_depth(depths)
        if depths is None:
            raise RuntimeError("All depth values in a row are missing — cannot proceed with world coordinate calculation.")
        
        for row_idx, row in enumerate(board_matrix):
            for col_idx, grid in enumerate(row):
                _, (x,y) = grid
                y_offset = 0
                if row_idx == 2:
                    y_offset = Y_BOTTOM_OFFSET
                elif row_idx == 1:
                    y_offset = Y_MIDDLE_OFFSET
                else:
                    y_offset = Y_TOP_OFFSET

                world_points[row_idx][col_idx] = self._calc_world_coordinate(image_responses, x, y, depths[row_idx][col_idx], offset_distance, y_offset)

        return world_points        
    
    def _fill_missing_depth(self, depths):
        """Sometimes the camera fails to get depth value of the outer grid, handle by assigning the missing depth with the neighboring grid's depth"""
        for row in range(3):
            row_values = depths[row]

            if all(v is None for v in row_values):
                return None

            fallback = next((v for v in row_values if v is not None), None)

            for col in range(3):
                if depths[row][col] is None:
                    depths[row][col] = fallback
                    print(f'{row}, {col} depth is missing, accuracy might be impacted!')

        return depths

    def _get_depth(self, image_responses, x, y):
        """Returns the depth of a pixel in camera frame"""
        cv_depth = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint16)
        cv_depth = cv_depth.reshape(image_responses[0].shot.image.rows, image_responses[0].shot.image.cols)
        depth = cv_depth[y, x]/1000
        return None if depth == 0 else depth

    def _sort_board_grids(self, board_grids):
        """Map the grids into tictactoe board 2D Matrix"""
        sorted_grids = [[get_grid(board_grids, 0,0), get_grid(board_grids, 0,1), get_grid(board_grids, 0,2)], 
                        [get_grid(board_grids, 1,0), get_grid(board_grids, 1,1), get_grid(board_grids, 1,2)], 
                        [get_grid(board_grids, 2,0), get_grid(board_grids, 2,1), get_grid(board_grids, 2,2)]]
        
        return sorted_grids
    

    ### SPOT General Commands

    def power_on(self):
        print('Powering on robot... This may take several seconds.')
        self.robot.power_on(timeout_sec=20)
        assert self.robot.is_powered_on(), 'Robot power on failed.'
        print('Robot powered on.')

    def power_off(self):
        self.robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not self.robot.is_powered_on(), 'Robot power off failed.'
        print('Robot safely powered off.')

    ### SPOT Movement Commands

    def stand(self, body_height = 0.0):
        stand_command = RobotCommandBuilder.synchro_stand_command(body_height = body_height)
        cmd_id = self.command_client.robot_command(stand_command)
        blocking_stand(self.command_client, cmd_id)

    def velocity_move(self, duration, x = 0.0, y = 0.0, rot = 0.0):
        cmd = RobotCommandBuilder.synchro_velocity_command(v_x = x, v_y = y, v_rot = rot)
        self.command_client.robot_command(command = cmd, end_time_secs = time.time() + duration)
        time.sleep(duration)
    
    def adjust_roll(self, roll, duration, body_height = 0.0):
        footprint_R_body = bosdyn.geometry.EulerZXY(yaw=0.0, roll=roll, pitch=0.0)
        cmd = RobotCommandBuilder.synchro_stand_command(footprint_R_body=footprint_R_body, body_height=body_height)
        self.command_client.robot_command(command = cmd)
        time.sleep(duration)
    
    def trajectory_move(self, x, y, yaw, frame_name = VISION_FRAME_NAME, end_time = 3.0):
        start_time = time.time()
        current_time = time.time()
        while not self._is_at_target(x, y , yaw) and (current_time - start_time < end_time):
            cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x = x, 
            goal_y = y, 
            goal_heading = yaw, 
            frame_name = frame_name,
            body_height = 0.0, 
            params = self._set_mobility_params(),
            locomotion_hint = spot_command_pb2.HINT_AUTO)

            current_time = time.time()
            self.command_client.robot_command(command = cmd, end_time_secs = time.time() + end_time)
    
    def arm_stow(self, timeout_sec = 1):
        stow = RobotCommandBuilder.arm_stow_command()
        stow_command_id = self.command_client.robot_command(stow)
        block_until_arm_arrives(self.command_client, stow_command_id, timeout_sec)
    
    def arm_pose(self, world_point, frame_name = VISION_FRAME_NAME, timeout_sec = 10):
        arm_command = RobotCommandBuilder.arm_pose_command(
            frame_name = frame_name,
            x = world_point[0][0],
            y = world_point[0][1],
            z = world_point[0][2],
            qw = np.cos(world_point[1] / 2),
            qx = 0.0,
            qy = 0.0,
            qz = np.sin(world_point[1] / 2),
            seconds=2
        )
        follow_arm_command = RobotCommandBuilder.follow_arm_command()
        synchro_command = RobotCommandBuilder.build_synchro_command(arm_command, follow_arm_command)
        command_id = self.command_client.robot_command(synchro_command)
        block_until_arm_arrives(self.command_client, command_id, timeout_sec)
        self.command_client.robot_command(RobotCommandBuilder.stop_command())

    def arm_push(self, duration, v_r):
        
        cylindrical_velocity = arm_command_pb2.ArmVelocityCommand.CylindricalVelocity()
        cylindrical_velocity.linear_velocity.r = v_r
        cylindrical_velocity.linear_velocity.theta = 0
        cylindrical_velocity.linear_velocity.z = 0

        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cylindrical_velocity=cylindrical_velocity,
            end_time=self.robot.time_sync.robot_timestamp_from_local_secs(time.time() + duration))

        robot_command = robot_command_pb2.RobotCommand()
        robot_command.synchronized_command.arm_command.arm_velocity_command.CopyFrom(arm_velocity_command)

        self.command_client.robot_command(command=robot_command, end_time_secs=time.time() + duration)

    def gripper_open(self, timeout_sec = 1):
        print("Open gripper")
        gripper_command = RobotCommandBuilder.claw_gripper_open_command()
        command_id = self.command_client.robot_command(gripper_command)
        block_until_arm_arrives(self.command_client, command_id, timeout_sec)
    
    def gripper_close(self, timeout_sec = 1):
        gripper_command = RobotCommandBuilder.claw_gripper_close_command()
        command_id = self.command_client.robot_command(gripper_command)
        block_until_arm_arrives(self.command_client, command_id, timeout_sec)
    
    def go_to_initial(self):
        if self.initial_coord:
            self.trajectory_move(self.initial_coord['x'], self.initial_coord['y'], self.initial_coord['yaw'])     

    ### Other

    def _grid_avg_area(self, board_grid):
        sum_area = 0.0
        for grid in board_grid:
            area = cv2.contourArea(grid)
            sum_area += area
        return sum_area/len(board_grid)

    def _is_pushing_under_threshold(self, force_threshold):
        robot_state = self.robot_state_client.get_robot_state()
        force = robot_state.manipulator_state.estimated_end_effector_force_in_hand
        force_x = abs(force.x)
        print(f'Current applied force: {force_x:.2f}', end='\r')
        return force_x < force_threshold   

    def _is_at_target(self, x, y, yaw, epsilon = 0.1):
        current_state = get_vision_tform_body(self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot)
        current_angle = current_state.rot.to_yaw()
        return (abs(current_state.x - x) < epsilon and
                abs(current_state.y - y) < epsilon and
                abs(current_angle - yaw) < 0.025) 

    def _is_board_empty(self, occupancy_grid):
        for row_idx, row in enumerate(occupancy_grid):
            for col_idx, grid_occupance in enumerate(row):
                if grid_occupance == 1:
                    return False
        return True

    def save_img_log(self):
        cv2.imwrite('./log/err_log.jpg', self.visual_frame)

    ### Setters & Getters

    def _set_mobility_params(self):
        """Set robot mobility params to disable obstacle avoidance."""
        obstacles = spot_command_pb2.ObstacleParams(disable_vision_body_obstacle_avoidance=True,
                                                    disable_vision_foot_obstacle_avoidance=True,
                                                    disable_vision_foot_constraint_avoidance=True,
                                                    obstacle_avoidance_padding=.001)
        # Default body control settings
        body_control = self._set_default_body_control()
        speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(
            linear=Vec2(x=0.5, y=0.5), angular=1))

        mobility_params = spot_command_pb2.MobilityParams(
            obstacle_params=obstacles, vel_limit=speed_limit, body_control=body_control,
            locomotion_hint=spot_command_pb2.HINT_AUTO)

        return mobility_params
    
    def _set_default_body_control(self):
        """Set default body control params to current body position."""
        footprint_R_body = bosdyn.geometry.EulerZXY()
        position = geometry_pb2.Vec3(x=0.0, y=0.0, z=0.0)
        rotation = footprint_R_body.to_quaternion()
        pose = geometry_pb2.SE3Pose(position=position, rotation=rotation)
        point = trajectory_pb2.SE3TrajectoryPoint(pose=pose)
        traj = trajectory_pb2.SE3Trajectory(points=[point])
        return spot_command_pb2.BodyControlParams(base_offset_rt_footprint=traj)

    def get_robot_coordinates(self):
        coords = {}
        robot_state = get_vision_tform_body(self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot)
        coords['x'] = robot_state.x
        coords['y'] = robot_state.y
        coords['z'] = robot_state.z
        coords['yaw'] = robot_state.rot.to_yaw()
        return coords

    def get_image_client(self):
        return self.image_client
    
    def get_command_client(self):
        return self.command_client
    
    def get_robot_state_client(self):
        return self.robot_state_client
