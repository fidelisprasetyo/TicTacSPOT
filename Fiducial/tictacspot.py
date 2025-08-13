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

from contour_detection import *
import fetch_only_pickup as fetch

ARM_BOARD_GAZE_OFFSET = 0.15
TOLERANCE = 30
BODY_HEIGHT = 0.3
FORCE_THRESHOLD = 20
Y_BOTTOM_OFFSET = 8
Y_MIDDLE_OFFSET = 10
Y_TOP_OFFSET = 10

class TicTacSpot:

    def __init__(self, robot, options):
        self.robot = robot
        self.options = options
        self.image_client = robot.ensure_client(ImageClient.default_service_name)
        self.command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self.robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

        self.initial_coord = {}
        self.best_view_roll = 0.0
        self.board_gaze_coord = {}

        _image_responses = self.image_client.get_image_from_sources(["left_fisheye_image"])
        self._width, self._height = (_image_responses[0].shot.image.cols, _image_responses[0].shot.image.rows)


    ### Tictacspot Main Methods
    
    def find_board(self, timeout_time = 30):
        start_time = time.time()
        current_time = time.time()
        is_board_found = False
        roll = 0.0

        while current_time - start_time < timeout_time:
            image_responses = self.image_client.get_image_from_sources(["left_depth_in_visual_frame", "left_fisheye_image"])
            gray_frame = cv2.imdecode(np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8), -1)
            board_grids = get_board_grids(gray_frame)
            
            bin_frame = convert_to_bin(gray_frame)
            draw_board_centers(bin_frame, board_grids)
            cv2.imshow("Tictacspot", cv2.resize(bin_frame, (640, 480)))

            grid_count = len(board_grids)
            if grid_count == 9:
                grid_11 = compute_center(get_grid(board_grids, 1, 1))

                if is_x_aligned(grid_11[0], self._width/2, TOLERANCE):
                    if is_y_aligned(grid_11[1], self._height/2, TOLERANCE):
                        print(f"[Detected grids: {grid_count}], Board is found and aligned!")
                        self._set_initial_coordinates()
                        self.best_view_roll = roll
                        self.board_gaze_coord = self._get_board_world_coords(image_responses, board_grids, ARM_BOARD_GAZE_OFFSET)
                        if self.board_gaze_coord:
                            is_board_found = True
                            print("Board is found and successfully mapped")
                        break
                    else:
                        print(f"[Detected grids: {grid_count}], Board found: aligning the board vertically...")
                        roll = roll + 0.1
                        self.adjust_roll(roll, body_height = BODY_HEIGHT)
                else:
                    print(f"[Detected grids: {grid_count}], Board found: aligning the board horizontally")
                    self.velocity_move(duration = 0.5, rot = 0.5)

            elif grid_count >= 3 and grid_count < 9:
                grid_01 = compute_center(get_grid(board_grids, 0, 1))
                if is_x_aligned(grid_01[0], self._width/2, TOLERANCE):
                    print(f"[Detected grids: {grid_count}], Board partially found: aligning the board vertically...")
                    roll = roll + 0.1
                    self.adjust_roll(roll, body_height = BODY_HEIGHT)
                else:
                    print(f"[Detected grids: {grid_count}], Board partically found: aligning the board horizontally")
                    if(grid_01[0] > self._width/2):
                        self.velocity_move(duration = 0.5, rot = -0.5)
                    else:
                        self.velocity_move(duration = 0.5, rot = 0.5)
            else:
                print("No board is found, try rotate")
                self.velocity_move(duration = 1, rot = 1)

            current_time = time.time()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        if not is_board_found:
            print("Failed to find the board")
            self.power_off()    

    def pick_up(self):
        self.stand()
        while fetch.pick_up(self.options, self.robot) == False:
            self.go_to_initial()
    
    def place(self, grid_id):
        self.trajectory_move(self.initial_coord['x'], self.initial_coord['y'], self.initial_coord['yaw'] + np.pi/2)   
        self.arm_pose(self.board_gaze_coord[grid_id])

        while self._is_pushing_under_threshold(FORCE_THRESHOLD):
            self.arm_push(0.5, 0.1)
            time.sleep(0.25)

        print("Touched the board! Release the grip")
        time.sleep(0.5)
        self.gripper_open()
        time.sleep(0.5)
        self.arm_push(1, -1)
        time.sleep(1)

        self.arm_stow()
        self.gripper_close()

    def detect_O_positions(self):
        O_piece_list = []
        self.go_to_initial()
        self.adjust_roll(self.best_view_roll, body_height=BODY_HEIGHT)

        image_response = self.image_client.get_image_from_sources(["left_fisheye_image"])
        gray_frame = cv2.imdecode(np.frombuffer(image_response[0].shot.image.data, dtype=np.uint8), -1)
        board_grids = get_board_grids(gray_frame)
        if len(board_grids) == 9:
            board_grids = self._sort_board_grids(board_grids)
            circles = find_circles(gray_frame)
            for circle in circles:
                for grid_id in board_grids:
                    if is_px_inside_contour(board_grids[grid_id], circle[0], circle[1]):
                        O_piece_list.append(grid_id)
        else:
            self.stand()
            return -1
        return O_piece_list


    ### Pixel processing methods

    def _calc_world_coordinate(self, image_responses, x, y, depth_val, offset_distance, grid_key):
        """Returns the world coordinate of a pixel on camera frame and quaternion"""
        if grid_key[0] == '2':
            y = y - Y_BOTTOM_OFFSET
        elif grid_key[0] == '1':
            y = y - Y_MIDDLE_OFFSET
        else:
            y = y - Y_TOP_OFFSET
    
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

    def _get_board_world_coords(self, image_responses, board_grids, offset_distance):
        """Returns the world coordinates of each grid of the game board"""
        sorted_grids = self._sort_board_grids(board_grids)

        depth_map = {}
        world_points = {}
        for grid_key in sorted_grids:
            grid_px = compute_center(sorted_grids[grid_key])
            depth_map[grid_key] = self._get_depth(image_responses, grid_px[0], grid_px[1])

        depth_map = self._fill_missing_depth(depth_map)
        if depth_map == None:
            return None

        for grid_key in sorted_grids:
            grid_px = compute_center(sorted_grids[grid_key])
            world_points[grid_key] = self._calc_world_coordinate(image_responses, grid_px[0], grid_px[1], depth_map[grid_key], offset_distance, grid_key)

        return world_points


    def _get_depth(self, image_responses, x, y):
        """Returns the depth of a pixel in camera frame"""
        cv_depth = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint16)
        cv_depth = cv_depth.reshape(image_responses[0].shot.image.rows, image_responses[0].shot.image.cols)
        depth = cv_depth[y, x]/1000
        return None if depth == 0 else depth

    def _fill_missing_depth(self, depth_map):
        for row in ['0', '1', '2']:
            row_keys = [row + col for col in ['0', '1', '2']]
            row_values = [depth_map[k] for k in row_keys]

            if all(v is None for v in row_values):
                return None

            fallback = next((v for v in row_values if v is not None), None)

            for k in row_keys:
                if depth_map[k] is None:
                    depth_map[k] = fallback
                    print('There are some missing depth values, accuracy might be impacted!')

        return depth_map

    def _sort_board_grids(self, board_grids):
        """Map the grids into tictactoe board"""
        """IMPORTANT: row 0 and 2 are switched to match with minmax 2d matrix idx"""
        sorted_grids = {"00" : get_grid(board_grids, 2,0), "01" : get_grid(board_grids, 2,1), "02" : get_grid(board_grids, 2,2),
                        "10" : get_grid(board_grids, 1,0), "11" : get_grid(board_grids, 1,1), "12" : get_grid(board_grids, 1,2),
                        "20" : get_grid(board_grids, 0,0), "21" : get_grid(board_grids, 0,1), "22" : get_grid(board_grids, 0,2)}
        return sorted_grids
    
    ### SPOT General Commands

    def power_on(self):
        self.robot.logger.info('Powering on robot... This may take several seconds.')
        self.robot.power_on(timeout_sec=20)
        assert self.robot.is_powered_on(), 'Robot power on failed.'
        self.robot.logger.info('Robot powered on.')

    def power_off(self):
        self.robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not self.robot.is_powered_on(), 'Robot power off failed.'
        self.robot.logger.info('Robot safely powered off.')

    ### SPOT Movement Commands

    def stand(self, body_height = 0.0):
        stand_command = RobotCommandBuilder.synchro_stand_command(body_height = body_height)
        cmd_id = self.command_client.robot_command(stand_command)
        blocking_stand(self.command_client, cmd_id)

    def velocity_move(self, duration, x = 0.0, y = 0.0, rot = 0.0):
        cmd = RobotCommandBuilder.synchro_velocity_command(v_x = x, v_y = y, v_rot = rot)
        self.command_client.robot_command(command = cmd, end_time_secs = time.time() + duration)
        time.sleep(duration)
    
    def adjust_roll(self, roll, body_height = 0.0):
        footprint_R_body = bosdyn.geometry.EulerZXY(yaw=0.0, roll=roll, pitch=0.0)
        cmd = RobotCommandBuilder.synchro_stand_command(footprint_R_body=footprint_R_body, body_height=body_height)
        self.command_client.robot_command(command = cmd)
        time.sleep(1)
    
    def trajectory_move(self, x, y, yaw, frame_name = VISION_FRAME_NAME, end_time = 5.0):
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
    
    def arm_stow(self, timeout_sec = 3.0):
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
            seconds=2.0
        )
        follow_arm_command = RobotCommandBuilder.follow_arm_command()
        synchro_command = RobotCommandBuilder.build_synchro_command(arm_command, follow_arm_command)
        command_id = self.command_client.robot_command(synchro_command)
        block_until_arm_arrives(self.command_client, command_id, timeout_sec)

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

    def gripper_open(self, timeout_sec = 3):
        print("Open gripper")
        gripper_command = RobotCommandBuilder.claw_gripper_open_command()
        command_id = self.command_client.robot_command(gripper_command)
        block_until_arm_arrives(self.command_client, command_id, timeout_sec)
    
    def gripper_close(self, timeout_sec = 3):
        gripper_command = RobotCommandBuilder.claw_gripper_close_command()
        command_id = self.command_client.robot_command(gripper_command)
        block_until_arm_arrives(self.command_client, command_id, timeout_sec)
    
    def go_to_initial(self):
        if self.initial_coord:
            self.trajectory_move(self.initial_coord['x'], self.initial_coord['y'], self.initial_coord['yaw'])     

    ### Other methods

    def _is_pushing_under_threshold(self, force_threshold):
        robot_state = self.robot_state_client.get_robot_state()
        force = robot_state.manipulator_state.estimated_end_effector_force_in_hand
        force_x = abs(force.x)
        print(f'Current force: {force_x}, force threshold: {force_threshold}', end='\r')
        return force_x < force_threshold   

    def _is_at_target(self, x, y, yaw, epsilon = 0.1):
        current_state = get_vision_tform_body(self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot)
        current_angle = current_state.rot.to_yaw()
        return (abs(current_state.x - x) < epsilon and
                abs(current_state.y - y) < epsilon and
                abs(current_angle - yaw) < 0.075) 

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

    ### Setters & Getters

    def _set_initial_coordinates(self):
        robot_state = get_vision_tform_body(self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot)
        self.initial_coord['x'] = robot_state.x
        self.initial_coord['y'] = robot_state.y
        self.initial_coord['z'] = robot_state.z
        self.initial_coord['yaw'] = robot_state.rot.to_yaw()

    def get_image_client(self):
        return self.image_client
    
    def get_command_client(self):
        return self.command_client
    
    def get_robot_state_client(self):
        return self.robot_state_client