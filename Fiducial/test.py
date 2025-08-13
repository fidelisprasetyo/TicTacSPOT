import cv2
import numpy as np
import argparse
import time
import math
import tictactoe as ttt

from tictacspot import TicTacSpot
from board_visual_input import BoardVisualInput

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

def main():
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-j', '--jpeg-quality-percent', help='JPEG quality percentage (0-100)',
                        type=int, default=50)
    parser.add_argument('-s', '--ml-service',
                        help='Service name of external machine learning server.', required=True)
    parser.add_argument('-m', '--model', help='Model name running on the external server.',
                        required=True)
    parser.add_argument('-c', '--confidence-piece',
                        help='Minimum confidence to return an object for the dogoy (0.0 to 1.0)',
                        default=0.8, type=float)
    parser.add_argument('-d', '--distance-margin', default=0.60,
                        help='Distance [meters] that the robot should stop from the fiducial.')
    parser.add_argument('--limit-speed', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='If the robot should limit its maximum speed.')
    parser.add_argument('--avoid-obstacles', default=False, type=lambda x:
                        (str(x).lower() == 'true'),
                        help='If the robot should have obstacle avoidance enabled.')
    options = parser.parse_args()

    sdk = bosdyn.client.create_standard_sdk('TicTacSPOT')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()


    # new main
    assert not robot.is_estopped()
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        
        spot = TicTacSpot(robot, options)
        spot.power_on()
        spot.stand()
        spot.find_board()
        spot.stand()

        spot.pick_up()
        spot.place('01')
        spot.go_to_initial()

        # spot.go_to_initial()

        # Tic Tac Toe
        # player_turn = ttt.O
        # spot_turn = ttt.X
        # board = BoardVisualInput()
        # board.print()

        # empty_grid_count = board.get_empty_grid_count()

        # while empty_grid_count > 0:

        #     if empty_grid_count <= 4:
        #         piece = ttt.winner(board.get_board_state())
        #         if piece == ttt.X:
        #             print("Spot wins")
        #             break
        #         elif piece == ttt.O:
        #             print("Player wins")
        #             break
            
        #     if empty_grid_count % 2 == 0:
        #         current_turn = spot_turn
        #     else:
        #         current_turn = player_turn
            
        #     if current_turn == player_turn:
        #         print("Player's turn!")
        #         spot.stand()
        #         time.sleep(5) # time based check
        #         O_positions = spot.detect_O_positions()

        #         if O_positions == -1:
        #             print("Failed to detect the board....")
        #             continue

        #         move = board.get_player_move(O_positions)
        #         if move:
        #             print(f"Player's move: ({move[0]}, {move[1]})")
        #             board.update_O(move)
        #             board.print()
                
        #     else:
        #         print("Spot's turn!")
        #         move, _ = ttt.minimax(board.get_board_state())
        #         print(f"Spot's move: {move}")

        #         spot.pick_up()
        #         spot.place(f"{move[0]}{move[1]}")
        #         board.update_X(move)
        #         board.print()

        #         spot.go_to_initial()
            
        #     empty_grid_count = board.get_empty_grid_count()

        # if piece == None:
        #     print("Draw!")

        # spot.power_off()

    

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