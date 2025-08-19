import cv2
import numpy as np
import argparse
import time
import math
import tictactoe as ttt

from contour_detection import *
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


        # Tic Tac Toe
        player_turn = ttt.O
        spot_turn = ttt.X
        board = BoardVisualInput()
        board.print()

        empty_grid_count = board.get_empty_grid_count()

        while empty_grid_count > 0:

            if empty_grid_count <= 4:
                piece = ttt.winner(board.get_board_state())
                if piece == ttt.X:
                    print("Spot wins")
                    break
                elif piece == ttt.O:
                    print("Player wins")
                    break
            
            if empty_grid_count % 2 == 0:
                current_turn = spot_turn
            else:
                current_turn = player_turn
            
            if current_turn == player_turn:
                print("Player's turn!")
                spot.stand()
                time.sleep(5) # time based check

                occupancy_grid = spot.get_board_occupancy()
                move = board.check_board_changes(occupancy_grid)

                if move:
                    print(f"Player's move: {move}")
                    board.update_board(move, current_turn)
                    board.print()
                
            else:
                print("Spot's turn!")
                move, _ = ttt.minimax(board.get_board_state())
                row, col = move

                spot.pick_up()
                spot.place(row, col)
                
                print(f"Spot's move: {move}")
                board.update_board(move, current_turn)
                board.print()

                spot.go_to_initial()
            
            empty_grid_count = board.get_empty_grid_count()

        if piece == None:
            print("Draw!")

        spot.power_off()

    






    ### for testing purposes

    # def area_dif(area1, area2):
    #     return abs(area1 - area2) < 200

    # image_client = robot.ensure_client(ImageClient.default_service_name)
    # while True:
    #     image_responses = image_client.get_image_from_sources(["left_depth_in_visual_frame", "left_fisheye_image"])

    #     cv_visual = cv2.imdecode(np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8), -1)
    #     bin_img = convert_to_bin(cv_visual)


    #     params = cv2.SimpleBlobDetector_Params()
    #     detector = cv2.SimpleBlobDetector_create(params)
    #     keypoints = detector.detect(bin_img)
    #     # Draw detected blobs as red circles.
    #     # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    #     im_with_keypoints = cv2.drawKeypoints(bin_img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



        # rectangles = defaultdict(list)
        # grids = []
        # # Find contours
        # contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # if hierarchy is not None:
        #     for idx, cnt in enumerate(contours):
        #         parent = hierarchy[0][idx][3]
        #         if parent == -1:
        #             continue
        #         cnt_approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        #         # filter out non-rectangles
        #         if len(cnt_approx) == 4 and cv2.isContourConvex(cnt_approx):
        #             area = cv2.contourArea(cnt_approx)
        #             if area_dif(area, 2277):
        #                 rectangles[parent].append(cnt_approx)

        # for parent, children in rectangles.items():
        #     if len(children) >= 3:
        #         for child in children:
        #             grids.append(child)


        # for grid in grids:
        #     draw_board_centers(bin_img, grid)



    #     cv2.imshow("Tictacspot", im_with_keypoints)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()