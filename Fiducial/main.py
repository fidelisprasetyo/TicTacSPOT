import argparse
import tictactoe as ttt
import cv2
import time

from tictacspot import TicTacSpot
from board_input import BoardInput
from board_input import OCCUPANCE_MULTIPLE_MOVES_ERROR

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util

PLAYER_TURN_TIME = 5
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
    parser.add_argument('--first', choices=['player', 'spot'], default='player',
                        help='Who goes first: player (O) or spot (X)')
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
        spot.pick_up()
        # board = BoardInput()
        # player_turn = ttt.O
        # spot_turn = ttt.X

        # if options.first == 'player':
        #     ttt.START_PLAYER = player_turn
        # else:
        #     ttt.START_PLAYER = spot_turn

        # spot.power_on()
        # spot.stand()

        # cv2.namedWindow('Tictacspot', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Tictacspot', 640, 480)
        # cv2.moveWindow("Tictacspot", 500, 0)

        # cv2.namedWindow('Tictacspot_bin', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Tictacspot_bin', 640, 480)
        # cv2.moveWindow("Tictacspot_bin", 1140, 0)

        # cv2.namedWindow('X-piece detection', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('X-piece detection', 480, 360)
        # cv2.moveWindow('X-piece detection', 500, 510)

        # cv2.namedWindow('Tic-tac-toe', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Tic-tac-toe', 480, 480)
        # cv2.moveWindow('Tic-tac-toe', 500, 900)

        # spot.find_board()
        # board.print()

        # empty_grid_count = board.get_empty_grid_count()
        # move_number = 0

        # while empty_grid_count > 0:

        #     if empty_grid_count <= 4:
        #         piece = ttt.winner(board.get_board_state())
        #         if piece == ttt.X:
        #             print("Spot wins")
        #             board.print()
        #             break
        #         elif piece == ttt.O:
        #             print("Player wins")
        #             board.print()
        #             break

        #     if move_number % 2 == 0:
        #         current_turn = player_turn if options.first == 'player' else spot_turn
        #     else:
        #         current_turn = spot_turn if options.first == 'player' else player_turn
            
        #     if current_turn == player_turn:
        #         print("Player's turn!")
        #         spot.stand()
        #         for i in range(PLAYER_TURN_TIME, 0, -1):
        #             print(f"Waiting... {i} seconds remaining", end = '\r')
        #             time.sleep(1)
        #         print("Checking the board for changes...")
        #         occupancy_grid = spot.get_board_occupancy()
        #         move = board.check_board_changes(occupancy_grid)

        #         if move and move != OCCUPANCE_MULTIPLE_MOVES_ERROR:
        #             print(f"Player's move: {move}")
        #             board.update_board(move, current_turn)
        #             board.print()
                
        #         if move == None:
        #             continue
                
        #     else:
        #         print("Spot's turn!")
        #         move = ttt.minimax(board.get_board_state())
        #         row, col = move

        #         spot.pick_up()
        #         spot.place(row, col)

        #         occupancy_grid = spot.get_board_occupancy()
        #         move_detected = board.check_board_changes(occupancy_grid)

        #         try_count = 0
        #         while move_detected == OCCUPANCE_MULTIPLE_MOVES_ERROR:
        #             if try_count == 5:
        #                 print("[ERROR] Multiple detections error, stopping the game.")
        #                 break
        #             occupancy_grid = spot.get_board_occupancy()
        #             move_detected = board.check_board_changes(occupancy_grid)
        #             try_count += 1

        #         if move_detected is None:
        #             print(f"Spot failed to place an X-piece at {move}, try again.")
        #             continue
                
        #         if move_detected != move:
        #             print(f"[ERROR] Spot's move = {move}, detected move = {move_detected}, stopping the game.")
        #             spot.save_img_log()
        #             time.sleep(5)
        #             break

        #         print(f"Spot's move: {move}")
        #         board.update_board(move, current_turn)
        #         board.print()

        #         spot.go_to_initial()
            
        #     move_number += 1
        #     empty_grid_count = board.get_empty_grid_count()

        # if piece == None:
        #     print("Draw!")
        #     board.print()

        # spot.power_off()


if __name__ == '__main__':
    main()