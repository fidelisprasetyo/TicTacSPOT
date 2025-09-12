import numpy as np
import cv2

OCCUPANCE_MULTIPLE_MOVES_ERROR = -1

class BoardInput:

    def __init__(self):
        self.board_state = [
            [None, None, None],
            [None, None, None],
            [None, None, None]
        ]
        self.empty_grid_count = 9

    def print(self):
        img_size = 600
        cell_size = img_size // 3
        line_color = (0, 0, 0)
        x_color = (0, 0, 255)   
        o_color = (255, 0, 0)
        thickness = 4

        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

        for i in range(1, 3):
            cv2.line(img, (0, i * cell_size), (img_size, i * cell_size), line_color, thickness)
            cv2.line(img, (i * cell_size, 0), (i * cell_size, img_size), line_color, thickness)

        for row in range(3):
            for col in range(3):
                value = self.board_state[row][col]
                center_x = col * cell_size + cell_size // 2
                center_y = row * cell_size + cell_size // 2

                if value == 'X':
                    offset = cell_size // 4
                    cv2.line(img, (center_x - offset, center_y - offset),
                                  (center_x + offset, center_y + offset), x_color, thickness)
                    cv2.line(img, (center_x + offset, center_y - offset),
                                  (center_x - offset, center_y + offset), x_color, thickness)
                elif value == 'O':
                    radius = cell_size // 4
                    cv2.circle(img, (center_x, center_y), radius, o_color, thickness)

        cv2.imshow('Tic-tac-toe', img)
        cv2.waitKey(10)

    def get_empty_grid_count(self):
        return self.empty_grid_count
    
    def get_board_state(self):
        return self.board_state
    
    def check_board_changes(self, occupancy_grid):
        change = []
        for row in range(3):
            for col in range(3):
                prev_state = self.board_state[row][col]
                cur_state = occupancy_grid[row][col]
                if cur_state == 1 and prev_state == None:
                    change.append((row,col))
        
        if len(change) == 1:
            return change.pop()
        elif len(change) == 0:
            print("No new move detected!")
            return None
        else:
            print(f"Warning: Multiple moves detected: {change} or the board was obstructed.")
            return OCCUPANCE_MULTIPLE_MOVES_ERROR

    def update_board(self, move, player):
        self.board_state[move[0]][move[1]] = player
        self.empty_grid_count -= 1
        print(f"{player} has been placed at {move}")