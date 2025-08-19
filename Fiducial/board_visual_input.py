
class BoardVisualInput:

    def __init__(self):
        self.board_state = [
            [None, None, None],
            [None, None, None],
            [None, None, None]
        ]
        self.empty_grid_count = 9

    def print(self):
        for row in self.board_state:
            for elem in row:
                if elem not in ['X', 'O']:
                    elem = '-'
                print(elem, end = "  ")
            print()

    def get_empty_grid_count(self):
        return self.empty_grid_count
    
    def get_board_state(self):
        return self.board_state
    
    def check_board_changes(self, occupancy_grid):
        print("Checking for any changes on the game board...")
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
            return None

    def update_board(self, move, player):
        self.board_state[move[0]][move[1]] = player
        self.empty_grid_count -= 1
        print(f"{player} has been placed at {move}")