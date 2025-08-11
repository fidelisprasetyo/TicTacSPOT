

class BoardVisualInput:

    def __init__(self):
        self.board_state = [
            [None, None, None],
            [None, None, None],
            [None, None, None]
        ]
        self.empty_grid_count = 9
        self.O = []
        self.X = []

    def print(self):
        for row in self.board_state:
            for elem in row:
                if elem is None:
                    elem = '-'
                print(elem, end = "  ")
            print()

    def get_empty_grid_count(self):
        return self.empty_grid_count
    
    def get_board_state(self):
        return self.board_state

    def get_player_move(self, O_positions):
        latest_move = set(O_positions) - set(self.O)
        if latest_move:
            return latest_move.pop()
        else:
            return None
    
    def update_O(self, move):
        self.O.append(move)
        self.board_state[int(move[0])][int(move[1])] = 'O'
        self.empty_grid_count -= 1
        print(f"Player placed O piece at {move}")

    def update_X(self, move):
        self.board_state[move[0]][move[1]] = 'X'
        self.empty_grid_count -= 1