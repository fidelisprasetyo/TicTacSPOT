# TicTacSPOT

This project is about a game of Tic Tac Toe played by a user and Boston Dynamics' SPOT robot.

## Overview

The SPOT robot uses object detection and computer vision to interact with the game. It identifies where the game piece is and determines the best spot to place it on the game board.

## Object Detection

Object detection is used to identify the game pieces. This involves recognizing and locating specific objects within the visual field of the robot.

## Board Detection

Computer vision is used to identify and locate the game board. Board are located and identified by utilizing contour's tracing, combined with homography transformation to compensate with changes in robot's visual because of the robot's movement.

## Gameplay

The game starts with the user or spot making the first move. The SPOT robot then calculates the best move using a Tic Tac Toe algorithm and places its piece on the board. The game continues until there's a winner or the board is full. The starting player can also be swapped.

+ Demo: https://drive.google.com/file/d/18WocukvSnO16KLynNbc8ZAOIfnAXY3pF/view?usp=drive_link

## Future Work

We plan to improve the board detection by using all cameras for the detection. We also aim to add more interactive features to make the game more engaging.

## How to Run the Project

1. Activate virtual environment:
```.\Documents\TicTacSPOT\new_spot_env\Scripts\activate.bat```

2. Run the server-side
```python network_compute_server.py -m "%USERPROFILE%\Documents\TicTacSPOT\Utilities\models\my_efficient_model\saved_model" "%USERPROFILE%\Documents\TicTacSPOT\pieces\annotations\annotations\label_map.pbtxt" -d -n "tictactoe" 192.168.80.3```

3. Run the client-side:
```python main.py -m my_efficient_model -s tictactoe --first player -c 0.85 192.168.80.3```

- -m : Model name running on the external server.
- -s : Service name of external machine learning server.
- --first : Who goes first ['player', 'spot']
- -c : Minimum confidence to return an object for the x-piece (0.0 to 1.0)

## TROUBLESHOOTING

AttributeError: module 'tensorflow' has no attribute 'gfile'

- Go to label_map_util.py
- replace tf.gfile.GFile to tf.io.gfile.GFile

Problem in installing requirements.txt:
- In the requirements.txt, change the file path accordingly:
- object_detection @ file:///C:/Users/fidel/Documents/TicTacSPOT/Utilities/models-with-protos/research
