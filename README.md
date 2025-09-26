# TicTacSPOT
This project is about a game of Tic Tac Toe played by a user and Boston Dynamics' SPOT robot. 

## Overview

The SPOT robot uses object detection and computer vision to interact with the game. It identifies where the game piece is and determines the best spot to place it on the game board. 

## Object Detection

Object detection is used to identify the game pieces. This involves recognizing and locating specific objects within the visual field of the robot. 

## Board Detection

Computer vision is used to identify and locate the game board. Board are located and identified by utilizing contour's tracing, combined with homography transformation to compensate with changes in robot's visual because of the robot's movement.

## Gameplay

The game starts with the user or making the first move. The SPOT robot then calculates the best move using a Tic Tac Toe algorithm and places its piece on the board. The game continues until there's a winner or the board is full. The starting player can also be swapped.

## Future Work

We plan to improve the board detection by using all cameras for the detection. We also aim to add more interactive features to make the game more engaging.
