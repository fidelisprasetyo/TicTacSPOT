import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.api import trajectory_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.util import seconds_to_duration
from bosdyn.api import image_pb2

import Fiducial.stitch_front_images.stitch_front_images as front_cameras
import OpenGL.GL as gl
import contour

from PIL import Image
import io


def extract_image_and_intrinsics(image_response):

    # Decode image (JPEG assumed)
    image_format = image_response.shot.image.format
    if image_format == image_response.shot.image.FORMAT_JPEG:
        img = np.asarray(Image.open(io.BytesIO(image_response.shot.image.data)))
    else:
        raise ValueError("Unsupported image format: only JPEG is supported here.")

    intr = image_response.source.pinhole.intrinsics

    # Camera matrix K
    fx = intr.focal_length.x
    fy = intr.focal_length.y
    cx = intr.principal_point.x
    cy = intr.principal_point.y

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])

    print(K)

    # Distortion coefficients (k1, k2, p1, p2, k3)
    dist = np.array([
        getattr(intr, 'k1', 0),
        getattr(intr, 'k2', 0),
        getattr(intr, 'p1', 0),
        getattr(intr, 'p2', 0),
        getattr(intr, 'k3', 0)
    ])

    return img, K, dist


def track(image_client: ImageClient):

    while True:
        image_responses = image_client.get_image_from_sources(["frontleft_fisheye_image", "frontright_fisheye_image"])

        img_left, K_left, dist_left = extract_image_and_intrinsics(image_responses[0])
        img_right, K_right, dist_right = extract_image_and_intrinsics(image_responses[1])

        img_left_undistorted = cv2.undistort(img_left, K_left, dist_left)
        img_right_undistorted = cv2.undistort(img_right, K_right, dist_right)

        img_left_undistorted = cv2.rotate(img_left_undistorted, cv2.ROTATE_90_CLOCKWISE)
        img_right_undistorted = cv2.rotate(img_right_undistorted, cv2.ROTATE_90_CLOCKWISE)


        #Stitch horizontally
        gray_frame = np.hstack((img_right_undistorted, img_left_undistorted))

        # gray_frame = stitch_images(img_left_undistorted, img_right_undistorted)


def main():
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-j', '--jpeg-quality-percent', help='JPEG quality percentage (0-100)',
                        type=int, default=50)
    options = parser.parse_args()

    sdk = bosdyn.client.create_standard_sdk('TicTacSPOT')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    
    assert not robot.is_estopped()
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    
    front_cameras.stitch(robot, options)
    track(image_client)


if __name__ == '__main__':
    main()