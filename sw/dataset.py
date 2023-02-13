# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import cv2
import time
import json
import math
import uuid
import argparse
import numpy as np

from hat import Hat, Icon
from common import Vector2
from huemask import hue_mask
from env import MoabEnv, EnvState
from typing import Tuple, List, Optional
from dataclasses import dataclass, astuple
from detector import pixels_to_meters, draw_ball, save_img
from controllers import pid_controller, pid_circle_controller, joystick_controller


def dataset_decorator(fn, logfile="/tmp/dataset", max):
    # Unique filename for the logging run
    moabian_ver = os.enviroment["MOABIAN"]
    run_datetime = datetime.datetime.now().strftime("%y-%m-%d--%H%M-%S")

    # Unique id of user's computer from their mac address
    # TODO: use bcrypt or something more secure?? How important is it to secure
    # the mac address
    unique_id = hash(str(uuid.getnode()))

    logfile += "--" + str(unique_id)
    logfile += "--" + moabian_ver
    logfile += "--" + run_datetime
    logfile += ".tar.gz"

    detected_count = 0
    undetected_count = 0

    # Acts like a normal controller function
    def decorated_fn(state, env_info):
        nonlocal detected_count, undetected_count

        # Run the actual controller
        action, info = fn(state, env_info)

        with open(logfile, "a") as fd:
            print(l, file=fd)

        return action, info

    return decorated_fn


class MoabDatasetEnv(MoabEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.detector = old_hsv_detector(debug=True)  # Use the old detector for now

    def step(self, action, hue=70):
        plate_x, plate_y = action
        self.hat.set_angles(plate_x, plate_y)
        frame, elapsed_time = self.camera()
        img_copy = np.copy(frame)
        ball_detected, cicle_feature = self.detector(frame, hue=hue)
        ball_center, ball_radius = cicle_feature
        x, y = ball_center
        vel_x, vel_y = self.vel_x(x), self.vel_y(y)
        self.sum_x += x
        self.sum_y += y
        buttons = self.hat.get_buttons()
        state = EnvState(x, y, vel_x, vel_y, self.sum_x, self.sum_y)
        return state, ball_detected, buttons, img_copy


def main(controller, filepath="/tmp/dataset/"):
    def wait(env):
        sleep_time = 1 / env.frequency
        while True:
            env.hat.noop()  # Force new transfer to have up to date button reading
            buttons = env.hat.get_buttons()
            if buttons.menu_button:
                return buttons
            time.sleep(sleep_time)

    ball_colors = [
        ("orange", 60),
        ("yellow", 80),
        ("green", 130),
        ("blue", 200),
        ("pink", 320),
        ("purple", 270),  # This doesn't work anyway...
        ("no ball", 0),
    ]
    print("For every ball color, click menu to display color to place in the")
    print("center then click to start the dataset generation for each ball. After")
    print("it's finished, screen will display a new ball color to try.\n")
    counter_interval = 5

    # Create folder to save images
    if not os.path.isdir(filepath):
        os.makedirs(filepath)

    with MoabDatasetEnv(frequency=30, debug=True) as env:
        env.hat.enable_servos()
        env.hat.display_string_icon("", Icon.PAUSE)

        for (color, hue) in ball_colors:
            print("Running:", color)
            env.hat.display_string_icon(color, Icon.PAUSE)
            wait(env)  # Wait until menu is clicked
            state, detected, buttons, img = env.reset(text="RUNNING", icon=Icon.DOT)

            start_time = time.time()
            counter = 0
            while time.time() - start_time < 5:  # Run for ~5 seconds
                action, info = controller((state, detected, buttons))
                state, detected, buttons, img = env.step(action)

                if counter % counter_interval == 0:
                    i = counter // counter_interval
                    if detected:
                        filename = f"{color}.{i}.jpg"
                    else:
                        filename = f"undetected.{color}.{i}.jpg"
                    
                    print(filename)
                    save_img(filepath + filename, img, quality=80)

                counter += 1


if __name__ == "__main__":
    CONTROLLERS = {
        "pid": pid_controller,
        "pid-circle": pid_circle_controller,
        "joystick": joystick_controller,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--controller",
        default="joystick",
        choices=list(CONTROLLERS.keys()),
        help=f"""Select what type of action to take.
        Options are: {CONTROLLERS.keys()}
        """,
    )
    parser.add_argument(
        "-p",
        "--path",
        default="/tmp/dataset/",
        type=str,
        help="""The directory to save the dataset (required trailing /).
        Could be something like ~/dataset/bright-daylight/
        """,
    )
    args, _ = parser.parse_known_args()
    controller = CONTROLLERS[args.controller]()
    main(controller, filepath=args.path)