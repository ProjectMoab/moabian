import numpy as np


def kalman_filter(dt=1 / 30):
    g = 9.81  # gravity
    F = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    G = np.array(
        [
            [0.5 * dt ** 2 * (3 / 5) * g, 0],
            [0, 0.5 * dt ** 2 * (3 / 5) * g],
            [dt * (3 / 5) * g, 0],
            [0, dt * (3 / 5) * g],
        ]
    )
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]).T
    x = None
    P = np.zeros((4, 4))  # Use initial state uncertainty... figure out what it is
    R = 0.01 * np.eye((2, 2))  # Figure out what it is
    I = np.eye(4)

    def fn(measurement, action):
        """
        args:
            measurement: [x_pos, y_pos]
            action: [pitch, roll]

        returns:
            state: [x_pos_hat, y_pos_hat, x_pos_dot_hat, y_pos_dot_hat]
        """
        nonlocal x, P
        u = np.array(action).reshape(2, 1)

        # If this is the first measurement just return the measurement
        if x is None:
            x = np.array([*measurement, 0, 0]).reshape(4, 1)
            P = F @ P @ F.T + Q
        else:
            # Kalman filter steps
            # 1. Compute Kalman Gain
            K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
            # 2. Update estimate with measurement
            z = np.array(measurement).reshape(2, 1)
            x = x + K @ (z - H @ x)
            # 3. Update the estimate uncertainty
            P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T

        x_current = x.copy()
        # P_current = P.copy()

        # FOR NEXT TIMESTEP - Time update (predict next state info)
        # 1. Calculate next state estimate using current state
        x = F @ x + G @ u
        # 2. Exrapolate next state uncertainty using current uncertainty
        P = F @ P @ F + Q

        return x_current.reshape(-1).tolist()

    return fn
