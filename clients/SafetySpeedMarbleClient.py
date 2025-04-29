from proto import service_pb2
from proto import service_pb2_grpc

import math
import time
from clients.marble_client import MarbleClient

class SafetySpeedMarbleClient(MarbleClient):
    def __init__(
            self,
            host: str,
            port: int,
            screen_dir: str,
            name: str,
            *,
            v_max: float = 12.0,
            brake_multiplier: float = 1.1,
            decision_interval: float = 0.2,
    ):
        super().__init__(host, port, screen_dir, name)
        self.V_MAX         = v_max
        self.BRAKE_THRESH  = brake_multiplier * v_max
        self.DT            = decision_interval

    def decision(self, state):
        lv    = state.linear_velocity
        speed = math.sqrt(lv.x**2 + lv.y**2 + lv.z**2)

        forward = (speed < self.V_MAX)
        reset   = (speed > self.BRAKE_THRESH)

        print(f"spd={speed:.2f}, V_MAX={self.V_MAX:.2f}, BRAKE_THRESH={self.BRAKE_THRESH:.2f} "
              f"→ fwd={forward}, reset={reset}")
        time.sleep(self.DT)

        return service_pb2.InputRequest(
            forward=forward,
            back=False,
            left=False,
            right=False,
            reset=reset
        )