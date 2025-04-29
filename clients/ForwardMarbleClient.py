import time
from clients.marble_client import MarbleClient
from proto import service_pb2
from proto import service_pb2_grpc

class ForwardMarbleClient(MarbleClient):
    def __init__(
            self,
            host: str,
            port: int,
            screen_dir: str,
            name: str,
            *,
            decision_interval: float = 0.2,
    ):
        super().__init__(host, port, screen_dir, name)
        self.DT = decision_interval

    def decision(self, state):
        time.sleep(self.DT)

        return service_pb2.InputRequest(
            forward=True,
            back=False,
            left=False,
            right=False,
            reset=False
        )
