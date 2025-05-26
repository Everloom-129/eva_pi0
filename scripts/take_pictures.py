
import eva
from eva.eva import init_context

if __name__ == "__main__":
    with init_context() as runner:
        runner.save_camera_feed()
