
from franka_wliang.manager import load_runner


if __name__ == "__main__":
    runner = load_runner()
    runner.reset_robot()
