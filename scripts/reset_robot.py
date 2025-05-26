
import eva
from eva.eva import init_context, init_parser


if __name__ == "__main__":
    with init_context() as runner:
        runner.reset_robot()
