
import eva
import argparse
import subprocess
import os

from eva.utils.misc_utils import get_latest_trajectory, get_latest_image

def send_data(source, destination, partial=False):
    if not os.path.exists(source):
        # Interpret source as data type, and send the latest data of that type
        assert source in ["latest_trajectory", "latest_image"]
        if source == "latest_trajectory":
            source = get_latest_trajectory(success_only=True)
        elif source == "latest_image":
            source = get_latest_image()

    source = source.rstrip("/")
    if partial:
        command = f"rsync -avz --exclude='*.svo2' --exclude='*.jpg' {source} exx:{destination}"
    else:
        command = f"rsync -avz {source} exx:{destination}"
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str)
    parser.add_argument("destination", type=str)
    parser.add_argument("--partial", action="store_true", help="Use partial transfer")
    args = parser.parse_args()

    send_data(args.source, args.destination, partial=args.partial)
