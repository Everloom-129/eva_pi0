import time
import multiprocessing
import subprocess
import threading
from pathlib import Path
import os
import glob

data_dir = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data"))

def time_ms():
    return time.time_ns() // 1_000_000

def run_terminal_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True, executable="/bin/bash", encoding="utf8")
    return process

def run_threaded_command(command, args=(), daemon=True):
    thread = threading.Thread(target=command, args=args, daemon=daemon)
    thread.start()
    return thread

def run_multiprocessed_command(command, args=()):
    process = multiprocessing.Process(target=command, args=args)
    process.start()
    return process

def get_latest_trajectory():
    data_dirs = glob.glob(str(data_dir) + "*/**/", recursive=True)
    data_dirs = [d for d in data_dirs if os.path.exists(os.path.join(d, "trajectory.h5"))]
    data_dirs.sort(key=os.path.getmtime)
    data_dirs = data_dirs[-1:]
    return data_dirs[0]

def get_latest_image():
    data_dirs = glob.glob(str(data_dir) + "/images/*")
    data_dirs.sort(key=os.path.getmtime)
    data_dirs = data_dirs[-1:]
    return data_dirs[0]
