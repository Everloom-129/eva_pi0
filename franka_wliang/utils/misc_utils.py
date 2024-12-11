import time
import multiprocessing
import subprocess
import threading
from pynput import keyboard
from contextlib import contextmanager


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

@contextmanager
def keyboard_listener():
    key = {"pressed": None, "released": None}
    def on_press(pressed_key):
        nonlocal key
        key["pressed"] = pressed_key
        key["released"] = None
    def on_release(pressed_key):
        nonlocal key
        key["pressed"] = None
        key["released"] = pressed_key
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    try:
        yield key
    finally:
        listener.stop()
