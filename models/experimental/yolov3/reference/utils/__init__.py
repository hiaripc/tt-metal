# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# YOLOv3 🚀 by Ultralytics, AGPL-3.0 license
"""
utils/initialization
"""

import contextlib
import platform
import threading


def emojis(str=""):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode("ascii", "ignore") if platform.system() == "Windows" else str


class TryExcept(contextlib.ContextDecorator):
    # YOLOv3 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager
    def __init__(self, msg=""):
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


def threaded(func):
    # Multi-threads a target function and returns thread. Usage: @threaded decorator
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


def join_threads(verbose=False):
    # Join all daemon threads, i.e. atexit.register(lambda: join_threads())
    main_thread = threading.current_thread()
    for t in threading.enumerate():
        if t is not main_thread:
            if verbose:
                print(f"Joining thread {t.name}")
            t.join()


def notebook_init(verbose=True):
    # Check system software and hardware
    print("Checking setup...")

    import os
    import shutil

    from utils.general import check_font, check_requirements, is_colab
    from utils.torch_utils import select_device  # imports

    check_font()

    import psutil
    from IPython import display  # to display images and clear console output

    if check_requirements("wandb", install=False):
        os.system("pip uninstall -y wandb")  # eliminate unexpected account creation prompt with infinite hang
    if is_colab():
        shutil.rmtree("/content/sample_data", ignore_errors=True)  # remove colab /sample_data directory

    # System info
    if verbose:
        gb = 1 << 30  # bytes to GiB (1024 ** 3)
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage("/")
        display.clear_output()
        s = f"({os.cpu_count()} CPUs, {ram / gb:.1f} GB RAM, {(total - free) / gb:.1f}/{total / gb:.1f} GB disk)"
    else:
        s = ""

    select_device(newline=False)
    print(emojis(f"Setup complete ✅ {s}"))
    return display
