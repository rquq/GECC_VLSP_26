"""Find owned subprocess groups before signaling a notebook's process tree."""

import os
import subprocess


def descendant_groups(root_pid):
    output = subprocess.check_output(["ps", "-eo", "pid=,ppid=,pgid="], text=True)
    rows = [
        tuple(map(int, line.split())) for line in output.splitlines() if line.strip()
    ]
    owned = {root_pid}
    while True:
        children = {pid for pid, parent, _ in rows if parent in owned}
        if children <= owned:
            break
        owned |= children
    # A notebook launcher and every model stage create their own sessions.
    # Never signal the notebook kernel's own process group.
    return {
        group
        for pid, _, group in rows
        if pid in owned and group > 0 and group != os.getpgrp()
    }


def signal_groups(groups, signal):
    for group in sorted(groups):
        try:
            os.killpg(group, signal)
        except ProcessLookupError:
            pass
