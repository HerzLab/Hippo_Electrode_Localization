"""Purpose: Review coronal images and export the final subject exclusion table.

Author: Qirui Zhang
Created: 2026-09-18
Adapted for the standalone library: 2026-09-27.
"""

from pathlib import Path
import ctypes
import argparse
import hashlib
import sys
import tkinter as tk

# %% Fixed QC inputs and reviewer outputs
parser = argparse.ArgumentParser(description=__doc__)
parser.parse_args()
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'utils'))
from localization_review_utils import ReviewState, ReviewWindow

qc_dir = root / 'results/localization_qc'
image_dir = qc_dir / 'coronal_qc'

# %% Prevent two windows from overwriting the same manual decisions
if sys.platform == 'win32':
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    mutex_name = 'Local\\HippoElectrodeLocalizationVisualQC_' + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    mutex = kernel.CreateMutexW(None, False, mutex_name)
    if mutex is None:
        raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error() == 183:
        kernel.CloseHandle(mutex)
        raise RuntimeError('The localization QC reviewer is already open.')
    ctypes.windll.user32.SetProcessDPIAware()
else:
    import fcntl
    review_lock = (qc_dir / '.review.lock').open('a')
    try:
        fcntl.flock(review_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        review_lock.close()
        raise RuntimeError('The localization QC reviewer is already open.') from None

# %% Restore progress and open the interactive viewer
try:
    state = ReviewState(qc_dir / 'subject_qc.csv', qc_dir)
    window = tk.Tk()
    state.persist()
    viewer = ReviewWindow(window, state, image_dir)
    window.mainloop()
finally:
    if sys.platform == 'win32':
        kernel.CloseHandle(mutex)
    else:
        review_lock.close()
