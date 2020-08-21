import os
from . import terminal_colors
from . import logging
log = logging.Logger()

SAVEDIR = os.path.join(os.path.dirname(__file__), "..", "tune_temp_files")
if not os.path.isdir(SAVEDIR):
    os.mkdir(SAVEDIR)


def get_savepath(name):
    return os.path.join(SAVEDIR, name)
