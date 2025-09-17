import copy
import logging
import os
import sys

from cbam.utils import config

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

COLORS = {
    'WARNING': YELLOW,
    'INFO': BLUE,
    'DEBUG': WHITE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}

FORMAT = "[%(asctime)s][%(levelname)-18s] " \
         "%(message)s ($BOLD%(filename)s$RESET:%(lineno)d) "


class ColoredFormatter(logging.Formatter):
    def __init__(self, use_color=True):
        if use_color:
            fmt = FORMAT.replace("$RESET", RESET_SEQ).replace("$BOLD",
                                                              BOLD_SEQ)
        else:
            fmt = FORMAT.replace("$RESET", "").replace("$BOLD", "")
        logging.Formatter.__init__(self, fmt)
        self.use_color = use_color

    def format(self, record):
        record = copy.copy(record)
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            color: str = COLOR_SEQ % (30 + COLORS[levelname])
            record.msg = color + record.msg + RESET_SEQ
            record.levelname = color + levelname + RESET_SEQ
        return logging.Formatter.format(self, record)


def add_colored_formatter(
        logger: logging.Logger = logging.getLogger()) -> None:
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ColoredFormatter())
    logger.addHandler(console)


def add_filehandler(file: str,
                    logger: logging.Logger = logging.getLogger()
                    ) -> logging.FileHandler:
    filehandler = logging.FileHandler(file)
    filehandler.setFormatter(logging.Formatter(
        "[%(asctime)s][%(levelname)-7s]  " 
        "%(message)s (%(filename)s:%(lineno)d) "))
    logger.addHandler(filehandler)
    return filehandler


def configure_root_loger(logging_level: int,
                         file: str or None = None) -> logging.Logger:
    root = logging.getLogger()
    for h in root.handlers:
        root.removeHandler(h)
    root.setLevel(logging_level)
    add_colored_formatter(logger=root)
    if file is not None:
        os.makedirs(os.path.dirname(file), exist_ok=True)
        add_filehandler(file, logger=root)
    error_handler = add_filehandler(
        config.Config.get_logdir() + 'error.log',
        logger=root)
    error_handler.setLevel(logging.ERROR)
    return root
