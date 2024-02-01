"""Logging functions"""
import logging
from typing import Optional
from pathlib import Path


def create_logger(
    name: str,
    level: int,
    folder: str,
    fname: Optional[str] = None,
    print_log: bool = False,
):
    """
    Configure a basic logger.

    Args:
        name (str): logger name
        level (int): level logging.[INFO, WARNING, ERROR, DEBUG]
        folder (str): where to save the log
        fname (Optional, optional): Name of the logging file. If None, save it as "log.txt".Defaults to None.
        print_log (bool): If true, send the messages to the std out (screen) as well
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s:%(levelname)s -> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if fname is None:
        fname = Path(folder) / "log.txt"
    else:
        fname = Path(folder) / name

    logger.handlers.clear()
    handler = logging.FileHandler(fname.as_posix(), mode="a")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if print_log is True:
        debug_handler = logging.StreamHandler()
        debug_handler.setFormatter(formatter)
        logger.addHandler(debug_handler)

    return logger
