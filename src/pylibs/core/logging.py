"""Logging setup: library code only ever calls `get_logger(__name__)`, never
`basicConfig`. `configure_console_logging` and `configure_project_logging`
are the only places handlers get attached -- the latter called whenever a
project is loaded or created, the former standalone at CLI startup (before
any project necessarily exists) and again internally by the latter.
`set_debug_logging` toggles both to DEBUG level process-wide.
"""

import logging
from pathlib import Path

PACKAGE_LOGGER_NAME = "pylibs"
LOG_FILENAME = "pylibs.log"

_CONSOLE_MARKER = "_pylibs_console_handler"
_FILE_MARKER = "_pylibs_project_log_handler"

_debug_enabled = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_debug_logging(enabled: bool) -> None:
    """Toggle DEBUG-level logging process-wide -- set once at CLI startup,
    before any command that might log anything runs. Takes effect on the
    console immediately (see `configure_console_logging`) and on the next
    project's file handler (see `configure_project_logging`)."""
    global _debug_enabled
    _debug_enabled = enabled


def configure_console_logging(debug: bool = False) -> None:
    """Attach a console handler to the "pylibs" logger (idempotent -- safe
    to call more than once, and updates an already-attached handler's level
    rather than adding a second one), and set both it and the logger's own
    level: WARNING and above to stderr by default (replicating Python's own
    handler-of-last-resort, since adding any handler here disables it), or
    DEBUG and above when `debug`. Callable standalone -- e.g. at CLI
    startup, before any project exists -- so commands that never touch a
    project (like `convert-ele-to-libs`) still get debug-level console
    output; `configure_project_logging` also calls this for its own
    console step."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    for handler in logger.handlers:
        if getattr(handler, _CONSOLE_MARKER, False):
            handler.setLevel(logging.DEBUG if debug else logging.WARNING)
            return

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.WARNING)
    setattr(console_handler, _CONSOLE_MARKER, True)
    logger.addHandler(console_handler)


def configure_project_logging(root: Path) -> None:
    """Route every `pylibs.*` log record to `<root>/pylibs.log`, in addition
    to a console handler (see `configure_console_logging`).

    Called by `Project.create`/`Project.load`, so the file target always
    follows whichever project is currently active in this process --
    operating on a second project re-points the file handler instead of
    interleaving two projects' messages in one file.
    """
    configure_console_logging(_debug_enabled)
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)

    for handler in list(logger.handlers):
        if getattr(handler, _FILE_MARKER, False):
            logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(Path(root) / LOG_FILENAME)
    file_handler.setLevel(logging.DEBUG if _debug_enabled else logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    setattr(file_handler, _FILE_MARKER, True)
    logger.addHandler(file_handler)
