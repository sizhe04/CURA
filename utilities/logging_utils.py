import sys


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        primary = self._streams[0] if self._streams else None
        return bool(getattr(primary, "isatty", lambda: False)())

    def fileno(self):
        primary = self._streams[0] if self._streams else None
        if hasattr(primary, "fileno"):
            try:
                return primary.fileno()
            except Exception:
                return -1
        return -1

    @property
    def encoding(self):
        primary = self._streams[0] if self._streams else None
        return getattr(primary, "encoding", "utf-8")

    def writable(self):
        return True

    def readable(self):
        return False

    def seekable(self):
        return False


def setup_console_tee(log_fp):
    """Replace sys.stdout/stderr with a tee writing to given file object.

    Returns (stdout_backup, stderr_backup).
    Caller is responsible for closing file and calling restore_console.
    """
    stdout_backup, stderr_backup = sys.stdout, sys.stderr
    sys.stdout = _Tee(sys.stdout, log_fp)
    sys.stderr = _Tee(sys.stderr, log_fp)
    return stdout_backup, stderr_backup


def restore_console(stdout_backup, stderr_backup):
    try:
        sys.stdout, sys.stderr = stdout_backup, stderr_backup
    except Exception:
        pass


