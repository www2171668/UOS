
"""Git utilities."""
import os


def _get_repo_root():
    """Get ALF repo root path."""
    return os.path.join(os.path.dirname(__file__), "..", "..")


def _exec(command):
    cwd = os.getcwd()
    os.chdir(_get_repo_root())
    stream = os.popen(command)
    ret = stream.read()
    stream.close()
    os.chdir(cwd)
    return ret


def get_revision():
    """Get the current revision of ALF at HEAD."""
    return _exec("git rev-parse HEAD").strip()


def get_diff():
    """Get the diff of ALF at HEAD.

    If the repo is clean, the returned value is an empty string.
    Returns:
        current diff.
    """
    return _exec("git -c core.fileMode=false diff --diff-filter=M")
