"""Data discovery and checkout provenance for separate-task runs."""
from pathlib import Path
import subprocess

DATA_FILENAME = "VGEC_balanced.json"


def discover_data(explicit, project_root, input_root=Path("/kaggle/input"), filename=DATA_FILENAME):
    """Prefer an explicit path, then an unambiguous input, then repo data."""
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"DATA_PATH does not exist: {path}")
        return path
    matches = sorted(path.resolve() for path in input_root.rglob(filename) if path.is_file()) if input_root.exists() else []
    if len(matches) > 1:
        raise ValueError("Multiple Kaggle datasets match; set DATA_PATH explicitly: " + ", ".join(map(str, matches)))
    if matches:
        return matches[0]
    local_matches = sorted(path.resolve() for path in (project_root / "dataset").rglob(filename) if path.is_file())
    if len(local_matches) > 1:
        raise ValueError("Multiple local datasets match; set data.path explicitly: " + ", ".join(map(str, local_matches)))
    if local_matches:
        return local_matches[0]
    raise FileNotFoundError(f"Attach a private dataset containing {filename!r} or set data.path explicitly.")


def source_identity(project_root):
    """Read the exact checkout used for training, without trusting parent defaults."""
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=project_root,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    try:
        root = Path(git("rev-parse", "--show-toplevel"))
    except subprocess.CalledProcessError:
        return {"repo_url": None, "branch": None, "commit": None,
                "dirty": None, "project_subdir": None}
    try:
        repo_url = git("remote", "get-url", "origin")
    except subprocess.CalledProcessError:
        repo_url = None
    # Do not retain URL-embedded credentials in public run artifacts.
    from urllib.parse import urlsplit, urlunsplit
    if repo_url and urlsplit(repo_url).username is not None:
        parts = urlsplit(repo_url)
        repo_url = urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))
    return {"repo_url": repo_url, "branch": git("branch", "--show-current") or None,
            "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "project_subdir": str(Path(project_root).resolve().relative_to(root.resolve()))}
