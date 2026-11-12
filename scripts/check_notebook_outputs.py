"""Verify notebook outputs and explain how to fix a failed check."""

import shlex
import subprocess
import sys


def main():
    """Check the supplied notebooks without modifying them."""
    filenames = sys.argv[1:]
    result = subprocess.run(
        [sys.executable, "-m", "nbstripout", "--verify", *filenames]
    )
    if result.returncode:
        paths = shlex.join(filenames)
        print(
            "\nNotebook output check failed. Strip output cells with:\n"
            f"  python -m nbstripout {paths}\n"
            "Then stage the updated file(s) and retry your commit:\n"
            f"  git add -- {paths}",
            file=sys.stderr,
        )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
