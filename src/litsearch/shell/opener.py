from __future__ import annotations

import platform
import subprocess
from pathlib import Path


def open_pdf(path: str, page: int | None = None) -> None:
    """Open a local PDF with the platform default viewer.

    Most default viewers do not accept a portable page argument; callers may
    supply it for future platform-specific implementations.
    """
    pdf_path = str(Path(path).expanduser())
    system = platform.system()
    if system == "Darwin":
        command = ["open", pdf_path]
    elif system == "Windows":
        command = ["cmd", "/c", "start", "", pdf_path]
    elif system == "Linux":
        command = ["xdg-open", pdf_path]
    else:
        raise RuntimeError(f"Unsupported operating system: {system}")
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
