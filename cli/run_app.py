import subprocess
import sys
from pathlib import Path


def run() -> int:
    project_root = Path(__file__).resolve().parents[1]
    app_path = project_root / "src" / "api" / "app.py"
    return subprocess.call([sys.executable, str(app_path)], cwd=str(project_root))


if __name__ == "__main__":
    raise SystemExit(run())
