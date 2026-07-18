"""根目录入口：通过 uvicorn 或 CLI 启动应用。"""

from pathlib import Path
import sys


def main() -> None:
    """CLI 入口。"""
    from deep_research.cli import main as run_main
    run_main()


if __name__ == "__main__":
    main()
