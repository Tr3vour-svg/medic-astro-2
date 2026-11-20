"""Entry point. Run from the project root:  python app.py"""
import os

from core.ui import build_demo


def main() -> None:
    demo = build_demo()
    # share=True creates a PUBLIC link — only enable it deliberately (GRADIO_SHARE=true).
    demo.launch(share=os.getenv("GRADIO_SHARE", "false").lower() == "true")


if __name__ == "__main__":
    main()