"""Command line entry point for manual play."""
from __future__ import annotations

import argparse

from .controller import ControllerState
from .emulator import NESEmulator
from .renderer import ScreenRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play an NES ROM using nes-py")
    parser.add_argument("--rom", required=True, help="Path to the .nes ROM file")
    parser.add_argument("--frame-skip", type=int, default=1, help="Number of frames per input step")
    parser.add_argument("--title", default="FC Emulator", help="Window title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    emulator = NESEmulator(args.rom)
    renderer = ScreenRenderer(title=args.title)

    try:
        frame = emulator.reset()
        renderer.draw_frame(frame)

        while True:
            controller: ControllerState = renderer.poll_inputs()
            frame, _reward, done, _info = emulator.step(controller)
            renderer.draw_frame(frame)
            if done:
                frame = emulator.reset()
                renderer.draw_frame(frame)
    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()
        emulator.close()


if __name__ == "__main__":
    main()
