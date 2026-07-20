from pyboy import PyBoy
from pathlib import Path
from poke.utils import default_parser

def main() -> None:
    p = default_parser(rom=True, state=False, out="states/pokeballs.state")
    args = p.parse_args()

    rom = args.rom
    pyboy = PyBoy(rom, window="SDL2")
    pyboy.set_emulation_speed(target_speed=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    while pyboy.tick():
        pass

    with open(args.out, "wb") as f:
        pyboy.save_state(f)

    pyboy.stop()

if __name__ == "__main__":
    main()
