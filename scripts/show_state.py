import numpy as np
from pyboy import PyBoy
from PIL import Image
from poke.utils import default_parser

def main() -> None:
    p = default_parser(rom=True, state=True, out="frame.png")
    p.add_argument("--scale", type=int, default=3)
    args = p.parse_args()

    pyboy = PyBoy(args.rom, window="null")
    with open(args.state, "rb") as f:
        pyboy.load_state(f)

    pyboy.tick(1, render=True)
    frame = np.asarray(pyboy.screen.ndarray)[:, :, :3]
    Image.fromarray(frame).resize((frame.shape[1] * args.scale, frame.shape[0] * args.scale), Image.NEAREST).save(args.out)
    
    pyboy.stop()
    print("lePokeRed: Wrote", args.out)
    
if __name__ == "__main__":
    main()
