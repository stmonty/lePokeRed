import sys

import numpy as np
from pyboy import PyBoy
    
    
def main():
    rom = sys.argv[1]
    pyboy = PyBoy(rom, window="SDL2")
    pyboy.set_emulation_speed(target_speed=1)
    while pyboy.tick():
        pass
    # frame = np.asarray(pyboy.screen.ndarray)[:, :, :3]
    # print(frame.shape, frame.dtype)

    with open("states/pokeballs.state", "wb") as f:
        pyboy.save_state(f)

    pyboy.stop()


if __name__ == "__main__":
    main()
