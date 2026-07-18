import sys

import numpy as np
from pyboy import PyBoy
    
    
def main():
    rom = sys.argv[1]
    pyboy = PyBoy(rom, window="SDL2")

    playing = True
    while(playing):
        playing = pyboy.tick()

    frame = np.asarray(pyboy.screen.ndarray)[:, :, :3]
    print(frame.shape, frame.dtype)

    pyboy.stop()


if __name__ == "__main__":
    main()
