import argparse
from PIL import Image
from poke.env import PokeEnv
from poke.actions import ACTIONS


def save(frame, path, scale=3):
    img = Image.fromarray(frame)
    img = img.resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST)
    img.save(path)
    print("lePokeRed: wrote", path, "shape", frame.shape)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rom", default="pokered.gb")
    p.add_argument("--state", default="states/pokeballs.state")
    args = p.parse_args()

    env = PokeEnv(args.rom, args.state, headless=True)

    before = env.reset()
    save(before, "before.png")

    a_idx = ACTIONS.index("a")
    after = env.step(a_idx)
    save(after, "after.png")

    env.close()


if __name__ == "__main__":
    main()
