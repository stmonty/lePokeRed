import argparse

import numpy as np

from poke.actions import ACTIONS, to_multihot
from poke.env import PokeEnv
from poke.ram import party_count, starter_selected, starter_species
from poke.writer import EpisodeWriter

NOOP = ACTIONS.index("noop")

# from states/pokeballs.state the player faces the middle ball (which is Squirtle)
BALL_PREFIX = {
    "charmander": ["left", "up"],
    "squirtle": [],
    "bulbasaur": ["right", "up"],
}
TAKE = ["a"] * 12
DECLINE_THEN_TAKE = ["a"] * 7 + ["down", "a"] + TAKE
WANDER = [
    ["down", "up", "up"],
    ["b", "down", "up", "up"],
    ["left", "up", "down", "up"],
]


def scripted_plans():
    plans = []
    for ball, prefix in BALL_PREFIX.items():
        plans.append((ball, prefix + TAKE))
        plans.append((ball + "_noyes", prefix + DECLINE_THEN_TAKE))
    for i, prefix in enumerate(WANDER):
        plans.append((f"wander{i}", prefix + TAKE))
    return plans


def record(env, writer, action_idxs, kind, name):
    frame = env.reset()
    for idx in action_idxs:
        writer.add_step(frame[..., 0], to_multihot(idx), party_count(env))
        frame = env.step(idx)
    writer.add_step(frame[..., 0], to_multihot(NOOP), party_count(env))
    ok, species = starter_selected(env), starter_species(env)
    writer.end_episode(kind, name, ok, -1 if species is None else species)
    return ok, species


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rom", default="pokered.gb")
    p.add_argument("--state", default="states/pokeballs.state")
    p.add_argument("--out", default="episodes/poke.h5")
    p.add_argument("--noisy", type=int, default=32)
    p.add_argument("--noisy-prefix", type=int, default=3)
    p.add_argument("--random", type=int, default=200)
    p.add_argument("--random-steps", type=int, default=24)
    p.add_argument("--act-frames", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    env = PokeEnv(args.rom, args.state, headless=True, act_frames=args.act_frames)

    with EpisodeWriter(args.out, args.rom, args.state, args.act_frames, args.seed) as w:
        for name, plan in scripted_plans():
            ok, species = record(env, w, [ACTIONS.index(a) for a in plan], "scripted", name)
            print(f"scripted {name:18s} {len(plan) + 1:3d} steps  goal={ok} species={species}")
            if not ok:
                print("  WARNING: plan did not reach a starter")

        hits = 0
        for i in range(args.noisy):
            n = int(rng.integers(1, args.noisy_prefix + 1))
            prefix = rng.integers(0, len(ACTIONS), size=n).tolist()
            ok, _ = record(env, w, prefix + [ACTIONS.index(a) for a in TAKE], "noisy", f"noisy{i}")
            hits += ok
        print(f"noisy    {args.noisy} episodes  goal_hits={hits}")

        hits = 0
        for i in range(args.random):
            idxs = rng.integers(0, len(ACTIONS), size=args.random_steps).tolist()
            ok, _ = record(env, w, idxs, "random", f"random{i}")
            hits += ok
            if (i + 1) % 25 == 0:
                print(f"random   {i + 1}/{args.random}")
        print(f"random   {args.random} episodes x {args.random_steps} steps  goal_hits={hits}")

    env.close()
    print("lePokeRed: wrote", args.out)


if __name__ == "__main__":
    main()
