import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch as t
import torch.nn.functional as F

from poke.actions import ACTIONS
from poke.jepa import JEPA

STARTER_NAMES = ("charmander", "squirtle", "bulbasaur")


def action_lookup(device: t.device) -> t.Tensor:
    lookup_table = t.zeros((len(ACTIONS), len(ACTIONS) - 1), device=device)
    lookup_table[1:, :] = t.eye(len(ACTIONS) - 1, device=device)
    return lookup_table

@t.inference_mode()
def plan_costs(
    model: JEPA,
    initial: t.Tensor,
    goals: t.Tensor,
    action_sequences: t.Tensor,
) -> t.Tensor:
    actions = action_lookup(initial.device)[action_sequences]
    initial_states = initial.expand(action_sequences.size(0), -1)
    predicted_states = model.rollout(initial_states, actions)
    distances = (
        predicted_states[:, :, None, :] - goals[None, None, :, :]
    ).square().mean(dim=-1)
    return distances.flatten(start_dim=1).min(dim=1).values


@t.inference_mode()
def cem_plan(
    model: JEPA,
    initial: t.Tensor,
    goals: t.Tensor,
    horizon: int = 14,
    samples: int = 512,
    elites: int = 64,
    iterations: int = 15,
    smoothing: float = 0.2,
    min_probability: float = 0.01,
    seed: int = 0,
) -> tuple[t.Tensor, float, list[float]]:
    """Use categorical Cross-Entropy Method to find an action sequence."""

    if horizon < 1 or samples < 1 or iterations < 1:
        raise ValueError("horizon, samples, and iterations must be positive")
    if not 0 < elites <= samples:
        raise ValueError("elites must be between 1 and samples")
    if not 0 <= smoothing < 1:
        raise ValueError("smoothing must be in [0, 1)")
    if not 0 <= min_probability < 1 / len(ACTIONS):
        raise ValueError(
            "min_probability must be in [0, 1 / number of actions)"
        )

    device = initial.device
    generator = t.Generator(device=device).manual_seed(seed)

    probabilities = t.full(
        (horizon, len(ACTIONS)),
        1.0 / len(ACTIONS),
        device=device,
    )

    # Track the best individual sequence across every iteration.
    best_sequence = t.empty(horizon, dtype=t.long, device=device)
    best_cost = float("inf")
    cost_history: list[float] = []

    for iteration in range(iterations):
        sequences = t.multinomial(
            probabilities,
            samples,
            replacement=True,
            generator=generator,
        ).transpose(0, 1)

        costs = plan_costs(model, initial, goals, sequences)
        elite_indices = costs.topk(elites, largest=False).indices
        elite_sequences = sequences[elite_indices]

        iteration_best_index = costs.argmin()
        iteration_best_cost = costs[iteration_best_index].item()
        if iteration_best_cost < best_cost:
            best_cost = iteration_best_cost
            best_sequence = sequences[iteration_best_index].clone()

        cost_history.append(best_cost)

        elite_probabilities = F.one_hot(
            elite_sequences,
            num_classes=len(ACTIONS),
        ).float().mean(dim=0)

        probabilities = (
            smoothing * probabilities
            + (1.0 - smoothing) * elite_probabilities
        )
        probabilities = probabilities.clamp_min(min_probability)
        probabilities = probabilities / probabilities.sum(
            dim=-1,
            keepdim=True,
        )

        print(
            f"cem iteration={iteration + 1:02d} "
            f"best_cost={best_cost:.6f}"
        )

    return best_sequence.cpu(), best_cost, cost_history


def default_frame_indices(data_path: str) -> tuple[int, list[int]]:
    """Find the fixed initial frame and one goal frame for each starter."""

    with h5py.File(data_path, "r") as file:
        # The first frame of the first episode matches the saved start state.
        start_index = int(file["episodes"][0, 0])
        goals_by_name: dict[str, int] = {}

        for (start, length), metadata in zip(
            file["episodes"][:],
            file["episode_meta"][:],
        ):
            name = metadata["name"].decode()
            if name not in STARTER_NAMES or name in goals_by_name:
                continue

            start = int(start)
            length = int(length)
            positive_offsets = np.flatnonzero(
                file["party_counts"][start : start + length] > 0
            )
            if len(positive_offsets):
                goals_by_name[name] = start + int(positive_offsets[0])

    missing = [name for name in STARTER_NAMES if name not in goals_by_name]
    if missing:
        raise RuntimeError(
            f"dataset is missing successful goal frames for: {', '.join(missing)}"
        )

    return start_index, [goals_by_name[name] for name in STARTER_NAMES]


def load_frame(data_path: str, index: int) -> t.Tensor:
    with h5py.File(data_path, "r") as file:
        frame = file["frames"][index, :, :, 0].astype(np.float32)
    frame /= 127.5
    frame -= 1.0
    return t.from_numpy(frame).unsqueeze(0).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search for a discrete action plan with categorical CEM"
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--data", default="episodes/poke.h5")
    parser.add_argument("--out", default="plans/poke.json")
    parser.add_argument("--start-index", type=int)
    parser.add_argument(
        "--goal-index",
        dest="goal_indices",
        type=int,
        action="append",
        help="goal frame index; repeat to provide multiple valid goals",
    )
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--elites", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--smoothing", type=float, default=0.2)
    parser.add_argument("--min-probability", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if t.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = t.device(args.device)
    checkpoint = t.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    model = JEPA(
        num_frames=checkpoint["config"]["num_frames"]
    ).to(device)
    model.load_state_dict(checkpoint["jepa"])
    model.eval()

    default_start, default_goals = default_frame_indices(args.data)
    start_index = (
        default_start if args.start_index is None else args.start_index
    )
    goal_indices = (
        default_goals if args.goal_indices is None else args.goal_indices
    )

    with t.inference_mode():
        initial = model.encode(load_frame(args.data, start_index).to(device))
        goals = t.cat(
            [
                model.encode(load_frame(args.data, index).to(device))
                for index in goal_indices
            ]
        )

    sequence, cost, history = cem_plan(
        model,
        initial,
        goals,
        horizon=args.horizon,
        samples=args.samples,
        elites=args.elites,
        iterations=args.iterations,
        smoothing=args.smoothing,
        min_probability=args.min_probability,
        seed=args.seed,
    )

    action_indices = sequence.tolist()
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data": str(Path(args.data).resolve()),
        "start_index": start_index,
        "goal_indices": goal_indices,
        "predicted_cost": cost,
        "cost_history": history,
        "action_indices": action_indices,
        "actions": [ACTIONS[index] for index in action_indices],
        "planner": {
            "horizon": args.horizon,
            "samples": args.samples,
            "elites": args.elites,
            "iterations": args.iterations,
            "smoothing": args.smoothing,
            "min_probability": args.min_probability,
            "seed": args.seed,
        },
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print(f"plan: {' '.join(result['actions'])}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
