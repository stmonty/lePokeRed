"""Execute a saved action plan in PyBoy and count real successes."""

import argparse
import json
from pathlib import Path

from poke.actions import ACTIONS
from poke.env import PokeEnv
from poke.ram import starter_selected, starter_species


def load_action_indices(path: str | Path) -> list[int]:
    with Path(path).open(encoding="utf-8") as file:
        plan = json.load(file)

    action_indices = plan.get("action_indices")
    if not isinstance(action_indices, list) or not action_indices:
        raise ValueError("plan must contain a non-empty action_indices list")
    if any(
        not isinstance(index, int) or not 0 <= index < len(ACTIONS)
        for index in action_indices
    ):
        raise ValueError("plan contains an invalid action index")

    return action_indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute a saved starter-selection plan in PyBoy"
    )
    parser.add_argument("--plan", default="plans/poke.json")
    parser.add_argument("--rom", default="pokered.gb")
    parser.add_argument("--state", default="states/pokeballs.state")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--act-frames", type=int, default=60)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--stop-on-success",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("runs must be positive")

    action_indices = load_action_indices(args.plan)
    print(f"plan: {' '.join(ACTIONS[index] for index in action_indices)}")

    environment = PokeEnv(
        args.rom,
        args.state,
        headless=not args.show,
        act_frames=args.act_frames,
    )
    successes = 0

    try:
        for run in range(1, args.runs + 1):
            environment.reset()
            actions_executed = 0

            for action_index in action_indices:
                environment.step(action_index)
                actions_executed += 1
                if args.stop_on_success and starter_selected(environment):
                    break

            success = starter_selected(environment)
            successes += int(success)
            print(
                f"run={run:02d} "
                f"success={success} "
                f"actions={actions_executed} "
                f"species={starter_species(environment)}"
            )
    finally:
        environment.close()

    success_rate = successes / args.runs
    print(f"successes={successes}/{args.runs} rate={success_rate:.1%}")

    if successes == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
