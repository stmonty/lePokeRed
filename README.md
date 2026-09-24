# lePokeRed

A small JEPA-style world model for Pokémon Red.

## Requirements

- Python 3.13+ and [uv](https://docs.astral.sh/uv/)
- **Your own legally acquired Pokémon Red ROM**, saved as `pokered.gb`. No ROM is included or distributed here.
- A graphical desktop to create the save state; a CUDA GPU is recommended for training.

## Run

Run commands from the repository root.

```bash
uv sync --locked

# Play until you can choose a starter. Stand below the middle Poké Ball,
# facing it, with no dialogue open and no Pokémon in your party.
# Close the window to save.
uv run python -m scripts.capture_state pokered.gb --out states/pokeballs.state

# Collect gameplay. All three named starter routes should report goal=True.
uv run python -m scripts.collect --random 800 --random-steps 48 --noisy 200

# Train, probe, and fine-tune rollouts.
uv run python -m scripts.train
uv run python -m scripts.probe checkpoints/best.pt
uv run python -m scripts.finetune_rollout checkpoints/best.pt

# Plan and execute in the emulator.
uv run python -m scripts.plan_cem checkpoints/rollout/best.pt --out plans/poke-rollout.json
uv run python -m scripts.eval_plan --plan plans/poke-rollout.json --runs 1 --show
```

Base training settings are in `DefaultConfig` in `scripts/train.py`. Other commands expose options through `--help`. Runs can overwrite existing datasets and checkpoints. Only load checkpoints you trust.

## Acknowledgements

- [LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://arxiv.org/abs/2603.19312) — Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero (2026).
- [LeJEPA](https://arxiv.org/abs/2511.08544), which introduces SIGReg.
- [LeMario](https://github.com/benyebai/LeMario) by benyebai, for inspiration and code I studied while building this project.

## License

[MIT](LICENSE). This license covers the project's original code, not Pokémon ROMs or game assets.
