import numpy as np

ACTIONS = ["noop", "up", "down", "left", "right", "a", "b"]

_BUTTONS = {"up": "up", "down": "down", "left": "left", "right": "right", "a": "a", "b": "b"}

BUTTON_ORDER = ["up", "down", "left", "right", "a", "b"]

def n_actions() -> int:
    return len(ACTIONS)

def to_button(idx: int) -> str | None:
    return _BUTTONS.get(ACTIONS[idx], None)

def to_multihot(idx: int):
    multihot = np.zeros(len(BUTTON_ORDER), dtype=np.float32)
    button = to_button(idx)

    if button is not None:
        multihot[BUTTON_ORDER.index(button)] = 1.0
    return multihot
    

