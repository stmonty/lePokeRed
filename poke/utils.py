import argparse

def default_parser(rom=True, state=True, out=None):
    p = argparse.ArgumentParser()
    if rom:
        p.add_argument("rom")
    if state:
        p.add_argument("state")
    if out is not None:
        p.add_argument("--out", default=out)
    return p
