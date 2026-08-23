from poke.env import PokeEnv

PARTY_COUNT = 0xD163      # wPartyCount:  number of pokemon in party (0..6)
PARTY_SPECIES = 0xD164    # wPartySpecies: internal index of slot 0 (Squirtle = 177)

# sentinel the game stores in an empty species slot
EMPTY_SPECIES = 0xFF

def party_count(env: PokeEnv) -> int:
    return env.read_ram(PARTY_COUNT)


def starter_selected(env: PokeEnv) -> bool:
    return party_count(env) >= 1

def starter_species(env: PokeEnv) -> int | None:
    species = env.read_ram(PARTY_SPECIES)
    return None if species == EMPTY_SPECIES else species
