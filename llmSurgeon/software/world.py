"""
Which scenario world to run. Set the WORLD env var:

    WORLD=incident   (default)  on-call alerts: wait / page / restart
    WORLD=poker                 flop spots:    fold / call / raise

Everything downstream of this file only ever calls three methods on a scenario --
describe(), best_action(), regret(action) -- so both worlds are genuine drop-ins and
nothing else in the project needs to know which one is loaded.
"""

import os
import random


class World:
    def __init__(self, name, actions, make, module):
        self.name = name
        self.actions = actions
        self._make = make
        self.module = module

    def make(self, rng=None):
        return self._make(rng or random)

    def batch(self, n, seed):
        rng = random.Random(seed)
        return [self._make(rng) for _ in range(n)]

    def __repr__(self):
        return f"<World {self.name}: {'/'.join(self.actions)}>"


def load(name=None) -> World:
    name = (name or os.environ.get("WORLD", "incident")).strip().lower()
    if name in ("poker", "cards", "holdem"):
        import poker as m
        return World("poker", m.ACTIONS, m.deal_spot, m)
    if name in ("incident", "oncall", "on-call", "incidentio", "incident.io"):
        import incident as m
        return World("incident", m.ACTIONS, m.make_incident, m)
    raise ValueError(f"unknown WORLD={name!r}; use 'incident' or 'poker'")
