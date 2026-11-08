"""
Simplified heads-up poker spot: one decision per hand, scored by expected value (EV).

Setup:
  - Hero has 2 hole cards, the flop (3 board cards) is out.
  - Pot before the opponent acts = POT. Opponent bets BET. Hero has STACK chips left.
  - Hero chooses: fold, call, or raise (= all-in, assume opponent calls).

EV (chips gained from this point; money already in the pot is sunk):
  EV(fold)  = 0
  EV(call)  = eq * (POT + 2*BET)   - BET
  EV(raise) = eq * (POT + 2*STACK) - STACK

Consequence (worth knowing for the demo):
  fold  if equity < pot odds  (BET / (POT + 2*BET))
  raise if equity > 0.5
  call  otherwise
"""

import random
from dataclasses import dataclass
from treys import Card, Deck, Evaluator

ACTIONS = ["fold", "call", "raise"]
_evaluator = Evaluator()


@dataclass
class Spot:
    hole: list          # treys ints
    board: list         # treys ints (3 cards = flop)
    pot: int
    bet: int
    stack: int
    equity: float = 0.0

    # ---------- scoring ----------
    def ev(self, action: str) -> float:
        if action == "fold":
            return 0.0
        if action == "call":
            return self.equity * (self.pot + 2 * self.bet) - self.bet
        if action == "raise":
            return self.equity * (self.pot + 2 * self.stack) - self.stack
        raise ValueError(action)

    def best_action(self) -> str:
        return max(ACTIONS, key=self.ev)

    def regret(self, action: str) -> float:
        """How many chips worse than the best action. 0 = optimal."""
        return self.ev(self.best_action()) - self.ev(action)

    @property
    def pot_odds(self) -> float:
        return self.bet / (self.pot + 2 * self.bet)

    def urgency(self) -> float:
        """0-1, how much this situation demands acting. Drives vibration FREQUENCY on the
        panel. Mirrors Incident.urgency() so both worlds feed the hardware identically."""
        return max(0.0, min(1.0, self.equity))

    # ---------- text for the LLM ----------
    def describe(self, show_equity: bool = True) -> str:
        hint = (f"Your estimated chance of winning: {round(self.equity*100)}%. "
                f"You need at least {round(self.pot_odds*100)}% to make calling worthwhile.\n") if show_equity else ""
        return (
            f"Your hand: {cards_str(self.hole)}\n"
            f"Board (flop): {cards_str(self.board)}\n"
            f"Pot: {self.pot} chips. Your opponent bets {self.bet} chips.\n"
            f"You have {self.stack} chips left.\n"
            + hint +
            f"Options: fold, call ({self.bet} chips), or raise (go all-in for {self.stack} chips)."
        )


def cards_str(cards) -> str:
    # treys prints e.g. "Ks" -> make it readable: "K♠"
    suits = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
    out = []
    for c in cards:
        s = Card.int_to_str(c)
        rank = "10" if s[0] == "T" else s[0]
        out.append(rank + suits[s[1]])
    return " ".join(out)


def equity(hole, board, iters=2000, rng=None) -> float:
    """Monte Carlo win probability vs a random opponent hand, running out turn + river."""
    rng = rng or random
    used = set(hole + board)
    deck = [c for c in Deck.GetFullDeck() if c not in used]
    wins = 0.0
    for _ in range(iters):
        draw = rng.sample(deck, 4)          # 2 opp cards + turn + river
        opp, runout = draw[:2], draw[2:]
        full = board + runout
        h = _evaluator.evaluate(full, hole)  # lower = better in treys
        o = _evaluator.evaluate(full, opp)
        wins += 1.0 if h < o else 0.5 if h == o else 0.0
    return wins / iters


def deal_spot(rng=None, pot=100, stack=400) -> Spot:
    rng = rng or random
    deck = Deck.GetFullDeck()[:]
    rng.shuffle(deck)
    hole, board = deck[:2], deck[2:5]
    bet = rng.choice([50, 100, 150, 200])   # varied bet sizes -> best action roughly 1/3 each
    spot = Spot(hole=hole, board=board, pot=pot, bet=bet, stack=stack)
    spot.equity = equity(hole, board, rng=rng)
    return spot


if __name__ == "__main__":
    rng = random.Random(0)
    for _ in range(5):
        s = deal_spot(rng)
        print(s.describe())
        print(f"  equity={s.equity:.2f}  pot_odds={s.pot_odds:.2f}  "
              + "  ".join(f"EV({a})={s.ev(a):+.1f}" for a in ACTIONS)
              + f"  -> best: {s.best_action()}\n")
