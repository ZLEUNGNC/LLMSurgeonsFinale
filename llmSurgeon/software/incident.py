"""
Incident-response scenarios: the "quiz master". No ML in this file.

Generates an alert, computes the correct answer with real maths, and hands the model a
paragraph to read. Replaces poker.py (which still works, same interface).

Two independent quantities decide the right answer, which is what makes the task more than
"read the number we gave you":

  p      = probability this alert is a REAL incident (Bayes, from the alert's features)
  blast  = how much damage an unattended real incident does (from users affected)

Actions:
  wait     keep monitoring, touch nothing
  page     wake the on-call engineer
  restart  restart the service now

Cost model (EV = negative expected damage, so higher is better):

  EV(wait)    = -p * BLAST
  EV(page)    = -C_PAGE            - p * BLAST * R_PAGE
  EV(restart) = -C_RESTART         - p * BLAST * R_RESTART - (1 - p) * C_BLIP

R_PAGE > R_RESTART because a human takes minutes to respond while a restart is immediate.
C_BLIP is the cost of restarting a service that was actually healthy.

Consequence: for a given blast radius, `wait` is right at low p, `page` in the middle,
`restart` at high p -- the same three-way monotonic structure poker had. A small blast
radius makes `wait` right almost everywhere, which is correct and is the trap the twitchy
agent falls into.

The interface agent.py depends on is exactly three methods: describe(), best_action(),
regret(action). Anything providing those three is a drop-in replacement for this file.
"""

import math
import random
from dataclasses import dataclass, field

ACTIONS = ["wait", "page", "restart"]

# ---------------------------------------------------------------- cost constants
# Tuned by grid search so the correct answer is ~1/3 wait, 1/3 page, 1/3 restart over the
# generated distribution (33/34/34 measured). A lopsided mix would let an agent score well by
# always picking the same action, which would make the LED meaningless.
C_PAGE = 20.0        # waking a human at 3am
C_RESTART = 120.0    # dropped in-flight requests, cache cold start
C_BLIP = 400.0       # restarting a service that was actually fine
R_PAGE = 0.50        # fraction of the outage that still burns while the human responds
R_RESTART = 0.05     # restart is immediate

PRIOR = 0.35         # base rate that an alert of this kind is a real incident
BLAST_DIV = 6.0      # damage of an unattended real incident = users / BLAST_DIV

SERVICES = [
    "checkout-api", "payments-worker", "auth-gateway", "search-indexer",
    "notification-relay", "cart-service", "media-transcoder", "ledger-sync",
]


def _lr_error_rate(er: float) -> float:
    if er < 0.01:
        return 0.30
    if er < 0.05:
        return 1.50
    if er < 0.15:
        return 4.00
    return 9.00


def _lr_deploy(minutes: float) -> float:
    if minutes < 10:
        return 4.00
    if minutes < 60:
        return 1.50
    return 0.70


def _lr_users(users: int) -> float:
    if users < 500:
        return 0.50
    if users < 5000:
        return 1.20
    return 2.50


@dataclass
class Incident:
    service: str
    error_rate: float          # 0..1
    users: int                 # users affected
    deploy_min: float          # minutes since last deploy
    flapping: bool             # alert has fired and cleared repeatedly
    dep_alerting: bool         # an upstream dependency is also alerting
    p: float = 0.0             # posterior probability this is real
    blast: float = 0.0         # damage of an unattended real incident
    evidence: list = field(default_factory=list)

    # ---------------- scoring ----------------
    def ev(self, action: str) -> float:
        if action == "wait":
            return -self.p * self.blast
        if action == "page":
            return -C_PAGE - self.p * self.blast * R_PAGE
        if action == "restart":
            return (-C_RESTART - self.p * self.blast * R_RESTART
                    - (1.0 - self.p) * C_BLIP)
        raise ValueError(action)

    def best_action(self) -> str:
        return max(ACTIONS, key=self.ev)

    def regret(self, action: str) -> float:
        """How much worse than the best action, in damage units. 0 = optimal."""
        return self.ev(self.best_action()) - self.ev(action)

    # ---------------- decision thresholds ----------------
    def thresholds(self):
        """The two probabilities at which the correct answer changes, derived from the cost
        model above:

            page beats wait      when p > C_PAGE / (blast * (1 - R_PAGE))
            restart beats page   when p > (C_RESTART + C_BLIP - C_PAGE)
                                         / (blast * (R_PAGE - R_RESTART) + C_BLIP)

        If the first exceeds the second then `page` is never optimal for this blast radius,
        and the only crossover is wait -> restart. Returns (t_act, t_restart) where t_act may
        equal t_restart in that degenerate case.

        We hand these to the model the same way poker.py hands over pot odds: a 1.5B cannot do
        a three-way expected-value comparison, and making it try measures its arithmetic
        rather than the thing we are actually demonstrating, which is the knob.
        """
        b = self.blast
        t1 = C_PAGE / (b * (1 - R_PAGE)) if b > 0 else 1.0
        t2 = ((C_RESTART + C_BLIP - C_PAGE)
              / (b * (R_PAGE - R_RESTART) + C_BLIP))
        if t1 >= t2:
            # page is dominated; single crossover from wait straight to restart
            t = (C_RESTART + C_BLIP) / (b * (1 - R_RESTART) + C_BLIP)
            return min(t, 1.0), min(t, 1.0)
        return min(t1, 1.0), min(t2, 1.0)

    # ---------------- text for the LLM ----------------
    def describe(self, show_p: bool = True) -> str:
        flap = "yes, it has fired and cleared 6 times in 20 minutes" if self.flapping else "no"
        dep = "yes" if self.dep_alerting else "no"
        hint = ""
        if show_p:
            # Wording chosen by A/B test over 6 variants x 2 positions on 40 scenarios.
            # An explicit if/then rule placed AFTER the alert details scored best (50%);
            # the same facts phrased as "doing nothing is only worth it below X%" scored 30%
            # in that position. Terse "risk score / threshold" phrasing was worst (20-32%).
            t_act, t_restart = self.thresholds()      # compute once, not twice
            a, r = round(t_act * 100), round(t_restart * 100)
            hint = (
                f"Chance this is a real incident: {round(self.p * 100)}%.\n"
                f"Rule: if that is below {a}% choose wait; if it is between {a}% and {r}% "
                f"choose page; if it is above {r}% choose restart.\n"
            )
        return (
            f"ALERT on {self.service}\n"
            f"Error rate: {self.error_rate * 100:.1f}%\n"
            f"Users affected: {self.users}\n"
            f"Last deploy: {self.deploy_min:.0f} minutes ago\n"
            f"Alert flapping: {flap}\n"
            f"Upstream dependency also alerting: {dep}\n"
            + hint +
            "Options: wait (keep monitoring), page (wake the on-call team), "
            "or restart (restart the service now)."
        )

    def urgency(self) -> float:
        """0-1, how much this situation demands acting now. Drives vibration FREQUENCY on the
        panel, which is a free extra channel: amplitude already carries indecision."""
        return max(0.0, min(1.0, self.p))

    def readout(self) -> str:
        """One line for the console. Not the model's words -- a readout."""
        return (f"{self.service:20s} err={self.error_rate*100:4.1f}%  users={self.users:6d}  "
                f"p={self.p:.2f}  blast={self.blast:6.0f}  -> {self.best_action().upper()}")


def make_incident(rng=None) -> Incident:
    rng = rng or random

    # features
    error_rate = min(0.95, math.exp(rng.uniform(math.log(0.001), math.log(0.40))))
    users = int(math.exp(rng.uniform(math.log(80), math.log(60_000))))
    deploy_min = math.exp(rng.uniform(math.log(2), math.log(2000)))
    flapping = rng.random() < 0.30
    dep_alerting = rng.random() < 0.35

    inc = Incident(service=rng.choice(SERVICES), error_rate=error_rate, users=users,
                   deploy_min=deploy_min, flapping=flapping, dep_alerting=dep_alerting)

    # Bayes: posterior odds = prior odds * product of likelihood ratios
    odds = PRIOR / (1 - PRIOR)
    lrs = [
        ("error rate", _lr_error_rate(error_rate)),
        ("deploy recency", _lr_deploy(deploy_min)),
        ("blast radius", _lr_users(users)),
        ("flapping", 0.25 if flapping else 1.30),
        ("dependency", 3.00 if dep_alerting else 0.80),
    ]
    for _, lr in lrs:
        odds *= lr
    inc.p = odds / (1 + odds)
    inc.evidence = lrs

    # damage of an unattended real incident, in the same units as the costs above
    inc.blast = users / BLAST_DIV
    return inc


def make_incidents(n: int, seed: int) -> list:
    rng = random.Random(seed)
    return [make_incident(rng) for _ in range(n)]


if __name__ == "__main__":
    import collections, sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    rng = random.Random(0)
    mix = collections.Counter()
    for _ in range(2000):
        mix[make_incident(rng).best_action()] += 1
    print("action mix over 2000 generated alerts:",
          {a: f"{100*mix[a]/2000:.0f}%" for a in ACTIONS}, "\n")

    rng = random.Random(1)
    for _ in range(n):
        inc = make_incident(rng)
        print(inc.describe())
        print("  " + "  ".join(f"EV({a})={inc.ev(a):+8.1f}" for a in ACTIONS)
              + f"  -> best: {inc.best_action()}\n")
