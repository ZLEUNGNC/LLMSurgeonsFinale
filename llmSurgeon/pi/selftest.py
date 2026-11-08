"""
Test each component on its own, before anything is integrated. Use this to find the real
calibration numbers for config.py.

    python selftest.py --all               # everything in sequence
    python selftest.py --servo 1           # park servo 1 at 0, 90, 180
    python selftest.py --servo 1 --sweep   # slow full sweep: find SERVO_MIN_US / MAX_US
    python selftest.py --servo 1 --us 1500 # hold one exact pulse width
    python selftest.py --led 2
    python selftest.py --pots              # live pot readings, to find POT_INVERT
    python selftest.py --buttons
    python selftest.py --dither 3          # see what DITHER_MAX_DEG / HZ look like

If a servo buzzes and strains at an end, SERVO_MIN_US or SERVO_MAX_US is outside what that
unit can reach. Narrow it. A stalled servo browns out the Pi and cooks its own gears.
"""

import argparse
import time

import config as C
import hardware as HW


def test_servo(aid, sweep=False, us=None):
    s = HW.Servos()
    a = C.agent_by_id(aid)
    if a is None:
        print(f"no agent {aid} in config.AGENTS")
        return
    print(f"servo for agent {aid} ({a['name']}) on BCM {a['servo_pin']}, "
          f"backend={s.backend}")
    try:
        if us is not None:
            print(f"holding {us}us -- note the angle, then ctrl-c")
            if s.backend == "pigpio" and s._impl is not None:
                s._impl.set_servo_pulsewidth(a["servo_pin"], int(us))
            while True:
                time.sleep(0.2)
        if sweep:
            print("sweeping 0 -> 180 -> 0 slowly. Watch for straining at the ends.")
            for ang in list(range(0, 181, 2)) + list(range(180, -1, -2)):
                s.set(aid, ang)
                print(f"\r  {ang:3d} deg   "
                      f"{C.SERVO_MIN_US + ang / 180 * (C.SERVO_MAX_US - C.SERVO_MIN_US):6.0f}us",
                      end="", flush=True)
                time.sleep(0.05)
            print()
        else:
            for ang in (0, 90, 180, 90):
                print(f"  -> {ang} deg")
                s.set(aid, ang)
                time.sleep(1.2)
    except KeyboardInterrupt:
        pass
    finally:
        s.close()


def test_led(aid):
    l = HW.Leds()
    print(f"LED agent {aid} on BCM {C.agent_by_id(aid)['led_pin']} -- 5 blinks")
    try:
        for _ in range(5):
            l.set(aid, True)
            time.sleep(0.3)
            l.set(aid, False)
            time.sleep(0.3)
    finally:
        l.close()


def test_pots(seconds=30):
    p = HW.Pots()
    print(f"backend={p.backend}. Turn each knob fully both ways. ctrl-c to stop.")
    print("A knob that reads +1 when turned anticlockwise needs POT_INVERT set for it.\n")
    seen = {a["id"]: [1e9, -1e9] for a in C.AGENTS}
    t0 = time.time()
    try:
        while time.time() - t0 < seconds:
            vals = p.read_all()
            parts = []
            for a in C.AGENTS:
                v = vals[a["id"]]
                seen[a["id"]][0] = min(seen[a["id"]][0], v)
                seen[a["id"]][1] = max(seen[a["id"]][1], v)
                parts.append(f"{a['name']}={v:+.2f}")
            print("\r  " + "   ".join(parts) + "    ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nrange reached per knob (want about -1.00 to +1.00):")
        for a in C.AGENTS:
            lo, hi = seen[a["id"]]
            note = ""
            if hi - lo < 1.0:
                note = "  <- barely moved: check wiring / pot_channel"
            print(f"  {a['name']:6s} {lo:+.2f} .. {hi:+.2f}{note}")
        p.close()


def test_buttons(seconds=30):
    b = HW.Buttons()
    if not b.enabled:
        print("buttons unavailable")
        return
    print("press each button. ctrl-c to stop.")
    t0 = time.time()
    try:
        while time.time() - t0 < seconds:
            for aid in b.pressed():
                print(f"  agent {aid} ({C.agent_by_id(aid)['name']}) pressed")
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        b.close()


def test_dither(aid, seconds=20):
    s = HW.Servos()
    v = HW.Vibration(s)
    print(f"dither demo on agent {aid}: jitter ramps 0->1, urgency ramps 0->1.")
    print(f"mode={v.mode} max_deg={C.DITHER_MAX_DEG} hz={C.DITHER_MIN_HZ}-{C.DITHER_MAX_HZ}")
    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            if t > seconds:
                break
            frac = (t % seconds) / seconds
            off = v.offset(frac, frac, t)
            s.set(aid, 90 + off)
            v.set_motor(aid, frac)
            print(f"\r  jitter={frac:.2f} urgency={frac:.2f} offset={off:+5.1f}deg  ",
                  end="", flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        v.close()
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--servo", type=int)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--us", type=int, help="hold an exact pulse width, with --servo")
    ap.add_argument("--led", type=int)
    ap.add_argument("--pots", action="store_true")
    ap.add_argument("--buttons", action="store_true")
    ap.add_argument("--dither", type=int)
    a = ap.parse_args()

    print(HW.describe() + "\n")

    if a.servo is not None:
        test_servo(a.servo, a.sweep, a.us)
    elif a.led is not None:
        test_led(a.led)
    elif a.pots:
        test_pots()
    elif a.buttons:
        test_buttons()
    elif a.dither is not None:
        test_dither(a.dither)
    elif a.all:
        for ag in C.AGENTS:
            test_servo(ag["id"])
            test_led(ag["id"])
        test_pots(15)
        test_buttons(15)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
