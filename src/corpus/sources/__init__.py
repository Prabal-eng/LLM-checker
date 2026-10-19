"""Registry of corpus sources. Each module exposes SPEC and load(raw_dir, cap)."""

from . import (aita2026, arena140k, beemo, dactyl, detector_bias, dolly, hc3, lede, legacy, maga, mage, mirage,
               no_robots, oasst1, openturingbench, raid, rasbt50k, stackexchange, stackexchange_heldout, wasitai2026)

SOURCES = {m.SPEC.name: m for m in (mage, hc3, raid, beemo, mirage, detector_bias, legacy,
                                   rasbt50k, aita2026, no_robots, dolly, oasst1, stackexchange,
                                   stackexchange_heldout, wasitai2026, lede, maga, dactyl, arena140k,
                                   openturingbench)}
