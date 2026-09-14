"""Flight itinerary environment.

The index is not imported here. ``python -m itinerary_rl.index`` must be able
to run that module as a script, and importing it during package load prevents that.
"""

from itinerary_rl.loader import REQUIRED_COLUMNS, FlightRecord, LoadError, iter_flights

__all__ = ["REQUIRED_COLUMNS", "FlightRecord", "LoadError", "iter_flights"]
