"""Spatial zone map for structured location transitions."""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ZoneExit:
    """An exit from a zone leading to another zone."""

    target: str  # target zone id
    direction: str  # e.g. "north", "east", "deeper"
    description: str  # e.g. "A narrow tunnel leads deeper into the cave"


@dataclass
class Zone:
    """A distinct area in the game world."""

    id: str
    name: str
    description: str
    exits: list[ZoneExit] = field(default_factory=list)


class ZoneMap:
    """Manages the spatial zone graph built from config."""

    def __init__(self, zone_configs: list[dict]):
        self.zones: dict[str, Zone] = {}
        self._name_to_id: dict[str, str] = {}

        for zc in zone_configs:
            exits = [
                ZoneExit(
                    target=e["target"],
                    direction=e["direction"],
                    description=e.get("description", ""),
                )
                for e in zc.get("exits", [])
            ]
            zone = Zone(
                id=zc["id"],
                name=zc["name"],
                description=zc.get("description", ""),
                exits=exits,
            )
            self.zones[zone.id] = zone
            self._name_to_id[zone.name.lower()] = zone.id

        logger.info(f"ZoneMap loaded with {len(self.zones)} zones")

    def get_zone(self, zone_id: str) -> Zone | None:
        return self.zones.get(zone_id)

    def get_zone_by_name(self, name: str) -> Zone | None:
        zone_id = self._name_to_id.get(name.lower())
        if zone_id:
            return self.zones.get(zone_id)
        return None

    def get_exit_target(self, zone_id: str, direction: str) -> Zone | None:
        """Get the zone reached by going in a direction from a zone."""
        zone = self.get_zone(zone_id)
        if not zone:
            return None
        direction_lower = direction.lower()
        for exit_ in zone.exits:
            if exit_.direction.lower() == direction_lower:
                return self.get_zone(exit_.target)
        return None

    def get_exit_names(self, zone_id: str) -> list[str]:
        """Get human-readable exit descriptions for a zone."""
        zone = self.get_zone(zone_id)
        if not zone:
            return []
        return [
            f"{e.direction}: {e.description}" for e in zone.exits
        ]

    def resolve_zone_id(self, display_name: str) -> str | None:
        """Resolve a display name to a zone ID."""
        return self._name_to_id.get(display_name.lower())
