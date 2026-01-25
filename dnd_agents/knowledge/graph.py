"""NetworkX-based knowledge graph for game state."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""
    CHARACTER = "character"
    LOCATION = "location"
    ITEM = "item"
    CREATURE = "creature"
    EVENT = "event"
    FACTION = "faction"


class RelationType(str, Enum):
    """Types of relationships between entities."""
    LOCATED_AT = "located_at"
    OWNS = "owns"
    ALLIED_WITH = "allied_with"
    HOSTILE_TO = "hostile_to"
    KNOWS = "knows"
    PART_OF = "part_of"
    CONNECTED_TO = "connected_to"
    PARTICIPATED_IN = "participated_in"


@dataclass
class Entity:
    """An entity in the knowledge graph."""

    id: str
    name: str
    entity_type: EntityType
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=EntityType(data["entity_type"]),
            properties=data.get("properties", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


@dataclass
class Relationship:
    """A relationship between two entities."""

    source_id: str
    target_id: str
    relation_type: RelationType
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Relationship":
        """Create from dictionary."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            properties=data.get("properties", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


class KnowledgeGraph:
    """Knowledge graph for tracking game world state."""

    def __init__(self):
        """Initialize empty knowledge graph."""
        self.graph = nx.DiGraph()
        self.entities: dict[str, Entity] = {}

    def add_entity(self, entity: Entity) -> None:
        """Add or update an entity."""
        self.entities[entity.id] = entity
        self.graph.add_node(
            entity.id,
            name=entity.name,
            entity_type=entity.entity_type.value,
            properties=entity.properties
        )
        logger.debug(f"Added entity: {entity.name} ({entity.entity_type.value})")

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        return self.entities.get(entity_id)

    def find_entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        """Find all entities of a given type."""
        return [
            e for e in self.entities.values()
            if e.entity_type == entity_type
        ]

    def find_entities_by_name(self, name: str) -> list[Entity]:
        """Find entities by name (case-insensitive partial match)."""
        name_lower = name.lower()
        return [
            e for e in self.entities.values()
            if name_lower in e.name.lower()
        ]

    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship between entities."""
        if relationship.source_id not in self.entities:
            logger.warning(f"Source entity not found: {relationship.source_id}")
            return
        if relationship.target_id not in self.entities:
            logger.warning(f"Target entity not found: {relationship.target_id}")
            return

        self.graph.add_edge(
            relationship.source_id,
            relationship.target_id,
            relation_type=relationship.relation_type.value,
            properties=relationship.properties
        )
        logger.debug(
            f"Added relationship: {relationship.source_id} "
            f"-[{relationship.relation_type.value}]-> {relationship.target_id}"
        )

    def get_relationships(
        self,
        entity_id: str,
        relation_type: RelationType | None = None,
        direction: str = "both"
    ) -> list[Relationship]:
        """
        Get relationships for an entity.

        Args:
            entity_id: Entity to get relationships for
            relation_type: Filter by relationship type
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of relationships
        """
        relationships = []

        if direction in ("outgoing", "both"):
            for _, target, data in self.graph.out_edges(entity_id, data=True):
                rel = Relationship(
                    source_id=entity_id,
                    target_id=target,
                    relation_type=RelationType(data["relation_type"]),
                    properties=data.get("properties", {})
                )
                if relation_type is None or rel.relation_type == relation_type:
                    relationships.append(rel)

        if direction in ("incoming", "both"):
            for source, _, data in self.graph.in_edges(entity_id, data=True):
                rel = Relationship(
                    source_id=source,
                    target_id=entity_id,
                    relation_type=RelationType(data["relation_type"]),
                    properties=data.get("properties", {})
                )
                if relation_type is None or rel.relation_type == relation_type:
                    relationships.append(rel)

        return relationships

    def get_entities_at_location(self, location_id: str) -> list[Entity]:
        """Get all entities at a location."""
        entities = []
        for source, target, data in self.graph.in_edges(location_id, data=True):
            if data.get("relation_type") == RelationType.LOCATED_AT.value:
                entity = self.entities.get(source)
                if entity:
                    entities.append(entity)
        return entities

    def update_entity_property(
        self,
        entity_id: str,
        key: str,
        value: Any
    ) -> bool:
        """Update a property on an entity."""
        entity = self.entities.get(entity_id)
        if not entity:
            return False

        entity.properties[key] = value
        entity.updated_at = datetime.now()
        self.graph.nodes[entity_id]["properties"] = entity.properties
        return True

    def get_relevant_facts(
        self,
        entity_ids: list[str],
        max_facts: int = 10
    ) -> list[str]:
        """
        Get relevant facts about entities for context.

        Returns human-readable fact strings.
        """
        facts = []

        for entity_id in entity_ids:
            entity = self.entities.get(entity_id)
            if not entity:
                continue

            # Add entity description
            props = entity.properties
            if entity.entity_type == EntityType.CHARACTER:
                hp = props.get("current_hp", "?")
                max_hp = props.get("max_hp", "?")
                facts.append(f"{entity.name} has {hp}/{max_hp} HP")
                if props.get("conditions"):
                    facts.append(
                        f"{entity.name} is {', '.join(props['conditions'])}"
                    )

            # Add relationships
            for rel in self.get_relationships(entity_id, direction="outgoing"):
                target = self.entities.get(rel.target_id)
                if target:
                    facts.append(
                        f"{entity.name} {rel.relation_type.value.replace('_', ' ')} "
                        f"{target.name}"
                    )

            if len(facts) >= max_facts:
                break

        return facts[:max_facts]

    def save(self, path: Path | str) -> None:
        """Save knowledge graph to JSON file."""
        path = Path(path)
        data = {
            "entities": [e.to_dict() for e in self.entities.values()],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "data": d
                }
                for u, v, d in self.graph.edges(data=True)
            ]
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved knowledge graph to {path}")

    def load(self, path: Path | str) -> None:
        """Load knowledge graph from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())

        self.graph.clear()
        self.entities.clear()

        for entity_data in data.get("entities", []):
            entity = Entity.from_dict(entity_data)
            self.add_entity(entity)

        for edge in data.get("edges", []):
            self.graph.add_edge(
                edge["source"],
                edge["target"],
                **edge["data"]
            )

        logger.info(
            f"Loaded knowledge graph from {path}: "
            f"{len(self.entities)} entities, {self.graph.number_of_edges()} edges"
        )
