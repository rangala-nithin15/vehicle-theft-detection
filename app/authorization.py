"""Prototype authorization store mapping vehicles to authorized person identities.

NOTE FOR AUDITORS & DEVELOPERS:
In live CCTV systems, tracker IDs (e.g. Person #4) reset between camera sessions.
This module defines the abstract authorization interface. In future phases,
a feature-based biometric or Appearance Re-ID module can provide stable person
identities that feed directly into this store.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import yaml


class AuthorizationStore:
    """Manages authorization rules linking vehicles to approved persons."""

    def __init__(
        self,
        rules: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the authorization store.
        
        Args:
            rules: Dictionary mapping vehicle IDs to authorized person lists.
        """
        # Map: vehicle_key (str) -> set of authorized_person_keys (str)
        self.vehicle_authorizations: Dict[str, Set[str]] = {}

        if rules:
            self.load_rules(rules)

    def load_rules(self, rules: Dict[str, Any]) -> None:
        """Populate authorization mappings from a dictionary."""
        vehicles_data = rules.get("vehicles", {})
        for v_key, data in vehicles_data.items():
            auth_list = data.get("authorized_persons", [])
            # Store keys normalized as strings for robust matching
            self.vehicle_authorizations[str(v_key)] = {str(p) for p in auth_list}

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "AuthorizationStore":
        """Instantiate store from a YAML configuration file."""
        instance = cls()
        if not yaml_path.exists():
            return instance

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        auth_data = data.get("authorization", {})
        instance.load_rules(auth_data)
        return instance

    def is_authorized(
        self,
        person_id: Optional[Union[int, str]],
        vehicle_id: Union[int, str],
    ) -> bool:
        """Check whether a person is authorized to access a vehicle.
        
        Args:
            person_id: Track ID or identity token of the person.
            vehicle_id: Track ID or registration token of the vehicle.
            
        Returns:
            True if person is explicitly listed as authorized for the vehicle.
            False if person is unknown, unauthorized, or None.
        """
        if person_id is None:
            return False

        v_key = str(vehicle_id)
        p_key = str(person_id)

        # Check vehicle-specific authorized list
        if v_key in self.vehicle_authorizations:
            return p_key in self.vehicle_authorizations[v_key]

        # Check wildcard vehicle entry "*" if configured
        if "*" in self.vehicle_authorizations:
            return p_key in self.vehicle_authorizations["*"]

        # If vehicle is not registered in authorized list, treat interaction as unknown/unauthorized
        return False

    def get_identity_label(
        self,
        person_id: Optional[Union[int, str]],
        vehicle_id: Union[int, str],
    ) -> str:
        """Return human-readable authorization status."""
        if person_id is None:
            return "No Person Near"
        if self.is_authorized(person_id, vehicle_id):
            return f"Authorized Person (#{person_id})"
        return f"Unknown Person (#{person_id})"
