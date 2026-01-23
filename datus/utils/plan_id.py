import uuid
from typing import Optional


class PlanIdManager:
    """Centralized plan id generator for consistent plan id creation."""

    @classmethod
    def new_plan_id(cls, prefix: Optional[str] = None) -> str:
        """Generate a new unique plan ID.

        Args:
            prefix: Optional prefix to prepend to the UUID

        Returns:
            A unique plan identifier string
        """
        plan_id = str(uuid.uuid4())
        return f"{prefix}_{plan_id}" if prefix else plan_id
