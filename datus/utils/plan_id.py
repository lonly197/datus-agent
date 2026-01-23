import uuid
from typing import Optional


class PlanIdManager:
    """Centralized plan id generator for consistent plan id creation."""

    @classmethod
    def new_plan_id(cls, prefix: Optional[str] = None) -> str:
        plan_id = str(uuid.uuid4())
        return f"{prefix}_{plan_id}" if prefix else plan_id
