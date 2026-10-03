"""Long-term user facts manager for Finch.ai."""
from typing import Any, Dict, List, Optional, Protocol


class FactStorageProtocol(Protocol):
    """Protocol for storage engines supporting user facts."""
    def add_fact(self, fact: str, category: str = "general") -> int: ...
    def list_facts(self, category: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]: ...
    def delete_fact(self, fact_id: int) -> bool: ...
    def wipe_all_facts(self) -> int: ...
    def import_facts(self, facts: List[Dict[str, Any]]) -> int: ...


class FactsManager:
    """Manages storing, retrieving, deleting, and injecting long-term verified user facts and preferences."""

    def __init__(self, storage: Optional[FactStorageProtocol] = None):
        self.storage = storage

    def set_storage(self, storage: FactStorageProtocol) -> None:
        """Bind or update the underlying storage engine."""
        self.storage = storage

    def add_fact(self, fact: str, category: str = "general") -> int:
        """Store a verified fact in long-term memory.
        
        Args:
            fact: Fact statement.
            category: Optional classification tag (e.g. 'preference', 'tech_stack', 'project').
            
        Returns:
            The integer ID of the inserted fact.
        """
        if not self.storage:
            raise RuntimeError("FactsManager has no active storage backend bound.")
        clean_fact = fact.strip()
        if not clean_fact:
            raise ValueError("Fact content cannot be empty.")
        return self.storage.add_fact(fact=clean_fact, category=category.strip() or "general")

    def list_facts(self, category: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve verified user facts from storage."""
        if not self.storage:
            return []
        return self.storage.list_facts(category=category, limit=limit)

    def delete_fact(self, fact_id: int) -> bool:
        """Remove a fact by its identifier."""
        if not self.storage:
            return False
        return self.storage.delete_fact(fact_id)

    def wipe_all_facts(self) -> int:
        """Wipe all facts from long-term memory."""
        if not self.storage:
            return 0
        return self.storage.wipe_all_facts()

    def import_facts(self, facts: List[Dict[str, Any]]) -> int:
        """Import a batch of facts into memory."""
        if not self.storage:
            return 0
        return self.storage.import_facts(facts)

    def format_facts_for_prompt(self, category: Optional[str] = None) -> str:
        """Format verified user facts into a markdown section suitable for system prompt injection."""
        facts = self.list_facts(category=category)
        if not facts:
            return ""

        lines = [
            "---",
            "### Verified User Facts & Long-Term Memory",
            "The following verified facts and preferences about the user and their environment are stored in long-term memory:",
        ]
        for f in facts:
            cat = f.get("category", "general")
            lines.append(f"- [Fact #{f['id']}] ({cat}): {f['fact']}")

        return "\n".join(lines)
