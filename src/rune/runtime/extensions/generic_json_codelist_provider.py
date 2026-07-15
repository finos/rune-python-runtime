import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import TypeVar, Type, Generic

logger = logging.getLogger(__name__)

# Define a TypeVar to allow autocomplete and strict type safety for whatever Rosetta model class is passed
T = TypeVar('T')

class GenericJSONCodelistProvider(Generic[T]):
    def __init__(self, codelist_dir: str | Path, target_class: Type[T]):

        # Fail fast: Validate the directory immediately
        self.codelist_dir = Path(codelist_dir)
        if not self.codelist_dir.is_dir():
            raise NotADirectoryError(f"Invalid codelist_dir: '{self.codelist_dir}' is not a directory or does not exist.")
            
        # Fail fast: Validate the target class has the required Rune deserializer interface
        if not hasattr(target_class, "rune_deserialize") or not callable(getattr(target_class, "rune_deserialize")):
            raise TypeError(f"Invalid target_class: '{target_class.__name__}' must implement a callable 'rune_deserialize' method.")
        
        self.target_class = target_class
        
        # Bind the LRU cache strictly to this instance context.
        # This guarantees 100% process isolation between models and prevents class-level memory leaks.
        self.load = lru_cache(maxsize=15)(self._load)

    def clear_cache(self):
        self.load.cache_clear()

    def _load(self, domain: str) -> T:
        """
        Internal target for file loading and deserialization.
        Invoked implicitly by the instance's bound self.load() wrapper.
        """
        logger.debug(f"Cache miss: Loading {self.target_class.__name__} for domain '{domain}' from disk context.")
        
        target_file: Path | None = None

        # Dynamically build a strict regex looking exactly for "domain-X-Y.json"
        pattern = re.compile(rf"^{re.escape(domain.lower())}-\d+-\d+\.json$")
        
        # Search the local directory for matching items
        try:
            for file_path in self.codelist_dir.iterdir():
                if file_path.is_file() and file_path.suffix == ".json" and pattern.match(file_path.name):
                    target_file = file_path
                    break
        except OSError as e:
            logger.error(f"Failed to scan directory {self.codelist_dir}: {e}")
            raise

        # Off-Flow Vector: Handle missing resources explicitly
        if not target_file:
            logger.error(f"Domain '{domain}' not found in target path: {self.codelist_dir}")
            raise FileNotFoundError(f"Could not find CodeList JSON for domain: '{domain}'")
            
        # Read the raw serialization content from the file stream
        try:
            with target_file.open('r', encoding='utf-8') as f:
                json_raw_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON syntax in {target_file.name}: {e}")
            raise 
            
        # Off-Flow Vector: Hand execution directly over to the underlying Rosetta deserializer
        try:
            # Execute the auto-generated deserialization framework method
            cdm_object = self.target_class.rune_deserialize(json_raw_data) # type: ignore[attr-defined]
            logger.debug(f"Successfully deserialized model for domain tracking: '{domain}'.")
            return cdm_object
        except Exception as e:
            logger.error(f"Rune deserialization framework failure for domain '{domain}': {e}")
            raise


    