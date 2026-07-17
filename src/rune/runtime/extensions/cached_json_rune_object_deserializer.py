import json
import logging
import importlib.util
import importlib.resources
from functools import lru_cache
from pathlib import Path
from typing import TypeVar, Type, Generic

logger = logging.getLogger(__name__)

# Define a TypeVar to allow autocomplete and strict type safety for whatever Rosetta model class is passed
T = TypeVar('T')

class CachedJSONRuneObjectDeserializer(Generic[T]):
    """
    A generic, memory-safe utility for loading and deserializing Rune objects from JSON files.
    
    This provider dynamically resolves the target resource location (supporting both standard 
    filesystem paths and zipped Python module distributions) and delegates the parsing 
    to the generated model's `rune_deserialize` method.
    
    Parameters:
    -----------
    codelist_dir : str | Path
        The filesystem path or Python module string where the JSON resources are stored.
    target_class : Type[T]
        The generated Rune class to be used for deserialization. Must implement `rune_deserialize`.
    maxsize : int, optional
        The maximum number of deserialized objects to store in the instance-bound LRU cache (default is 15).
    """
    
    def __init__(self, codelist_dir: str | Path, target_class: Type[T], maxsize: int = 15):

        # Fail fast: Validate maxsize parameter
        if not isinstance(maxsize, int) or maxsize < 1:
            raise ValueError(f"Invalid maxsize: '{maxsize}' must be a positive integer.")

        # Fail fast: Validate the target class has the required Rune deserializer interface
        if not hasattr(target_class, "rune_deserialize") or not callable(getattr(target_class, "rune_deserialize")):
            raise TypeError(f"Invalid target_class: '{target_class.__name__}' must implement a callable 'rune_deserialize' method.")
        
        self.target_class = target_class
        
        # Smart Resolution: Determine if codelist_dir is a Filesystem Path or a Python Module
        if isinstance(codelist_dir, Path) or Path(codelist_dir).is_dir():
            # It is a standard filesystem directory
            self._resource_path = Path(codelist_dir)
        else:
            # It is not a filesystem directory, attempt to locate it as a Python module
            if importlib.util.find_spec(str(codelist_dir)) is not None:
                # Valid module found. We use resources.files() to create a Traversable object.
                # This allows us to read the files even if the module is distributed as a zipped wheel
                self._resource_path = importlib.resources.files(str(codelist_dir))
            else:
                raise ValueError(f"Invalid codelist_dir: '{codelist_dir}' is neither a valid directory path nor a discoverable Python module.")
        
        # Bind the LRU cache strictly to this instance context.
        # This guarantees 100% process isolation between models and prevents class-level memory leaks.
        self.load = lru_cache(maxsize=maxsize)(self._load)

    def clear_cache(self):
        self.load.cache_clear()

    def _load(self, domain: str) -> T:
        """
        Internal target for file loading and deserialization.
        Invoked implicitly by the instance's bound self.load() wrapper.
        """
        # Using lazy logging to prevent string evaluation when debug is disabled
        logger.debug("Cache miss: Loading %s for domain '%s' from disk context.", self.target_class.__name__, domain)
        
        # Deterministic file matching: Look exactly for <domain>.json
        expected_filename = f"{domain}.json"
        target_file = self._resource_path.joinpath(expected_filename)

        # Handle missing resources explicitly
        try:
            if not target_file.is_file():
                logger.error("Domain '%s' not found in target path: %s", domain, self._resource_path)
                raise FileNotFoundError(f"Could not find CodeList JSON for domain: '{domain}'")
        except OSError as e:
            logger.error("Failed to access path %s: %s", self._resource_path, e)
            raise
            
        # Read the raw serialization content from the file stream
        try:
            with target_file.open('r', encoding='utf-8') as f:
                json_raw_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON syntax in %s: %s", target_file.name, e)
            raise 
            
        # Hand execution directly over to the underlying Rosetta deserializer
        try:
            # Execute the auto-generated deserialization framework method
            rune_object = self.target_class.rune_deserialize(json_raw_data) # type: ignore[attr-defined]
            logger.debug("Successfully deserialized model for domain tracking: '%s'.", domain)
            return rune_object
        except Exception as e:
            logger.error("Rune deserialization framework failure for domain '%s': %s", domain, e)
            raise