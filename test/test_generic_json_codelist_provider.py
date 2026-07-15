import json
import pytest
from pathlib import Path

# Import the provider from its new home in the extensions directory
from rune.runtime.extensions.generic_json_codelist_provider import GenericJSONCodelistProvider

# --- DUMMY TARGET CLASS ---

class DummyRuneModel:
    """
    A mock model class to isolate the provider tests from the actual CDM generated code.
    It simulates the `rune_deserialize` interface expected by the generic provider.
    """
    def __init__(self, data):
        self.data = data
        
    @classmethod
    def rune_deserialize(cls, raw_data: dict):
        if raw_data.get("trigger_error"):
            raise ValueError("Simulated Rune validation explosion!")
        return cls(raw_data)
    
class InvalidDummyModel:
    """A model completely missing the rune_deserialize method."""
    pass

# --- FIXTURES ---

@pytest.fixture
def mock_codelist_dir(tmp_path: Path):
    """
    Creates a temporary directory populated with mock JSON codelist files.
    This guarantees no dependency on external repos or file downloads.
    """
    # 1. Valid JSON file correctly named
    valid_file = tmp_path / "business-center-9-3.json"
    valid_file.write_text('{"description": "Valid business center", "trigger_error": false}', encoding='utf-8')
    
    # 2. Valid JSON structure, but designed to trigger our dummy deserialization error
    exploding_file = tmp_path / "exploding-domain-2-0.json"
    exploding_file.write_text('{"description": "Will explode", "trigger_error": true}', encoding='utf-8')
    
    # 3. Invalid JSON syntax
    bad_json_file = tmp_path / "bad-format-1-0.json"
    bad_json_file.write_text('{this is not valid json}', encoding='utf-8')
    
    return tmp_path


@pytest.fixture
def provider(mock_codelist_dir: Path):
    """
    Yields a fresh provider instance for each test, bound to the dummy model.
    """
    return GenericJSONCodelistProvider(
        codelist_dir=mock_codelist_dir, 
        target_class=DummyRuneModel
    )


# --- UNIT TESTS ---

def test_init_invalid_directory(tmp_path: Path):
    """Validates the provider fails fast if given a file instead of a directory."""
    not_a_dir = tmp_path / "just_a_file.txt"
    not_a_dir.write_text("Plain text")
    
    with pytest.raises(NotADirectoryError) as exc_info:
        GenericJSONCodelistProvider(codelist_dir=not_a_dir, target_class=DummyRuneModel)
    assert "is not a directory" in str(exc_info.value)

def test_init_invalid_target_class(mock_codelist_dir: Path):
    """Validates the provider fails fast if the target class lacks rune_deserialize."""
    with pytest.raises(TypeError) as exc_info:
        GenericJSONCodelistProvider(codelist_dir=mock_codelist_dir, target_class=InvalidDummyModel)
    assert "must implement a callable 'rune_deserialize' method" in str(exc_info.value)

def test_cache_clear(provider: GenericJSONCodelistProvider):
    """Proves the clear_cache method successfully wipes instance memory."""
    provider.load("business-center")
    assert provider.load.cache_info().hits == 0
    assert provider.load.cache_info().misses == 1
    
    # Prove hit works
    provider.load("business-center")
    assert provider.load.cache_info().hits == 1
    
    # Wipe memory
    provider.clear_cache()
    assert provider.load.cache_info().hits == 0
    assert provider.load.cache_info().misses == 0
    
    # Third load should be a miss again
    provider.load("business-center")
    assert provider.load.cache_info().misses == 1

def test_cache_isolation(mock_codelist_dir: Path):
    """Proves that cache is bound to the instance, not the class."""
    provider1 = GenericJSONCodelistProvider(mock_codelist_dir, DummyRuneModel)
    provider2 = GenericJSONCodelistProvider(mock_codelist_dir, DummyRuneModel)
    
    # Load into Provider 1
    provider1.load("business-center")
    assert provider1.load.cache_info().misses == 1
    
    # Loading the same data into Provider 2 MUST be a miss, proving isolation
    provider2.load("business-center")
    assert provider2.load.cache_info().misses == 1
    assert provider2.load.cache_info().hits == 0

def test_successful_load_and_cache(provider: GenericJSONCodelistProvider):
    """Validates that a correctly named file is parsed, deserialized, and cached."""
    
    # First call: Should be a cache miss
    result1 = provider.load("business-center")
    
    assert isinstance(result1, DummyRuneModel)
    assert result1.data["description"] == "Valid business center"
    
    info1 = provider.load.cache_info()
    assert info1.misses == 1
    assert info1.hits == 0
    
    # Second call: Should be a cache hit
    result2 = provider.load("business-center")
    
    assert result1 is result2  # Must be the exact same instance in memory
    
    info2 = provider.load.cache_info()
    assert info2.misses == 1
    assert info2.hits == 1


def test_file_not_found(provider: GenericJSONCodelistProvider):
    """Ensures looking for a domain without a corresponding file raises an error."""
    with pytest.raises(FileNotFoundError) as exc_info:
        provider.load("non-existent-domain")
        
    assert "Could not find CodeList JSON" in str(exc_info.value)


def test_strict_regex_matching(provider: GenericJSONCodelistProvider):
    """Verifies the regex matches the domain exactly, preventing partial matches."""
    # Searching for 'center' should not match 'business-center-9-3.json'
    with pytest.raises(FileNotFoundError):
        provider.load("center")


def test_invalid_json_handling(provider: GenericJSONCodelistProvider, caplog):
    """Ensures corrupt JSON files raise the appropriate JSONDecodeError and log the failure."""
    with pytest.raises(json.JSONDecodeError):
        provider.load("bad-format")
        
    assert "Failed to parse JSON syntax" in caplog.text


def test_deserialization_failure(provider: GenericJSONCodelistProvider, caplog):
    """Ensures exceptions raised by the target_class's deserializer are caught and propagated."""
    with pytest.raises(ValueError) as exc_info:
        provider.load("exploding-domain")
        
    assert "Simulated Rune validation explosion!" in str(exc_info.value)
    assert "Rune deserialization framework failure" in caplog.text

