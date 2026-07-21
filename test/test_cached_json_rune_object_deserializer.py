import json
import pytest
from pathlib import Path

# Import the provider
from rune.runtime.extensions.cached_json_rune_object_deserializer import CachedJSONRuneObjectDeserializer

# ==============================================================================
# --- DUMMY TARGET CLASSES ---
# ==============================================================================

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


# ==============================================================================
# --- FIXTURES ---
# ==============================================================================

@pytest.fixture
def mock_resource_dir(tmp_path: Path):
    """
    Creates a temporary directory populated with mock JSON files.
    
    Disclaimer: We use codelist JSON mock data because it is the reason why the 
    CachedJSONRuneObjectDeserializer class was implemented. This data and tests 
    should show the implementation of the deserializer to be enough resilient 
    and consistent as to work with any JSON data.
    """
    # Valid JSON file correctly named
    valid_file = tmp_path / "business-center-6-4.json"
    valid_file.write_text('{"description": "Valid business center", "trigger_error": false}', encoding='utf-8')
    
    # Valid JSON structure, but designed to trigger our dummy deserialization error
    exploding_file = tmp_path / "exploding-domain.json"
    exploding_file.write_text('{"description": "Will explode", "trigger_error": true}', encoding='utf-8')
    
    # Invalid JSON syntax
    bad_json_file = tmp_path / "bad-format.json"
    bad_json_file.write_text('{this is not valid json}', encoding='utf-8')
    
    return tmp_path

@pytest.fixture
def deserializer(mock_resource_dir: Path):
    """
    Yields a fresh deserializer instance for each test, bound to the dummy model.
    """
    return CachedJSONRuneObjectDeserializer(
        codelist_dir=mock_resource_dir, 
        target_class=DummyRuneModel
    )


# ==============================================================================
# --- INITIALIZATION & FAIL-FAST VALIDATION ---
# ==============================================================================

def test_init_invalid_target_class(mock_resource_dir: Path):
    """Validates the deserializer fails fast if the target class lacks rune_deserialize."""
    with pytest.raises(TypeError) as exc_info:
        CachedJSONRuneObjectDeserializer(codelist_dir=mock_resource_dir, target_class=InvalidDummyModel)
    assert "must implement a callable 'rune_deserialize' method" in str(exc_info.value)

@pytest.mark.parametrize("invalid_maxsize", [
    0, -5, 1.5, "15", None
])
def test_init_invalid_maxsize(mock_resource_dir: Path, invalid_maxsize):
    """Ensures maxsize strictly enforces positive integers to prevent cache misconfiguration."""
    with pytest.raises(ValueError) as exc_info:
        CachedJSONRuneObjectDeserializer(
            codelist_dir=mock_resource_dir, 
            target_class=DummyRuneModel, 
            maxsize=invalid_maxsize
        )
    assert "must be a positive integer" in str(exc_info.value)

def test_init_invalid_directory_or_module(tmp_path: Path):
    """Validates the deserializer fails if the string is neither a path nor a python module."""
    # Create a string that is definitely not a module and not an existing directory
    bad_input = str(tmp_path / "does_not_exist")
    
    with pytest.raises(ValueError) as exc_info:
        CachedJSONRuneObjectDeserializer(codelist_dir=bad_input, target_class=DummyRuneModel)
    assert "neither a valid directory path nor a discoverable Python module" in str(exc_info.value)

def test_init_with_valid_path_string(tmp_path: Path):
    """Validates the deserializer correctly handles a valid filesystem path provided as a string."""
    valid_dir_str = str(tmp_path)
    instance = CachedJSONRuneObjectDeserializer(
        codelist_dir=valid_dir_str, 
        target_class=DummyRuneModel
    )
    assert instance._resource_path == tmp_path

def test_init_with_valid_module_string(mocker):
    """
    Mocks the importlib utils to prove the class correctly accepts and wraps 
    Python module string identifiers using importlib.resources.
    """
    # Mock find_spec to pretend 'my.fake.module' is a real installed python package
    mocker.patch('importlib.util.find_spec', return_value=True)
    mock_files = mocker.patch('importlib.resources.files', return_value="TraversableMock")
    
    instance = CachedJSONRuneObjectDeserializer(
        codelist_dir="my.fake.module", 
        target_class=DummyRuneModel
    )
    
    mock_files.assert_called_once_with("my.fake.module")
    assert instance._resource_path == "TraversableMock"


# ==============================================================================
# --- CACHE MECHANICS ---
# ==============================================================================

def test_cache_clear(deserializer: CachedJSONRuneObjectDeserializer):
    """Proves the clear_cache method successfully wipes instance memory."""
    deserializer.load("business-center-6-4")
    assert deserializer.load.cache_info().hits == 0
    assert deserializer.load.cache_info().misses == 1
    
    # Prove hit works
    deserializer.load("business-center-6-4")
    assert deserializer.load.cache_info().hits == 1
    
    # Wipe memory
    deserializer.clear_cache()
    assert deserializer.load.cache_info().hits == 0
    assert deserializer.load.cache_info().misses == 0
    
    # Third load should be a miss again
    deserializer.load("business-center-6-4")
    assert deserializer.load.cache_info().misses == 1

def test_cache_isolation(mock_resource_dir: Path):
    """Proves that cache is bound to the instance, not the class."""
    deserializer1 = CachedJSONRuneObjectDeserializer(mock_resource_dir, DummyRuneModel)
    deserializer2 = CachedJSONRuneObjectDeserializer(mock_resource_dir, DummyRuneModel)
    
    # Load into Instance 1
    deserializer1.load("business-center-6-4")
    assert deserializer1.load.cache_info().misses == 1
    
    # Loading the same data into Instance 2 MUST be a miss, proving isolation
    deserializer2.load("business-center-6-4")
    assert deserializer2.load.cache_info().misses == 1
    assert deserializer2.load.cache_info().hits == 0


# ==============================================================================
# --- RESOURCE LOADING & DESERIALIZATION ---
# ==============================================================================

def test_successful_load_and_cache(deserializer: CachedJSONRuneObjectDeserializer):
    """Validates that a correctly named deterministic file is parsed, deserialized, and cached."""
    
    # First call: Should be a cache miss
    result1 = deserializer.load("business-center-6-4")
    
    assert isinstance(result1, DummyRuneModel)
    assert result1.data["description"] == "Valid business center"
    
    info1 = deserializer.load.cache_info()
    assert info1.misses == 1
    assert info1.hits == 0
    
    # Second call: Should be a cache hit
    result2 = deserializer.load("business-center-6-4")
    
    assert result1 is result2  # Must be the exact same instance in memory
    
    info2 = deserializer.load.cache_info()
    assert info2.misses == 1
    assert info2.hits == 1

def test_file_not_found(deserializer: CachedJSONRuneObjectDeserializer):
    """Ensures looking for a non-existent file raises an error."""
    with pytest.raises(FileNotFoundError) as exc_info:
        deserializer.load("non-existent-domain")
        
    assert "Could not find CodeList JSON" in str(exc_info.value)

def test_invalid_json_handling(deserializer: CachedJSONRuneObjectDeserializer, caplog):
    """Ensures corrupt JSON files raise the appropriate JSONDecodeError and log the failure."""
    with pytest.raises(json.JSONDecodeError):
        deserializer.load("bad-format")
        
    assert "Failed to parse JSON syntax" in caplog.text

def test_deserialization_failure(deserializer: CachedJSONRuneObjectDeserializer, caplog):
    """Ensures exceptions raised by the target_class's deserializer are caught and propagated."""
    with pytest.raises(ValueError) as exc_info:
        deserializer.load("exploding-domain")
        
    assert "Simulated Rune validation explosion!" in str(exc_info.value)
    assert "Rune deserialization framework failure" in caplog.text