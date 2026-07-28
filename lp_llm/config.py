"""
Configuration file for Lifelong Personalized LLM (LP-LLM) Cognitive Architecture
Authored by Shuvam (https://github.com/samshuvam)
Contains API configuration, paths, memory thresholds, validation settings, and system flags.
"""

import os
from pathlib import Path
from datetime import timedelta
from .identity import verify_system_integrity, __author__, __github__

# Trigger integrity check on config load
_SIG = verify_system_integrity()

# ============================================================================
# API KEYS & CREDENTIALS (Loaded securely from Environment Variables)
# ============================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ============================================================================
# DIRECTORY PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MEMORY_DIR = BASE_DIR / "semantic_memory"
VALIDATION_DIR = BASE_DIR / "validation_cache"
LEARNING_DIR = BASE_DIR / "learning_data"
METRICS_DIR = BASE_DIR / "research_metrics"
USER_PROFILE_DIR = BASE_DIR / "user_profiles"
LOGS_DIR = BASE_DIR / "logs"
EXPORTS_DIR = BASE_DIR / "exports"

# Ensure runtime directories exist
for directory in [MEMORY_DIR, VALIDATION_DIR, LEARNING_DIR, METRICS_DIR, 
                  USER_PROFILE_DIR, LOGS_DIR, EXPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
    if directory == VALIDATION_DIR:
        (directory / "pending").mkdir(exist_ok=True)
        (directory / "verified").mkdir(exist_ok=True)
        (directory / "flagged").mkdir(exist_ok=True)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
class LoggingConfig:
    """Logging configuration for clean console output and full file diagnostics"""
    LOG_LEVEL = "INFO"
    FILE_LOG_LEVEL = "DEBUG"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = LOGS_DIR / "system.log"
    
    SUPPRESS_FROM_CONSOLE = [
        'validation', 'learning', 'knowledge_graph', 'httpx', 
        'huggingface_hub', 'sentence_transformers', 'googleapiclient',
        'chromadb', 'urllib3', 'pytz'
    ]

# ============================================================================
# MEMORY SETTINGS (Ebbinghaus Forgetting & RIF Suppression)
# ============================================================================
class MemoryConfig:
    """Cognitive memory parameters"""
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    
    COLLECTION_NAME = "semantic_knowledge"
    DISTANCE_METRIC = "cosine"
    
    # Ebbinghaus decay parameters
    MEMORY_HALF_LIFE_HOURS = 24
    IMPORTANCE_WEIGHT = 0.15
    RETRIEVAL_WEIGHT = 0.05
    VALIDATION_WEIGHT = 0.20
    
    # Retrieval & RIF suppression
    DEFAULT_RETRIEVAL_COUNT = 5
    RELEVANCE_THRESHOLD = 0.4
    RIF_SUPPRESSION_FACTOR = 0.3
    RIF_CONCEPT_OVERLAP_THRESHOLD = 0.7
    
    # Memory Tiers
    EPHEMERAL_MEMORY_TTL = timedelta(hours=1)
    WORKING_MEMORY_TTL = timedelta(days=7)
    LONG_TERM_MEMORY_TTL = timedelta(days=90)
    
    # Concept frequency
    MIN_CONCEPT_FREQUENCY = 2
    HIGH_FREQUENCY_THRESHOLD = 5
    
    AUTO_SAVE_INTERVAL_SECONDS = 30
    USER_PROFILE_FILE = USER_PROFILE_DIR / "user_profile.json"

# ============================================================================
# POST-RESPONSE VALIDATION SETTINGS
# ============================================================================
class ValidationConfig:
    """Fact verification and guardrail thresholds"""
    MIN_RESPONSE_LENGTH = 100
    CLAIM_INDICATORS = [
        "is", "are", "was", "were", "will be", "has", "have", "had",
        "according to", "studies show", "research indicates", "evidence suggests"
    ]
    
    CONFIDENCE_THRESHOLD = 0.7
    HIGH_CONFIDENCE_THRESHOLD = 0.9
    
    MAX_SEARCH_RESULTS = 5
    SEARCH_TIMEOUT_SECONDS = 10
    
    ASYNC_VALIDATION = True
    VALIDATION_BATCH_SIZE = 5
    VALIDATION_CHECK_INTERVAL_SECONDS = 60
    
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 5

# ============================================================================
# CONTINUAL LEARNING SETTINGS (LoRA & Sleep Consolidation)
# ============================================================================
class LearningConfig:
    """Continual learning and LoRA tuning parameters"""
    LORA_RANK = 8
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    MIN_TRAINING_EXAMPLES = 10
    MAX_TRAINING_EXAMPLES = 100
    
    CONSOLIDATION_INTERVAL_HOURS = 24
    CONSOLIDATION_TOP_K = 50
    CONSOLIDATION_MIN_STRENGTH = 0.3
    BACKGROUND_CONSOLIDATION = True
    
    ADAPTIVE_LORA = True
    CATASTROPHIC_FORGETTING_THRESHOLD = 0.15
    
    PERSONAL_INFO_WEIGHT = 5
    PREFERENCE_WEIGHT = 4
    FACT_WEIGHT = 3
    OPINION_WEIGHT = 2
    CONVERSATION_WEIGHT = 1
    
    ADAPTER_SAVE_PATH = LEARNING_DIR / "adapters"
    AUTO_SAVE_ADAPTERS = True

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
class ModelConfig:
    """Base LLM configuration"""
    MODEL_PATH = str(MODEL_PATH)
    CONTEXT_SIZE = 4096
    MAX_NEW_TOKENS = 200
    TEMPERATURE = 0.7
    TOP_P = 0.9
    STOP_SEQUENCES = ["User:", "Assistant:", "\n\n"]
    
    N_THREADS = 4
    N_BATCH = 512
    USE_MMAP = True
    
    USE_CONFIDENCE_CALIBRATION = True
    N_CANDIDATE_GENERATIONS = 3
    TEMPERATURE_RANGE = [0.3, 0.7, 1.0]

# ============================================================================
# QUERY UNDERSTANDING SETTINGS
# ============================================================================
class QueryUnderstandingConfig:
    """Intent classification & realtime query parameters"""
    INTENT_CATEGORIES = [
        "information_request", "memory_storage", "correction",
        "follow_up", "greeting", "personal_info", "command",
        "confirmation", "casual_chat", "math_calculation"
    ]
    
    REALTIME_TYPES = [
        "weather", "time", "date", "news", "pm", "president",
        "sports", "stocks", "none"
    ]
    
    MIN_CONFIDENCE_FOR_ACTION = 0.5
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    
    MAX_CONTEXT_TURNS = 5
    FOLLOW_UP_WINDOW_TURNS = 3

# ============================================================================
# RESEARCH METRICS SETTINGS
# ============================================================================
class MetricsConfig:
    """Research benchmark tracking configuration"""
    TRACKED_METRICS = [
        "memory_retention_rate",
        "hallucination_rate",
        "adaptation_speed",
        "catastrophic_forgetting_score",
        "concept_graph_coherence",
        "validation_success_rate",
        "response_latency",
        "learning_efficiency"
    ]
    
    EXPORT_FORMAT = "json"
    EXPORT_INTERVAL_HOURS = 24
    AUTO_EXPORT = True
    
    BENCHMARK_QUERIES_FILE = LEARNING_DIR / "benchmark_queries.json"
    BENCHMARK_INTERVAL_HOURS = 12

# ============================================================================
# SYSTEM & AUTHOR CONFIGURATION
# ============================================================================
class SystemConfig:
    """General system metadata and safety flags"""
    SYSTEM_NAME = "LP-LLM Cognitive Engine"
    AUTHOR = "Shuvam"
    GITHUB = "https://github.com/samshuvam"
    DEFAULT_USER_NAME = "Shuvam"
    
    LOG_LEVEL = LoggingConfig.LOG_LEVEL
    LOG_FORMAT = LoggingConfig.LOG_FORMAT
    LOG_FILE = LoggingConfig.LOG_FILE
    
    ENABLE_TELEMETRY = False
    ASYNC_VALIDATION = True
    BACKGROUND_CONSOLIDATION = True
    
    MAX_CONVERSATION_LENGTH = 1000
    SANITIZE_INPUT = True
    BLOCKED_PATTERNS = []
    
    ENABLE_LORA_TRAINING = True
    ENABLE_KNOWLEDGE_GRAPH = True
    ENABLE_METRICS_TRACKING = True
    ENABLE_AUTO_CONSOLIDATION = True

def get_config():
    """Get full system configuration dictionary"""
    return {
        "memory": MemoryConfig,
        "validation": ValidationConfig,
        "learning": LearningConfig,
        "model": ModelConfig,
        "metrics": MetricsConfig,
        "system": SystemConfig,
        "logging": LoggingConfig,
        "query_understanding": QueryUnderstandingConfig,
        "author": __author__,
        "github": __github__
    }

def print_config_summary():
    """Print configuration summary to console"""
    print("=" * 60)
    print("LIFELONG PERSONALIZED LLM (LP-LLM) - COGNITIVE ARCHITECTURE")
    print(f"Authored by: Shuvam ({__github__})")
    print("=" * 60)
    print(f"System Name: {SystemConfig.SYSTEM_NAME}")
    print(f"Model Path: {ModelConfig.MODEL_PATH}")
    print(f"Memory Directory: {MEMORY_DIR}")
    print(f"Validation Directory: {VALIDATION_DIR}")
    key_disp = '***' + GOOGLE_API_KEY[-4:] if GOOGLE_API_KEY else 'Not Set (Optional)'
    print(f"Google API Key: {key_disp}")
    print(f"Search Engine ID: {SEARCH_ENGINE_ID if SEARCH_ENGINE_ID else 'Not Set (Optional)'}")
    print(f"Log Level: {SystemConfig.LOG_LEVEL} (Console), DEBUG (File)")
    print(f"LoRA Training: {'Enabled' if SystemConfig.ENABLE_LORA_TRAINING else 'Disabled'}")
    print(f"Auto Consolidation: {'Enabled' if SystemConfig.ENABLE_AUTO_CONSOLIDATION else 'Disabled'}")
    print("=" * 60)
