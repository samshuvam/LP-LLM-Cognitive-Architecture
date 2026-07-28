from .identity import verify_system_integrity, get_author_info, __author__, __version__
_SIG = verify_system_integrity()
import os
import sys
import json
import logging
import time
import threading
import asyncio
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from .realtime_fetcher import RealTimeFetcher
from .query_understanding import QueryUnderstanding
from .user_profile import UserProfile

from .config import (
    SystemConfig, ModelConfig, MemoryConfig, 
    ValidationConfig, LearningConfig, MetricsConfig,
    LoggingConfig, print_config_summary, MEMORY_DIR, VALIDATION_DIR, 
    LEARNING_DIR, METRICS_DIR, USER_PROFILE_DIR, LOGS_DIR, EXPORTS_DIR
)
from .memory import SemanticMemory
from .validation import PostResponseValidator, ValidationScheduler
from .learning import LearningManager
from .knowledge_graph import KnowledgeGraph
from .metrics import ResearchMetrics, BenchmarkSuite

# ============================================================================
# COMPLETE LOGGING SUPPRESSION (Category 3: Debug Log Suppression)
# ============================================================================
def setup_logging():
    """
    Configure logging to completely suppress debug logs from console
    All debug logs go to file only - clean chat interface
    """
    # Create logs directory
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create custom formatters
    console_formatter = logging.Formatter('%(message)s')
    file_formatter = logging.Formatter(LoggingConfig.LOG_FORMAT)
    
    # File handler (ALL logs - DEBUG level)
    file_handler = logging.FileHandler(LoggingConfig.LOG_FILE, encoding='utf-8')
    file_handler.setLevel(LoggingConfig.FILE_LOG_LEVEL)
    file_handler.setFormatter(file_formatter)
    
    # Console handler (only INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LoggingConfig.LOG_LEVEL)
    console_handler.setFormatter(console_formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress verbose libraries from console (file only)
    for logger_name in LoggingConfig.SUPPRESS_FROM_CONSOLE:
        lib_logger = logging.getLogger(logger_name)
        lib_logger.setLevel(logging.WARNING)
        lib_logger.propagate = False
    
    logger = logging.getLogger(__name__)
    logger.info("Logging configured: Console=INFO, File=DEBUG")

# Setup logging BEFORE any other imports
setup_logging()
logger = logging.getLogger(__name__)

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not available. Install with: pip install llama-cpp-python")


class ConversationContext:
    """
    Manages conversation history and context for follow-up questions
    Enhanced with better topic tracking and context inheritance
    """
    
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []
        self.current_topic: Optional[str] = None
        self.entities_mentioned: Dict[str, Any] = {}
        self.last_query_type: str = "general"
        self.topic_history: List[str] = []
    
    def add_turn(self, user_input: str, response: str, query_type: str = "general",
                 metadata: Optional[Dict] = None):
        """Add a conversation turn to history"""
        turn = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response,
            "query_type": query_type,
            "metadata": metadata or {}
        }
        
        self.history.append(turn)
        
        # Trim history if too long
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        # Update current topic
        self._update_topic(user_input, query_type)
        
        # Extract and track entities
        self._extract_entities(user_input)
        
        # Update last query type
        self.last_query_type = query_type
    
    def _update_topic(self, user_input: str, query_type: str):
        """Update current topic based on conversation"""
        topic_keywords = {
            "weather": ["weather", "temperature", "forecast", "rain", "sunny", "cloudy"],
            "location": ["live", "stay", "from", "location", "where", "city", "country"],
            "personal": ["name", "brother", "sister", "family", "friend", "mother", "father"],
            "time": ["time", "clock", "hour", "when", "o'clock"],
            "date": ["date", "day", "year", "month", "today", "tomorrow", "yesterday"],
            "politics": ["prime minister", "president", "government", "minister", "pm", "cm"],
            "news": ["news", "happening", "recent", "latest", "current"],
            "sports": ["cricket", "football", "match", "game", "score", "team"],
            "education": ["study", "school", "college", "university", "semester", "exam"]
        }
        
        user_lower = user_input.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in user_lower for kw in keywords):
                self.current_topic = topic
                self.topic_history.append(topic)
                if len(self.topic_history) > 10:
                    self.topic_history = self.topic_history[-10:]
                return
        
        if query_type != "general":
            self.current_topic = query_type
    
    def _extract_entities(self, user_input: str):
        """Extract and track entities mentioned in conversation"""
        # Name entities
        name_match = re.search(r"(?:my name is|i am|i'm)\s+([A-Za-z]+)", user_input, re.IGNORECASE)
        if name_match:
            self.entities_mentioned['user_name'] = name_match.group(1)
        
        # Location entities
        location_match = re.search(r"(?:live|stay|from)\s+in\s+([A-Za-z\s,]+)", user_input, re.IGNORECASE)
        if location_match:
            self.entities_mentioned['location'] = location_match.group(1).strip()
        
        # Family entities
        brother_match = re.search(r"brother(?:'s)? name is ([A-Za-z]+)", user_input, re.IGNORECASE)
        if brother_match:
            self.entities_mentioned['brother_name'] = brother_match.group(1)
    
    def get_context_for_prompt(self) -> str:
        """Build context string for LLM prompt from recent conversation"""
        if not self.history:
            return ""
        
        context_parts = []
        recent_turns = self.history[-5:]
        
        for turn in recent_turns:
            context_parts.append(f"User: {turn['user_input']}")
            context_parts.append(f"Sentrix: {turn['response']}")
        
        return "\n".join(context_parts)
    
    def is_follow_up_question(self, user_input: str) -> bool:
        """Detect if current input is a follow-up to previous conversation"""
        if not self.history:
            return False
        
        follow_up_indicators = [
            "there", "that", "those", "it", "they", "them", "this",
            "what about", "how about", "and", "also", "too", "else",
            "what else", "tell me more", "continue", "go on",
            "why", "because", "so", "then", "but", "however"
        ]
        
        user_lower = user_input.lower()
        
        if any(indicator in user_lower for indicator in follow_up_indicators):
            if len(user_input.split()) < 10:
                return True
        
        pronouns = ["he", "she", "it", "they", "him", "her", "them"]
        if any(pronoun in user_lower.split() for pronoun in pronouns):
            if len(user_input.split()) < 12:
                return True
        
        # Check topic continuity
        if self.current_topic:
            topic_keywords = {
                "weather": ["weather", "temperature", "forecast"],
                "politics": ["pm", "minister", "president", "government"],
                "location": ["there", "here", "city", "place"]
            }
            
            if self.current_topic in topic_keywords:
                if any(kw in user_lower for kw in topic_keywords[self.current_topic]):
                    return True
        
        return False
    
    def get_recent_query_types(self, n: int = 3) -> List[str]:
        """Get recent query types for context inheritance"""
        if not self.history:
            return ["general"]
        
        return [turn.get('query_type', 'general') for turn in self.history[-n:]]
    
    def clear(self):
        """Clear conversation history"""
        self.history = []
        self.current_topic = None
        self.entities_mentioned = {}
        self.last_query_type = "general"
        self.topic_history = []
        logger.info("Conversation context cleared")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get conversation context summary"""
        return {
            "total_turns": len(self.history),
            "current_topic": self.current_topic,
            "entities_mentioned": self.entities_mentioned,
            "last_query_type": self.last_query_type,
            "recent_query_types": self.get_recent_query_types(),
            "topic_history": self.topic_history[-5:]
        }


class CognitiveChatbot:
    """
    Main chatbot interface with full cognitive capabilities
    Production-ready with all 13 categories implemented
    """
    
    def __init__(self, profile_name: str = "default"):
        logger.info("Initializing Sentrix - Self-Evolving Cognitive Architecture...")
        
        # Initialize all components with error handling
        try:
            self.memory = SemanticMemory()
            logger.info("✓ Memory system initialized")
        except Exception as e:
            logger.error(f"Memory initialization failed: {e}")
            raise
        
        try:
            self.validator = PostResponseValidator()
            logger.info("✓ Validation system initialized")
        except Exception as e:
            logger.error(f"Validation initialization failed: {e}")
            raise
        
        try:
            self.learning_manager = LearningManager(self.memory, self.validator)
            logger.info("✓ Learning system initialized")
        except Exception as e:
            logger.error(f"Learning initialization failed: {e}")
            raise
        
        try:
            self.knowledge_graph = KnowledgeGraph()
            logger.info("✓ Knowledge graph initialized")
        except Exception as e:
            logger.error(f"Knowledge graph initialization failed: {e}")
            raise
        
        try:
            self.metrics = ResearchMetrics()
            logger.info("✓ Metrics system initialized")
        except Exception as e:
            logger.error(f"Metrics initialization failed: {e}")
            raise
        
        self.benchmark = BenchmarkSuite(self.metrics)
        
        try:
            self.realtime_fetcher = RealTimeFetcher()
            logger.info("✓ Real-time fetcher initialized")
        except Exception as e:
            logger.error(f"Real-time fetcher initialization failed: {e}")
            raise
        
        # Initialize user profile (Category 2: Memory Persistence)
        try:
            self.user_profile = UserProfile(profile_name)
            self.user_location = self.user_profile.get_location()
            self.user_name = self.user_profile.get_name() or SystemConfig.DEFAULT_USER_NAME
            logger.info(f"✓ User profile loaded: {profile_name}")
        except Exception as e:
            logger.error(f"User profile initialization failed: {e}")
            self.user_profile = UserProfile("default")
            self.user_location = None
            self.user_name = SystemConfig.DEFAULT_USER_NAME
        
        # Initialize query analyzer (Category 4: Query Understanding Accuracy)
        try:
            self.query_analyzer = QueryUnderstanding()
            logger.info("✓ Query analyzer initialized")
        except Exception as e:
            logger.error(f"Query analyzer initialization failed: {e}")
            raise
        
        # Initialize conversation context manager
        self.conversation_context = ConversationContext(max_history=20)
        
        # Initialize model with retry mechanism (Category 10: Error Handling)
        self.model = None
        self.generate_func = None
        self._initialize_model_with_retry()
        
        # System state
        self.system_name = SystemConfig.SYSTEM_NAME
        self.conversation_count = 0
        self.session_start_time = datetime.now()
        
        # Background tasks
        self.validation_scheduler = ValidationScheduler(self.validator)
        self.consolidation_scheduler = ConsolidationScheduler(self.learning_manager, self.memory)
        self.background_tasks = []
        
        # Load existing knowledge
        self._load_existing_knowledge()
        
        # Start background processes (Category 5 & 6: Validation & Consolidation)
        self._start_background_processes()
        
        # Display initialization summary
        self._display_initialization_summary()
        
        logger.info("✓ Sentrix initialization complete")
    
    def _initialize_model_with_retry(self, max_retries: int = 3):
        """Initialize model with retry mechanism (Category 10)"""
        for attempt in range(max_retries):
            try:
                self._initialize_model()
                if self.model is not None:
                    logger.info(f"Model loaded successfully on attempt {attempt + 1}")
                    return
            except Exception as e:
                logger.warning(f"Model load attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in 5 seconds...")
                    time.sleep(5)
        
        logger.error("Model loading failed after all retries. Using rule-based responses.")
    
    def _initialize_model(self):
        """Initialize the LLM model"""
        if not LLAMA_CPP_AVAILABLE:
            logger.warning("llama-cpp-python not available, using rule-based responses")
            return
        
        if not os.path.exists(ModelConfig.MODEL_PATH):
            logger.warning(f"Model file not found at: {ModelConfig.MODEL_PATH}")
            logger.warning("Using rule-based responses fallback")
            return
        
        try:
            logger.info(f"Loading GGUF model: {ModelConfig.MODEL_PATH}")
            self.model = Llama(  # type: ignore
                model_path=ModelConfig.MODEL_PATH,
                n_ctx=ModelConfig.CONTEXT_SIZE,
                n_threads=ModelConfig.N_THREADS,
                n_batch=ModelConfig.N_BATCH,
                use_mmap=ModelConfig.USE_MMAP,
                verbose=False
            )
            
            def generate_text(prompt: str, max_tokens: Optional[int] = None) -> str:
                if self.model is None:
                    raise RuntimeError("Model not loaded")
                max_tokens = max_tokens or ModelConfig.MAX_NEW_TOKENS
                output = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=ModelConfig.TEMPERATURE,
                    top_p=ModelConfig.TOP_P,
                    stop=ModelConfig.STOP_SEQUENCES,
                    stream=False
                )
                if isinstance(output, dict) and 'choices' in output:
                    return output['choices'][0]['text'].strip()
                return str(output).strip()
            
            self.generate_func = generate_text
            logger.info("GGUF model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.warning("Using rule-based responses fallback")
            self.model = None
            self.generate_func = None
    
    def _load_existing_knowledge(self):
        """Load existing knowledge from files"""
        verified_dir = self.validator.verified_dir
        if verified_dir.exists():
            count = 0
            for filepath in verified_dir.glob("*.json"):
                try:
                    with open(filepath, 'r') as f:
                        entry = json.load(f)
                    self.memory.add_memory(
                        f"User: {entry['query']}\n{self.system_name}: {entry['response']}",
                        metadata={"source": "verified", "concept": entry.get('metadata', {}).get('concept', 'general')}
                    )
                    count += 1
                except Exception as e:
                    logger.warning(f"Could not load entry {filepath}: {e}")
            logger.info(f"Loaded {count} verified interactions from cache")
    
    def _start_background_processes(self):
        """Start background tasks (Category 5 & 6)"""
        # Start validation scheduler
        if ValidationConfig.ASYNC_VALIDATION:
            async def start_scheduler():
                await self.validation_scheduler.start()
            
            scheduler_thread = threading.Thread(
                target=lambda: asyncio.run(start_scheduler()),
                daemon=True
            )
            scheduler_thread.start()
            self.background_tasks.append(scheduler_thread)
            logger.info("Validation scheduler started (60s interval)")
        
        # Start consolidation scheduler (Category 6)
        if LearningConfig.BACKGROUND_CONSOLIDATION:
            consolidation_thread = threading.Thread(
                target=self.consolidation_scheduler.start,
                daemon=True
            )
            consolidation_thread.start()
            self.background_tasks.append(consolidation_thread)
            logger.info("Consolidation scheduler started (24h interval)")
    
    def _display_initialization_summary(self):
        """Display initialization summary to user"""
        print("\n" + "=" * 70)
        print(f"  {self.system_name} - Self-Evolving Cognitive Architecture")
        print("  Production-Ready Implementation with:")
        print("  ✓ Ebbinghaus Forgetting Curve & RIF")
        print("  ✓ Post-Response Validation Pipeline")
        print("  ✓ Sleep-Phase Knowledge Consolidation")
        print("  ✓ Semantic Drift Detection")
        print("  ✓ Comprehensive Research Metrics (8 metrics)")
        print("  ✓ Real-Time Data Fetching (Weather, PM, News, Time, Date)")
        print("  ✓ Conversation Context & Follow-up Questions")
        print("  ✓ LLM-Based Query Understanding")
        print("  ✓ User Profile Persistence")
        print("  ✓ LoRA Fine-Tuning Pipeline")
        print("=" * 70)
        print("\nCommands:")
        print("  'status'    - Show system status")
        print("  'benchmark' - Run benchmark suite")
        print("  'export'    - Export research data")
        print("  'profile'   - Show user profile")
        print("  'clear'     - Clear conversation context")
        print("  'memory'    - Review stored memories")
        print("  'quit'      - Exit the system")
        print("=" * 70)
    
    def process_input(self, user_input: str) -> str:
        """Process user input with FULL query understanding (Category 4 & 9)"""
        start_time = time.time()
        self.conversation_count += 1
        
        # Auto-save user profile if needed (Category 2)
        self.user_profile.auto_save_if_needed()
        self.user_profile.increment_conversation_stats()
        
        # ========================================================================
        # STEP 1: UNDERSTAND THE QUERY (before any action!)
        # ========================================================================
        user_profile_dict = {
            "name": self.user_name,
            "location": self.user_location
        }
        
        query_analysis = self.query_analyzer.analyze_query(
            user_input,
            self.conversation_context.history,
            user_profile_dict
        )
        
        # Log at DEBUG level only (won't show in console - Category 3)
        logger.debug(f"Query: {user_input[:50]}...")
        logger.debug(f"Intent: {query_analysis['intent']} (confidence: {query_analysis['confidence']:.2f})")
        logger.debug(f"Needs realtime: {query_analysis['needs_realtime']} ({query_analysis['realtime_type']})")
        logger.debug(f"Is correction: {query_analysis['is_correction']}")
        logger.debug(f"Is follow-up: {query_analysis['is_follow_up']}")
        
        # ========================================================================
        # STEP 2: Extract user info
        # ========================================================================
        self._extract_user_info(user_input)
        
        # ========================================================================
        # STEP 3: Handle based on understood intent
        # ========================================================================
        
        if query_analysis["intent"] == "memory_storage":
            response = self._handle_memory_storage(user_input, query_analysis)
        
        elif query_analysis["intent"] == "correction":
            response = self._handle_correction(user_input, query_analysis)
        
        elif query_analysis["intent"] == "personal_info":
            response = self._handle_personal_info(user_input, query_analysis)
        
        elif query_analysis["needs_realtime"]:
            logger.info(f"Fetching real-time data: {query_analysis['realtime_type']}")
            realtime_data = self.realtime_fetcher.fetch(
                user_input,
                self.user_location,
                self.conversation_context.history
            )
            
            if realtime_data.get('success'):
                response = self._generate_response_with_realtime(
                    user_input, realtime_data, query_analysis['realtime_type']
                )
            else:
                response = self._generate_response_with_context(user_input, query_analysis)
        
        elif query_analysis["intent"] == "greeting":
            response = self._handle_greeting(user_input)
        
        elif query_analysis["intent"] == "math_calculation":
            response = self._handle_math(user_input, query_analysis)
        
        elif query_analysis["intent"] == "confirmation":
            response = self._handle_confirmation(user_input, query_analysis)
        
        else:
            response = self._generate_response_with_context(user_input, query_analysis)
        
        # ========================================================================
        # STEP 4: Post-response validation
        # ========================================================================
        if self.validator.should_validate(response):
            entry_id = self.validator.store_pending(
                user_input, response,
                metadata={
                    "user": self.user_name,
                    "conversation_count": self.conversation_count,
                    "query_intent": query_analysis["intent"],
                    "query_type": query_analysis["realtime_type"]
                }
            )
            if entry_id:
                logger.debug(f"Stored for validation: {entry_id[:8]}")
        
        # ========================================================================
        # STEP 5: Store interaction in memory
        # ========================================================================
        self.memory.add_memory(
            f"User: {user_input}\n{self.system_name}: {response}",
            metadata={
                "input": user_input,
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "source": "interaction",
                "intent": query_analysis["intent"],
                "query_type": query_analysis["realtime_type"]
            }
        )
        
        # ========================================================================
        # STEP 6: Update conversation context
        # ========================================================================
        self.conversation_context.add_turn(
            user_input, 
            response, 
            query_analysis["realtime_type"],
            metadata={
                "location": self.user_location, 
                "user_name": self.user_name,
                "intent": query_analysis["intent"]
            }
        )
        
        # ========================================================================
        # STEP 7: Update knowledge graph (Category 7)
        # ========================================================================
        self._update_knowledge_graph(user_input, response, query_analysis)
        
        # ========================================================================
        # STEP 8: Log metrics (Category 8)
        # ========================================================================
        latency = time.time() - start_time
        self.metrics.log("response_latency", latency, {
            "input_length": len(user_input),
            "intent": query_analysis["intent"],
            "is_follow_up": query_analysis["is_follow_up"],
            "needs_realtime": query_analysis["needs_realtime"]
        })
        
        logger.info(f"Processed in {latency:.2f}s")
        
        return response
    
    def _extract_user_info(self, user_input: str):
        """Extract user information and update profile (Category 2)"""
        # Name pattern
        name_pattern = r"(?:my name is|i am|i'm)\s+([A-Za-z]+)"
        matches = re.search(name_pattern, user_input, re.IGNORECASE)
        if matches:
            old_name = self.user_name
            self.user_name = matches.group(1)
            self.user_profile.update_personal_info("name", self.user_name)
            logger.info(f"Updated user name: {old_name} -> {self.user_name}")
            self.knowledge_graph.add_fact(self.user_name, "is_a", "user", confidence=1.0)
        
        # Location pattern
        location_patterns = [
            r"(?:live|stay|reside)\s+(?:in\s+)?([A-Za-z\s,]+?)(?:\.|,|!|\?|$)",
            r"(?:from)\s+([A-Za-z\s,]+?)(?:\.|,|!|\?|$)",
        ]
        
        for pattern in location_patterns:
            location_match = re.search(pattern, user_input, re.IGNORECASE)
            if location_match:
                location = location_match.group(1).strip()
                if location.lower() not in ['the', 'a', 'an', 'here', 'there']:
                    self.user_location = location
                    self.user_profile.update_personal_info("location", self.user_location)
                    logger.info(f"Updated user location: {self.user_location}")
                    self.knowledge_graph.add_fact(
                        self.user_name, "lives_in", self.user_location, confidence=0.95
                    )
                    self.memory.add_memory(
                        f"{self.user_name} lives in {self.user_location}",
                        metadata={"source": "extracted", "concept": "location", "importance": 5}
                    )
                    break
    
    def _handle_memory_storage(self, user_input: str, analysis: Dict) -> str:
        """Handle when user wants system to remember something"""
        entities = analysis["entities"]
        
        self.memory.add_memory(
            user_input,
            metadata={
                "source": "user_request",
                "importance": 5,
                "type": "user_requested_memory"
            }
        )
        
        if entities["person_names"]:
            for name in entities["person_names"]:
                self.knowledge_graph.add_fact(self.user_name, "remembers", name, confidence=0.9)
                self.user_profile.add_learned_fact(f"Remembers: {name}", "user_request")
        
        return f"Got it! I'll remember that for you, {self.user_name}."
    
    def _handle_correction(self, user_input: str, analysis: Dict) -> str:
        """Handle when user is correcting information (Category 4)"""
        context_refs = analysis["context_references"]
        
        if context_refs and self.conversation_context.history:
            ref_turn = self.conversation_context.history[context_refs[0]["turn_index"]]
            
            response = f"Thank you for the correction! I've noted that {user_input}. I'll update my understanding."
            
            self.memory.add_memory(
                f"CORRECTION: {user_input}",
                metadata={
                    "source": "correction",
                    "corrected_from": ref_turn.get("response", "")[:100],
                    "importance": 5
                }
            )
            
            self.user_profile.add_correction(
                ref_turn.get("response", "")[:100],
                user_input
            )
            
            return response
        else:
            return f"Thank you for correcting me! I've noted: {user_input}"
    
    def _handle_personal_info(self, user_input: str, analysis: Dict) -> str:
        """Handle questions about user's personal information"""
        if "name" in user_input.lower() and "my" in user_input.lower():
            if self.user_name != SystemConfig.DEFAULT_USER_NAME:
                return f"Your name is {self.user_name}, as you told me."
            return "I don't recall you telling me your name yet."
        
        if any(word in user_input.lower() for word in ["live", "location", "where"]):
            if self.user_location:
                return f"You mentioned you live in {self.user_location}."
            return "I don't have information about where you live."
        
        retrieved = self.memory.retrieve_similar(user_input, n_results=3)
        if retrieved:
            return f"Based on our conversations: {retrieved[0]['content'][:150]}..."
        
        return "I don't have that information about you yet."
    
    def _handle_greeting(self, user_input: str) -> str:
        """Handle greetings"""
        greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon"]
        
        for greeting in greetings:
            if greeting in user_input.lower():
                return f"Hello {self.user_name}! How can I assist you today?"
        
        if "how are you" in user_input.lower():
            return f"I'm doing well, thank you for asking, {self.user_name}! How can I help you today?"
        
        return f"Hello {self.user_name}! How can I assist you?"
    
    def _handle_math(self, user_input: str, analysis: Dict) -> str:
        """Handle mathematical calculations"""
        numbers = re.findall(r'\d+', user_input)
        
        if '+' in user_input and len(numbers) >= 2:
            result = sum(int(n) for n in numbers)
            return f"The answer is {result}."
        elif '-' in user_input and len(numbers) >= 2:
            result = int(numbers[0]) - int(numbers[1])
            return f"The answer is {result}."
        elif ('*' in user_input or 'x' in user_input.lower()) and len(numbers) >= 2:
            result = int(numbers[0]) * int(numbers[1])
            return f"The answer is {result}."
        elif '/' in user_input and len(numbers) >= 2:
            result = int(numbers[0]) / int(numbers[1])
            return f"The answer is {result:.2f}."
        
        return "I can help with basic math calculations. What would you like to calculate?"
    
    def _handle_confirmation(self, user_input: str, analysis: Dict) -> str:
        """Handle yes/no confirmations"""
        query_lower = user_input.lower()
        
        affirmative = ["yes", "yeah", "yep", "correct", "right", "exactly", "that's right"]
        negative = ["no", "nah", "nope", "wrong", "incorrect", "not really"]
        
        if any(word in query_lower for word in affirmative):
            context_refs = analysis["context_references"]
            if context_refs:
                return "Great! I've noted your confirmation."
            return "Understood!"
        
        elif any(word in query_lower for word in negative):
            context_refs = analysis["context_references"]
            if context_refs:
                return "I understand. Let me update my information."
            return "Got it, thanks for clarifying!"
        
        return "Understood!"
    
    def _generate_response_with_realtime(self, user_input: str, realtime_data: Dict, query_type: str) -> str:
        """Generate response using real-time fetched data (Category 9)"""
        
        if query_type == "weather":
            if realtime_data.get('success'):
                return f"The current weather in {realtime_data.get('location')} is {realtime_data.get('temperature')} and {realtime_data.get('condition')}. (Source: {realtime_data.get('source')})"
            return "I couldn't fetch weather data right now."
        
        elif query_type == "time":
            if realtime_data.get('success'):
                return f"The current time is {realtime_data.get('time')}. (Source: {realtime_data.get('source')})"
            return f"The current time is {datetime.now().strftime('%I:%M %p')}."
        
        elif query_type == "date":
            if realtime_data.get('success'):
                return f"Today is {realtime_data.get('formatted')}. (Source: {realtime_data.get('source')})"
            return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."
        
        elif query_type == "pm":
            if realtime_data.get('success'):
                return f"The current Prime Minister is {realtime_data.get('pm_name')}. (Source: {realtime_data.get('source')})"
            return "I couldn't fetch that information right now."
        
        elif query_type == "news":
            if realtime_data.get('success'):
                summary = realtime_data.get('summary', 'No news available')
                source = realtime_data.get('source', 'N/A')
                return f"Here's the latest: {summary}\n\n(Source: {source})"
            return "I couldn't fetch current news."
        
        else:
            return self._rule_based_response(user_input, "")
    
    def _generate_response_with_context(self, user_input: str, analysis: Dict) -> str:
        """Generate response using model with full context (Category 9)"""
        conversation_history = self.conversation_context.get_context_for_prompt()
        
        retrieved_memories = self.memory.retrieve_similar(user_input, n_results=5)
        memory_context = self._build_context(retrieved_memories)
        
        understanding_context = f"""
Query Analysis:
- Intent: {analysis['intent']}
- Is Follow-up: {analysis['is_follow_up']}
- Is Correction: {analysis['is_correction']}
- Entities Found: {analysis['entities']}
- Context References: {len(analysis['context_references'])} previous turns
"""
        
        if conversation_history.strip():
            prompt = f"""You are {self.system_name}, a helpful AI assistant with memory and context awareness.

PREVIOUS CONVERSATION:
{conversation_history}

RELEVANT INFORMATION FROM MEMORY:
{memory_context}

QUERY UNDERSTANDING:
{understanding_context}

IMPORTANT: 
- If this is a follow-up question, use context from previous conversation
- If user is correcting you, acknowledge the correction
- If asking about personal info, check what you know about the user
- Be natural and conversational

CURRENT QUERY:
User: {user_input}
{self.system_name}:"""
        else:
            prompt = f"""You are {self.system_name}, a helpful AI assistant.

RELEVANT INFORMATION FROM MEMORY:
{memory_context}

QUERY UNDERSTANDING:
{understanding_context}

User: {user_input}
{self.system_name}:"""
        
        try:
            if self.generate_func and self.model is not None and LLAMA_CPP_AVAILABLE:
                response = self.generate_func(prompt, max_tokens=ModelConfig.MAX_NEW_TOKENS)
                return response
            
            return self._rule_based_response(user_input, memory_context)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._rule_based_response(user_input, memory_context)
    
    def _build_context(self, retrieved_memories: List[Dict]) -> str:
        """Build context from retrieved memories"""
        context_parts = []
        
        for memory in retrieved_memories:
            content = memory['content']
            clean_content = content.replace('User:', 'Previous user said:').replace('Assistant:', f'{self.system_name} replied:').replace('Sentrix:', f'{self.system_name} replied:')
            context_parts.append(clean_content)
        
        return "\n".join(context_parts) if context_parts else f"No prior conversations with {self.user_name}."
    
    def _rule_based_response(self, user_input: str, context: str) -> str:
        """Rule-based response generation fallback"""
        user_lower = user_input.lower()
        
        if "name" in user_lower and ("what" in user_lower or "who" in user_lower):
            if "your" in user_lower or "you" in user_lower:
                return f"My name is {self.system_name}, as you requested."
            elif "my" in user_lower:
                if self.user_name != SystemConfig.DEFAULT_USER_NAME:
                    return f"Your name is {self.user_name}, as you previously told me."
                return "I don't recall you telling me your name yet."
        
        if any(word in user_lower for word in ["live", "stay", "from", "location"]):
            if self.user_location:
                return f"You mentioned you live in {self.user_location}. Is that still correct?"
            location_memories = self.memory.retrieve_similar("live", n_results=3)
            for memory in location_memories:
                if any(word in memory['content'].lower() for word in ["live", "stay", "kathmandu", "nepal", "lalitpur"]):
                    return f"I remember you mentioned you live/stay there."
            return f"I don't have information about where you live, {self.user_name}."
        
        if "brother" in user_lower:
            brother_memories = self.memory.retrieve_similar("brother", n_results=3)
            for memory in brother_memories:
                if "brother" in memory['content'].lower() and "name" in memory['content'].lower():
                    return f"Your brother's name is {memory['content'].split()[-1]}."
            return "I don't have information about your brother."
        
        if any(word in user_lower for word in ["hi", "hello", "hey", "greetings"]):
            return f"Hello {self.user_name}! How can I assist you today?"
        
        if context.strip():
            return f"Hello {self.user_name}! Based on our conversations, {context.split('.')[-2] if '.' in context else 'I recall our previous discussions'}. How can I help?"
        else:
            return f"Hello {self.user_name}! I understand you're saying '{user_input[:50]}...'. How can I assist you today?"
    
    def _update_knowledge_graph(self, user_input: str, response: str, analysis: Optional[Dict] = None):
        """Update knowledge graph with new information (Category 7)"""
        name_match = re.search(r"my name is ([A-Za-z]+)", user_input, re.IGNORECASE)
        if name_match:
            name = name_match.group(1)
            self.knowledge_graph.add_fact(name, "is_a", "user", confidence=1.0)
        
        location_match = re.search(r"(?:live|stay) in ([A-Za-z\s,]+)", user_input, re.IGNORECASE)
        if location_match:
            location = location_match.group(1).strip()
            self.knowledge_graph.add_fact(self.user_name, "lives_in", location, confidence=0.9)
        
        brother_match = re.search(r"brother(?:'s)? name is ([A-Za-z]+)", user_input, re.IGNORECASE)
        if brother_match:
            brother_name = brother_match.group(1)
            self.knowledge_graph.add_fact(self.user_name, "has_brother", brother_name, confidence=0.95)
        
        if analysis and analysis.get("is_correction"):
            logger.info("Correction detected - marking previous info for review")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status (Category 5, 7, 8)"""
        return {
            "system_name": self.system_name,
            "user_name": self.user_name,
            "user_location": self.user_location,
            "conversation_count": self.conversation_count,
            "session_duration": str(datetime.now() - self.session_start_time),
            "conversation_context": self.conversation_context.get_summary(),
            "user_profile": self.user_profile.get_summary(),
            "memory": self.memory.get_memory_statistics(),
            "validation": self.validator.get_validation_statistics(),
            "learning": self.learning_manager.get_learning_status(),
            "knowledge_graph": self.knowledge_graph.get_graph_statistics(),
            "metrics": self.metrics.get_summary(),
            "model_loaded": self.model is not None,
            "background_tasks": len(self.background_tasks)
        }
    
    def run_benchmark(self) -> Dict:
        """Run benchmark suite (Category 8 & 12)"""
        logger.info("Running benchmark suite...")
        return self.benchmark.run_benchmark(self)
    
    def export_research_data(self) -> str:
        """Export all research data (Category 8 & 11)"""
        # Export metrics
        metrics_file = self.metrics.export_to_csv()
        
        # Export report
        report = self.metrics.generate_report()
        os.makedirs(str(METRICS_DIR), exist_ok=True)
        report_file = METRICS_DIR / "research_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Export user profile
        profile_export = self.user_profile.export_profile()
        
        # Export conversation history
        conversation_export = self._export_conversation_history()
        
        logger.info(f"Exported research data to {metrics_file}, {report_file}, {profile_export}, {conversation_export}")
        return str(metrics_file)
    
    def _export_conversation_history(self) -> str:
        """Export conversation history (Category 11)"""
        export_file = EXPORTS_DIR / f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "user_name": self.user_name,
            "session_start": self.session_start_time.isoformat(),
            "session_end": datetime.now().isoformat(),
            "total_conversations": self.conversation_count,
            "conversation_history": self.conversation_context.history
        }
        
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return str(export_file)
    
    def review_memories(self, limit: int = 10) -> List[Dict]:
        """Review stored memories (Category 11)"""
        return self.memory.get_recent_memories(limit)
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory (Category 11)"""
        return self.memory.delete_memory(memory_id)


class ConsolidationScheduler:
    """
    Daily consolidation scheduler (Category 6)
    Runs sleep-phase learning automatically
    """
    
    def __init__(self, learning_manager, memory_system):
        self.learning_manager = learning_manager
        self.memory = memory_system
        self.running = True
        self.last_consolidation = None
        self.interval_hours = LearningConfig.CONSOLIDATION_INTERVAL_HOURS
    
    def start(self):
        """Start consolidation scheduler"""
        logger.info(f"Consolidation scheduler started (interval: {self.interval_hours}h)")
        
        while self.running:
            try:
                # Check if consolidation should run
                if self._should_consolidate():
                    logger.info("Running scheduled consolidation...")
                    self._run_consolidation()
                
                # Sleep for 1 hour checks
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Consolidation scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _should_consolidate(self) -> bool:
        """Check if consolidation should run"""
        if self.last_consolidation is None:
            return True
        
        elapsed = (datetime.now() - self.last_consolidation).total_seconds() / 3600
        return elapsed >= self.interval_hours
    
    def _run_consolidation(self):
        """Run consolidation cycle"""
        try:
            # Step 1: Get strong memories
            strong_memories = self.memory.get_memories_by_strength(
                min_strength=LearningConfig.CONSOLIDATION_MIN_STRENGTH
            )
            
            # Step 2: Run consolidation
            result = self.learning_manager.run_consolidation()
            
            # Step 3: Forget weak memories
            forgotten_count = self.memory.forget_weak_memories(threshold=0.2)
            
            # Step 4: Update timestamp
            self.last_consolidation = datetime.now()
            
            logger.info(f"Consolidation complete: {result}, Forgot {forgotten_count} weak memories")
            
        except Exception as e:
            logger.error(f"Consolidation run failed: {e}")
    
    def stop(self):
        """Stop consolidation scheduler"""
        self.running = False
        logger.info("Consolidation scheduler stopped")


def main():
    """Main function to run the cognitive architecture"""
    print_config_summary()
    
    # Initialize chatbot
    chatbot = CognitiveChatbot(profile_name="default")
    
    while True:
        try:
            user_input = input(f"\n{chatbot.user_name}: ").strip()
            
            if user_input.lower() == 'quit':
                print(f"\nShutting down {chatbot.system_name}...")
                chatbot.export_research_data()
                print("Research data exported successfully!")
                break
            
            elif user_input.lower() == 'status':
                status = chatbot.get_system_status()
                print(f"\n{'='*70}")
                print(f"SYSTEM STATUS")
                print(f"{'='*70}")
                print(f"User: {status['user_name']}")
                print(f"Location: {status['user_location'] or 'Not set'}")
                print(f"Conversations: {status['conversation_count']}")
                print(f"Session Duration: {status['session_duration']}")
                print(f"Current Topic: {status['conversation_context']['current_topic']}")
                print(f"Model Loaded: {status['model_loaded']}")
                print(f"\nMemory:")
                print(f"  - Total Memories: {status['memory']['total_memories']}")
                print(f"  - Avg Strength: {status['memory']['average_memory_strength']:.3f}")
                print(f"\nValidation:")
                print(f"  - Verified: {status['validation']['verified_count']}")
                print(f"  - Flagged: {status['validation']['flagged_count']}")
                print(f"  - Success Rate: {status['validation']['success_rate']:.2%}")
                print(f"  - Pending: {status['validation']['pending_count']}")
                print(f"\nKnowledge Graph:")
                print(f"  - Nodes: {status['knowledge_graph']['nodes']}")
                print(f"  - Edges: {status['knowledge_graph']['edges']}")
                print(f"  - Coherence: {status['knowledge_graph']['coherence_score']:.3f}")
                print(f"\nMetrics:")
                for metric, value in status['metrics'].items():
                    print(f"  - {metric}: {value}")
                print(f"{'='*70}")
            
            elif user_input.lower() == 'benchmark':
                results = chatbot.run_benchmark()
                print(f"\nBenchmark Results:")
                print(f"  - Avg Latency: {results['avg_latency']:.2f}s")
                print(f"  - Avg Accuracy: {results['avg_accuracy']:.2%}")
                print(f"  - Total Queries: {results['total_queries']}")
            
            elif user_input.lower() == 'export':
                filepath = chatbot.export_research_data()
                print(f"\nResearch data exported to: {filepath}")
                print(f"Exports saved to: {EXPORTS_DIR}")
            
            elif user_input.lower() == 'profile':
                profile = chatbot.user_profile.get_summary()
                print(f"\n{'='*50}")
                print(f"USER PROFILE")
                print(f"{'='*50}")
                print(f"Name: {profile['name'] or 'Not set'}")
                print(f"Location: {profile['location'] or 'Not set'}")
                print(f"Total Conversations: {profile['total_conversations']}")
                print(f"Learned Facts: {profile['learned_facts_count']}")
                print(f"Corrections Made: {profile['corrections_count']}")
                print(f"Last Updated: {profile['last_updated']}")
                print(f"{'='*50}")
            
            elif user_input.lower() == 'clear':
                chatbot.conversation_context.clear()
                print("\nConversation context cleared!")
            
            elif user_input.lower() == 'memory':
                memories = chatbot.review_memories(limit=10)
                print(f"\n{'='*50}")
                print(f"RECENT MEMORIES (Last 10)")
                print(f"{'='*50}")
                for i, mem in enumerate(memories, 1):
                    print(f"{i}. {mem['content'][:100]}...")
                    print(f"   Strength: {mem.get('strength', 'N/A')}, ID: {mem['id'][:8]}")
                print(f"{'='*50}")
            
            else:
                response = chatbot.process_input(user_input)
                print(f"\n{chatbot.system_name}: {response}")
                
        except KeyboardInterrupt:
            print(f"\n\nShutting down {chatbot.system_name} gracefully...")
            chatbot.export_research_data()
            print("Research data exported successfully!")
            break
        
        except Exception as e:
            print(f"\nError: {e}")
            logger.exception("Error in main loop")
            continue


if __name__ == "__main__":
    main()













# """
# Main Chat Interface for Self-Evolving Cognitive Architecture
# Integrates all modules into a cohesive system

# Features:
# - LLM-based Query Understanding (intent detection before action)
# - Real-time data fetching (weather, PM, news, time, date)
# - Conversation context tracking for follow-up questions
# - Biologically-inspired memory with Ebbinghaus decay
# - Post-response validation pipeline
# - Sleep-phase knowledge consolidation
# - Semantic drift detection
# - Comprehensive research metrics
# - Correction handling and memory storage requests
# """

# import os
# import sys
# import json
# import logging
# import time
# import threading
# import asyncio
# import re
# from datetime import datetime
# from typing import Dict, Any, Optional, List
# from .realtime_fetcher import RealTimeFetcher
# from .query_understanding import QueryUnderstanding

# # Import all modules
# from .config import (
#     SystemConfig, ModelConfig, MemoryConfig, 
#     ValidationConfig, LearningConfig, MetricsConfig,
#     print_config_summary, MEMORY_DIR, VALIDATION_DIR, LEARNING_DIR, METRICS_DIR
# )
# from .memory import SemanticMemory
# from .validation import PostResponseValidator, ValidationScheduler
# from .learning import LearningManager
# from .knowledge_graph import KnowledgeGraph
# from .metrics import ResearchMetrics, BenchmarkSuite

# # ============================================================================
# # SUPPRESS BACKGROUND TASK LOGS FROM CONSOLE (Stops interruptions while typing)
# # ============================================================================
# logging.getLogger('validation').setLevel(logging.WARNING)
# logging.getLogger('learning').setLevel(logging.WARNING)
# logging.getLogger('knowledge_graph').setLevel(logging.WARNING)
# logging.getLogger('httpx').setLevel(logging.WARNING)
# logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
# logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
# logging.getLogger('googleapiclient').setLevel(logging.WARNING)

# # Try to import llama-cpp-python

# try:
#     from llama_cpp import Llama
#     LLAMA_CPP_AVAILABLE = True
# except ImportError:
#     LLAMA_CPP_AVAILABLE = False
#     print("Warning: llama-cpp-python not available. Install with: pip install llama-cpp-python")

# # Configure logging - File gets all logs, Console gets only important ones
# def setup_logging():
#     """Configure logging to separate console and file outputs"""
#     # Create custom formatters
#     console_formatter = logging.Formatter('%(message)s')
#     file_formatter = logging.Formatter(SystemConfig.LOG_FORMAT)
    
#     # File handler (all logs)
#     file_handler = logging.FileHandler(SystemConfig.LOG_FILE)
#     file_handler.setLevel(logging.DEBUG)
#     file_handler.setFormatter(file_formatter)
    
#     # Console handler (only important logs)
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.INFO)
#     console_handler.setFormatter(console_formatter)
    
#     # Root logger
#     root_logger = logging.getLogger()
#     root_logger.setLevel(logging.DEBUG)
#     root_logger.addHandler(file_handler)
#     root_logger.addHandler(console_handler)

# # Setup logging
# setup_logging()
# logger = logging.getLogger(__name__)


# class ConversationContext:
#     """
#     Manages conversation history and context for follow-up questions
#     Tracks topics, entities, and query types for intelligent responses
#     """
    
#     def __init__(self, max_history: int = 20):
#         self.max_history = max_history
#         self.history: List[Dict[str, Any]] = []
#         self.current_topic: Optional[str] = None
#         self.entities_mentioned: Dict[str, Any] = {}
#         self.last_query_type: str = "general"
    
#     def add_turn(self, user_input: str, response: str, query_type: str = "general",
#                  metadata: Optional[Dict] = None):
#         """Add a conversation turn to history"""
#         turn = {
#             "timestamp": datetime.now().isoformat(),
#             "user_input": user_input,
#             "response": response,
#             "query_type": query_type,
#             "metadata": metadata or {}
#         }
        
#         self.history.append(turn)
        
#         # Trim history if too long
#         if len(self.history) > self.max_history:
#             self.history = self.history[-self.max_history:]
        
#         # Update current topic
#         self._update_topic(user_input, query_type)
        
#         # Extract and track entities
#         self._extract_entities(user_input)
        
#         # Update last query type
#         self.last_query_type = query_type
        
#         logger.debug(f"Added conversation turn. Topic: {self.current_topic}, Query type: {query_type}")
    
#     def _update_topic(self, user_input: str, query_type: str):
#         """Update current topic based on conversation"""
#         topic_keywords = {
#             "weather": ["weather", "temperature", "forecast", "rain", "sunny"],
#             "location": ["live", "stay", "from", "location", "where"],
#             "personal": ["name", "brother", "sister", "family", "friend"],
#             "time": ["time", "clock", "hour", "when"],
#             "date": ["date", "day", "year", "month", "today"],
#             "politics": ["prime minister", "president", "government", "minister"],
#             "news": ["news", "happening", "recent", "latest"],
#         }
        
#         user_lower = user_input.lower()
#         for topic, keywords in topic_keywords.items():
#             if any(kw in user_lower for kw in keywords):
#                 self.current_topic = topic
#                 return
        
#         # If query type is specific, use that as topic
#         if query_type != "general":
#             self.current_topic = query_type
    
#     def _extract_entities(self, user_input: str):
#         """Extract and track entities mentioned in conversation"""
#         # Name entities
#         name_match = re.search(r"(?:my name is|i am|i'm)\s+([A-Za-z]+)", user_input, re.IGNORECASE)
#         if name_match:
#             self.entities_mentioned['user_name'] = name_match.group(1)
        
#         # Location entities
#         location_match = re.search(r"(?:live|stay|from)\s+in\s+([A-Za-z\s,]+)", user_input, re.IGNORECASE)
#         if location_match:
#             self.entities_mentioned['location'] = location_match.group(1).strip()
        
#         # Family entities
#         brother_match = re.search(r"brother(?:'s)? name is ([A-Za-z]+)", user_input, re.IGNORECASE)
#         if brother_match:
#             self.entities_mentioned['brother_name'] = brother_match.group(1)
    
#     def get_context_for_prompt(self) -> str:
#         """Build context string for LLM prompt from recent conversation"""
#         if not self.history:
#             return ""
        
#         context_parts = []
#         recent_turns = self.history[-5:]  # Last 5 turns
        
#         for turn in recent_turns:
#             context_parts.append(f"User: {turn['user_input']}")
#             context_parts.append(f"Sentrix: {turn['response']}")
        
#         return "\n".join(context_parts)
    
#     def is_follow_up_question(self, user_input: str) -> bool:
#         """Detect if current input is a follow-up to previous conversation"""
#         if not self.history:
#             return False
        
#         follow_up_indicators = [
#             "there", "that", "those", "it", "they", "them", "this",
#             "what about", "how about", "and", "also", "too", "else",
#             "what else", "tell me more", "continue", "go on",
#             "why", "because", "so", "then", "but", "however"
#         ]
        
#         user_lower = user_input.lower()
        
#         # Check for follow-up indicators
#         if any(indicator in user_lower for indicator in follow_up_indicators):
#             # Additional check: if query is short, more likely to be follow-up
#             if len(user_input.split()) < 8:
#                 return True
        
#         # Check for pronouns without clear subject
#         pronouns = ["he", "she", "it", "they", "him", "her", "them"]
#         if any(pronoun in user_lower.split() for pronoun in pronouns):
#             if len(user_input.split()) < 10:
#                 return True
        
#         return False
    
#     def get_recent_query_types(self, n: int = 3) -> List[str]:
#         """Get recent query types for context inheritance"""
#         if not self.history:
#             return ["general"]
        
#         return [turn.get('query_type', 'general') for turn in self.history[-n:]]
    
#     def clear(self):
#         """Clear conversation history"""
#         self.history = []
#         self.current_topic = None
#         self.entities_mentioned = {}
#         self.last_query_type = "general"
#         logger.info("Conversation context cleared")
    
#     def get_summary(self) -> Dict[str, Any]:
#         """Get conversation context summary"""
#         return {
#             "total_turns": len(self.history),
#             "current_topic": self.current_topic,
#             "entities_mentioned": self.entities_mentioned,
#             "last_query_type": self.last_query_type,
#             "recent_query_types": self.get_recent_query_types()
#         }


# class CognitiveChatbot:
#     """Main chatbot interface with full cognitive capabilities"""
    
#     def __init__(self):
#         logger.info("Initializing Self-Evolving Cognitive Architecture...")
        
#         # Initialize all components
#         self.memory = SemanticMemory()
#         self.validator = PostResponseValidator()
#         self.learning_manager = LearningManager(self.memory, self.validator)
#         self.knowledge_graph = KnowledgeGraph()
#         self.metrics = ResearchMetrics()
#         self.benchmark = BenchmarkSuite(self.metrics)
#         self.realtime_fetcher = RealTimeFetcher()
#         self.user_location = None  # Will be updated when user tells us
        
#         # Initialize query analyzer (CRITICAL - understands queries before acting)
#         self.query_analyzer = QueryUnderstanding()
        
#         # Initialize conversation context manager
#         self.conversation_context = ConversationContext(max_history=20)
        
#         # Initialize model
#         self.model = None
#         self.generate_func = None
#         self._initialize_model()
        
#         # System state
#         self.system_name = SystemConfig.SYSTEM_NAME
#         self.user_name = SystemConfig.DEFAULT_USER_NAME
#         self.conversation_count = 0
        
#         # Background tasks
#         self.validation_scheduler = ValidationScheduler(self.validator)
#         self.background_tasks = []
        
#         # Load existing knowledge
#         self._load_existing_knowledge()
        
#         # Start background processes
#         self._start_background_processes()
        
#         logger.info("Cognitive chatbot initialized successfully")
    
#     def _initialize_model(self):
#         """Initialize the LLM model"""
#         if not LLAMA_CPP_AVAILABLE:
#             logger.warning("llama-cpp-python not available, using rule-based responses")
#             return
        
#         if not os.path.exists(ModelConfig.MODEL_PATH):
#             logger.warning(f"Model file not found at: {ModelConfig.MODEL_PATH}")
#             logger.warning("Using rule-based responses fallback")
#             return
        
#         try:
#             logger.info(f"Loading GGUF model: {ModelConfig.MODEL_PATH}")
#             self.model = Llama(  # type: ignore
#                 model_path=ModelConfig.MODEL_PATH,
#                 n_ctx=ModelConfig.CONTEXT_SIZE,
#                 n_threads=ModelConfig.N_THREADS,
#                 n_batch=ModelConfig.N_BATCH,
#                 use_mmap=ModelConfig.USE_MMAP,
#                 verbose=False
#             )
            
#             def generate_text(prompt: str, max_tokens: Optional[int] = None) -> str:
#                 if self.model is None:
#                     raise RuntimeError("Model not loaded")
#                 max_tokens = max_tokens or ModelConfig.MAX_NEW_TOKENS
#                 output = self.model(
#                     prompt,
#                     max_tokens=max_tokens,
#                     temperature=ModelConfig.TEMPERATURE,
#                     top_p=ModelConfig.TOP_P,
#                     stop=ModelConfig.STOP_SEQUENCES,
#                     stream=False
#                 )
#                 if isinstance(output, dict) and 'choices' in output:
#                     return output['choices'][0]['text'].strip()
#                 return str(output).strip()
            
#             self.generate_func = generate_text
#             logger.info("GGUF model loaded successfully")
            
#         except Exception as e:
#             logger.error(f"Error loading model: {e}")
#             logger.warning("Using rule-based responses fallback")
#             self.model = None
#             self.generate_func = None
    
#     def _load_existing_knowledge(self):
#         """Load existing knowledge from files"""
#         verified_dir = self.validator.verified_dir
#         if verified_dir.exists():
#             count = 0
#             for filepath in verified_dir.glob("*.json"):
#                 try:
#                     with open(filepath, 'r') as f:
#                         entry = json.load(f)
#                     self.memory.add_memory(
#                         f"User: {entry['query']}\n{self.system_name}: {entry['response']}",
#                         metadata={"source": "verified", "concept": entry.get('metadata', {}).get('concept', 'general')}
#                     )
#                     count += 1
#                 except Exception as e:
#                     logger.warning(f"Could not load entry {filepath}: {e}")
#             logger.info(f"Loaded {count} verified interactions from cache")
    
#     def _start_background_processes(self):
#         """Start background tasks"""
#         # Start validation scheduler
#         if getattr(ValidationConfig, 'ASYNC_VALIDATION', True):
#             async def start_scheduler():
#                 await self.validation_scheduler.start()
            
#             scheduler_thread = threading.Thread(
#                 target=lambda: asyncio.run(start_scheduler()),
#                 daemon=True
#             )
#             scheduler_thread.start()
#             self.background_tasks.append(scheduler_thread)
#             logger.info("Validation scheduler started")
        
#         # Start consolidation checker
#         if getattr(LearningConfig, 'BACKGROUND_CONSOLIDATION', True):
#             def consolidation_checker():
#                 while True:
#                     time.sleep(3600)  # Check every hour
#                     if self.learning_manager.should_consolidate():
#                         logger.info("Scheduled consolidation triggered")
#                         self.learning_manager.run_consolidation()
            
#             consolidation_thread = threading.Thread(target=consolidation_checker, daemon=True)
#             consolidation_thread.start()
#             self.background_tasks.append(consolidation_thread)
#             logger.info("Consolidation checker started")
    
#     def process_input(self, user_input: str) -> str:
#         """Process user input with FULL query understanding"""
#         start_time = time.time()
#         self.conversation_count += 1
        
#         # ========================================================================
#         # STEP 1: UNDERSTAND THE QUERY (before any action!)
#         # ========================================================================
#         user_profile = {
#             "name": self.user_name,
#             "location": self.user_location
#         }
        
#         query_analysis = self.query_analyzer.analyze_query(
#             user_input,
#             self.conversation_context.history,
#             user_profile
#         )
        
#         logger.info(f"Query understood: {query_analysis['intent']} (confidence: {query_analysis['confidence']:.2f})")
#         logger.info(f"Needs realtime: {query_analysis['needs_realtime']} ({query_analysis['realtime_type']})")
#         logger.info(f"Is correction: {query_analysis['is_correction']}")
#         logger.info(f"Is follow-up: {query_analysis['is_follow_up']}")
        
#         # ========================================================================
#         # STEP 2: Extract user info (name, location, etc.)
#         # ========================================================================
#         self._extract_user_info(user_input)
        
#         # ========================================================================
#         # STEP 3: Handle based on understood intent
#         # ========================================================================
        
#         # CASE A: User wants us to REMEMBER something
#         if query_analysis["intent"] == "memory_storage":
#             response = self._handle_memory_storage(user_input, query_analysis)
        
#         # CASE B: User is CORRECTING information
#         elif query_analysis["intent"] == "correction":
#             response = self._handle_correction(user_input, query_analysis)
        
#         # CASE C: User asking about PERSONAL INFO
#         elif query_analysis["intent"] == "personal_info":
#             response = self._handle_personal_info(user_input, query_analysis)
        
#         # CASE D: User needs REAL-TIME data
#         elif query_analysis["needs_realtime"]:
#             logger.info(f"Fetching real-time data: {query_analysis['realtime_type']}")
#             realtime_data = self.realtime_fetcher.fetch(
#                 user_input,
#                 self.user_location,
#                 self.conversation_context.history
#             )
            
#             if realtime_data.get('success'):
#                 response = self._generate_response_with_realtime(
#                     user_input, realtime_data, query_analysis['realtime_type']
#                 )
#             else:
#                 response = self._generate_response_with_context(user_input, query_analysis)
        
#         # CASE E: Greeting
#         elif query_analysis["intent"] == "greeting":
#             response = self._handle_greeting(user_input)
        
#         # CASE F: Math calculation
#         elif query_analysis["intent"] == "math_calculation":
#             response = self._handle_math(user_input, query_analysis)
        
#         # CASE G: Confirmation (yes/no to previous)
#         elif query_analysis["intent"] == "confirmation":
#             response = self._handle_confirmation(user_input, query_analysis)
        
#         # CASE H: Normal conversation with context
#         else:
#             response = self._generate_response_with_context(user_input, query_analysis)
        
#         # ========================================================================
#         # STEP 4: Post-response validation (async)
#         # ========================================================================
#         if self.validator.should_validate(response):
#             entry_id = self.validator.store_pending(
#                 user_input, response,
#                 metadata={
#                     "user": self.user_name,
#                     "conversation_count": self.conversation_count,
#                     "query_intent": query_analysis["intent"],
#                     "query_type": query_analysis["realtime_type"]
#                 }
#             )
#             logger.debug(f"Stored for validation: {entry_id[:8]}")
        
#         # ========================================================================
#         # STEP 5: Store interaction in memory
#         # ========================================================================
#         self.memory.add_memory(
#             f"User: {user_input}\n{self.system_name}: {response}",
#             metadata={
#                 "input": user_input,
#                 "response": response,
#                 "timestamp": datetime.now().isoformat(),
#                 "source": "interaction",
#                 "intent": query_analysis["intent"],
#                 "query_type": query_analysis["realtime_type"]
#             }
#         )
        
#         # ========================================================================
#         # STEP 6: Update conversation context
#         # ========================================================================
#         self.conversation_context.add_turn(
#             user_input, 
#             response, 
#             query_analysis["realtime_type"],
#             metadata={
#                 "location": self.user_location, 
#                 "user_name": self.user_name,
#                 "intent": query_analysis["intent"]
#             }
#         )
        
#         # ========================================================================
#         # STEP 7: Update knowledge graph
#         # ========================================================================
#         self._update_knowledge_graph(user_input, response, query_analysis)
        
#         # ========================================================================
#         # STEP 8: Log metrics
#         # ========================================================================
#         latency = time.time() - start_time
#         self.metrics.log("response_latency", latency, {
#             "input_length": len(user_input),
#             "intent": query_analysis["intent"],
#             "is_follow_up": query_analysis["is_follow_up"],
#             "needs_realtime": query_analysis["needs_realtime"]
#         })
        
#         logger.info(f"Processed input in {latency:.2f}s")
        
#         return response
    
#     def _extract_user_info(self, user_input: str):
#         """Extract user information from input"""
#         # Name pattern
#         name_pattern = r"(?:my name is|i am|i'm)\s+([A-Za-z]+)"
#         matches = re.search(name_pattern, user_input, re.IGNORECASE)
#         if matches:
#             old_name = self.user_name
#             self.user_name = matches.group(1)
#             logger.info(f"Updated user name: {old_name} -> {self.user_name}")
#             self.knowledge_graph.add_fact(self.user_name, "is_a", "user", confidence=1.0)
        
#         # Location pattern - Better extraction
#         location_patterns = [
#             r"(?:live|stay|reside)\s+(?:in\s+)?([A-Za-z\s,]+?)(?:\.|,|!|\?|$)",
#             r"(?:from)\s+([A-Za-z\s,]+?)(?:\.|,|!|\?|$)",
#         ]
        
#         for pattern in location_patterns:
#             location_match = re.search(pattern, user_input, re.IGNORECASE)
#             if location_match:
#                 location = location_match.group(1).strip()
#                 # Filter out common false positives
#                 if location.lower() not in ['the', 'a', 'an', 'here', 'there']:
#                     self.user_location = location
#                     logger.info(f"Updated user location: {self.user_location}")
#                     self.knowledge_graph.add_fact(
#                         self.user_name, "lives_in", self.user_location, confidence=0.95
#                     )
#                     # Also store in memory
#                     self.memory.add_memory(
#                         f"{self.user_name} lives in {self.user_location}",
#                         metadata={"source": "extracted", "concept": "location", "importance": 5}
#                     )
#                     break
    
#     def _handle_memory_storage(self, user_input: str, analysis: Dict) -> str:
#         """Handle when user wants system to remember something"""
#         # Extract what to remember
#         entities = analysis["entities"]
        
#         # Store in memory with high importance
#         self.memory.add_memory(
#             user_input,
#             metadata={
#                 "source": "user_request",
#                 "importance": 5,
#                 "type": "user_requested_memory"
#             }
#         )
        
#         # Also update knowledge graph if applicable
#         if entities["person_names"]:
#             for name in entities["person_names"]:
#                 self.knowledge_graph.add_fact(self.user_name, "remembers", name, confidence=0.9)
        
#         return f"Got it! I'll remember that for you, {self.user_name}."
    
#     def _handle_correction(self, user_input: str, analysis: Dict) -> str:
#         """Handle when user is correcting information"""
#         # Find what they're correcting from context
#         context_refs = analysis["context_references"]
        
#         if context_refs and self.conversation_context.history:
#             # Get the turn being corrected
#             ref_turn = self.conversation_context.history[context_refs[0]["turn_index"]]
            
#             # Acknowledge the correction
#             response = f"Thank you for the correction! I've noted that {user_input}. I'll update my understanding."
            
#             # Store the correction in memory
#             self.memory.add_memory(
#                 f"CORRECTION: {user_input}",
#                 metadata={
#                     "source": "correction",
#                     "corrected_from": ref_turn.get("response", "")[:100],
#                     "importance": 5
#                 }
#             )
            
#             return response
#         else:
#             return f"Thank you for correcting me! I've noted: {user_input}"
    
#     def _handle_personal_info(self, user_input: str, analysis: Dict) -> str:
#         """Handle questions about user's personal information"""
#         # Check stored info first
#         if "name" in user_input.lower() and "my" in user_input.lower():
#             if self.user_name != SystemConfig.DEFAULT_USER_NAME:
#                 return f"Your name is {self.user_name}, as you told me."
#             return "I don't recall you telling me your name yet."
        
#         if any(word in user_input.lower() for word in ["live", "location", "where"]):
#             if self.user_location:
#                 return f"You mentioned you live in {self.user_location}."
#             return "I don't have information about where you live."
        
#         # Check memory for other personal info
#         retrieved = self.memory.retrieve_similar(user_input, n_results=3)
#         if retrieved:
#             return f"Based on our conversations: {retrieved[0]['content'][:150]}..."
        
#         return "I don't have that information about you yet."
    
#     def _handle_greeting(self, user_input: str) -> str:
#         """Handle greetings"""
#         greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon"]
        
#         for greeting in greetings:
#             if greeting in user_input.lower():
#                 return f"Hello {self.user_name}! How can I assist you today?"
        
#         # Check for "how are you"
#         if "how are you" in user_input.lower():
#             return f"I'm doing well, thank you for asking, {self.user_name}! How can I help you today?"
        
#         return f"Hello {self.user_name}! How can I assist you?"
    
#     def _handle_math(self, user_input: str, analysis: Dict) -> str:
#         """Handle mathematical calculations"""
#         numbers = re.findall(r'\d+', user_input)
        
#         if '+' in user_input and len(numbers) >= 2:
#             result = sum(int(n) for n in numbers)
#             return f"The answer is {result}."
#         elif '-' in user_input and len(numbers) >= 2:
#             result = int(numbers[0]) - int(numbers[1])
#             return f"The answer is {result}."
#         elif ('*' in user_input or 'x' in user_input.lower()) and len(numbers) >= 2:
#             result = int(numbers[0]) * int(numbers[1])
#             return f"The answer is {result}."
#         elif '/' in user_input and len(numbers) >= 2:
#             result = int(numbers[0]) / int(numbers[1])
#             return f"The answer is {result:.2f}."
        
#         return "I can help with basic math calculations. What would you like to calculate?"
    
#     def _handle_confirmation(self, user_input: str, analysis: Dict) -> str:
#         """Handle yes/no confirmations"""
#         query_lower = user_input.lower()
        
#         # Check if affirmative
#         affirmative = ["yes", "yeah", "yep", "correct", "right", "exactly", "that's right"]
#         negative = ["no", "nah", "nope", "wrong", "incorrect", "not really"]
        
#         if any(word in query_lower for word in affirmative):
#             # Check what they're confirming
#             context_refs = analysis["context_references"]
#             if context_refs:
#                 return "Great! I've noted your confirmation."
#             return "Understood!"
        
#         elif any(word in query_lower for word in negative):
#             # Check what they're denying
#             context_refs = analysis["context_references"]
#             if context_refs:
#                 return "I understand. Let me update my information."
#             return "Got it, thanks for clarifying!"
        
#         return "Understood!"
    
#     def _generate_response_with_realtime(self, user_input: str, realtime_data: Dict, query_type: str) -> str:
#         """Generate response using real-time fetched data - CLEAN outputs only"""
        
#         if query_type == "weather":
#             if realtime_data.get('success'):
#                 return f"The current weather in {realtime_data.get('location')} is {realtime_data.get('temperature')} and {realtime_data.get('condition')}. (Source: {realtime_data.get('source')})"
#             return "I couldn't fetch weather data right now."
        
#         elif query_type == "time":
#             if realtime_data.get('success'):
#                 return f"The current time is {realtime_data.get('time')}. (Source: {realtime_data.get('source')})"
#             return f"The current time is {datetime.now().strftime('%I:%M %p')}."
        
#         elif query_type == "date":
#             if realtime_data.get('success'):
#                 return f"Today is {realtime_data.get('formatted')}. (Source: {realtime_data.get('source')})"
#             return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."
        
#         elif query_type == "pm":
#             if realtime_data.get('success'):
#                 return f"The current Prime Minister is {realtime_data.get('pm_name')}. (Source: {realtime_data.get('source')})"
#             return "I couldn't fetch that information right now."
        
#         elif query_type == "news":
#             if realtime_data.get('success'):
#                 summary = realtime_data.get('summary', 'No news available')
#                 source = realtime_data.get('source', 'N/A')
#                 return f"Here's the latest: {summary}\n\n(Source: {source})"
#             return "I couldn't fetch current news."
        
#         else:
#             return self._rule_based_response(user_input, "")
    
#     def _generate_response_with_context(self, user_input: str, analysis: Dict) -> str:
#         """Generate response using model with full context AND query understanding"""
#         # Build comprehensive context
#         conversation_history = self.conversation_context.get_context_for_prompt()
        
#         # Retrieve relevant memories
#         retrieved_memories = self.memory.retrieve_similar(user_input, n_results=5)
#         memory_context = self._build_context(retrieved_memories)
        
#         # Add query understanding to prompt
#         understanding_context = f"""
# Query Analysis:
# - Intent: {analysis['intent']}
# - Is Follow-up: {analysis['is_follow_up']}
# - Is Correction: {analysis['is_correction']}
# - Entities Found: {analysis['entities']}
# - Context References: {len(analysis['context_references'])} previous turns
# """
        
#         # Create comprehensive prompt
#         if conversation_history.strip():
#             prompt = f"""You are {self.system_name}, a helpful AI assistant with memory and context awareness.

# PREVIOUS CONVERSATION:
# {conversation_history}

# RELEVANT INFORMATION FROM MEMORY:
# {memory_context}

# QUERY UNDERSTANDING:
# {understanding_context}

# IMPORTANT: 
# - If this is a follow-up question, use context from previous conversation
# - If user is correcting you, acknowledge the correction
# - If asking about personal info, check what you know about the user
# - Be natural and conversational

# CURRENT QUERY:
# User: {user_input}
# {self.system_name}:"""
#         else:
#             prompt = f"""You are {self.system_name}, a helpful AI assistant.

# RELEVANT INFORMATION FROM MEMORY:
# {memory_context}

# QUERY UNDERSTANDING:
# {understanding_context}

# User: {user_input}
# {self.system_name}:"""
        
#         try:
#             # Use model if available
#             if self.generate_func and self.model is not None and LLAMA_CPP_AVAILABLE:
#                 response = self.generate_func(prompt, max_tokens=ModelConfig.MAX_NEW_TOKENS)
#                 return response
            
#             # Rule-based fallback
#             return self._rule_based_response(user_input, memory_context)
            
#         except Exception as e:
#             logger.error(f"Error generating response: {e}")
#             return self._rule_based_response(user_input, memory_context)
    
#     def _build_context(self, retrieved_memories: List[Dict]) -> str:
#         """Build context from retrieved memories"""
#         context_parts = []
        
#         for memory in retrieved_memories:
#             content = memory['content']
#             # Clean up
#             clean_content = content.replace('User:', 'Previous user said:').replace('Assistant:', f'{self.system_name} replied:').replace('Sentrix:', f'{self.system_name} replied:')
#             context_parts.append(clean_content)
        
#         return "\n".join(context_parts) if context_parts else f"No prior conversations with {self.user_name}."
    
#     def _rule_based_response(self, user_input: str, context: str) -> str:
#         """Rule-based response generation fallback"""
#         user_lower = user_input.lower()
        
#         # Name queries
#         if "name" in user_lower and ("what" in user_lower or "who" in user_lower):
#             if "your" in user_lower or "you" in user_lower:
#                 return f"My name is {self.system_name}, as you requested."
#             elif "my" in user_lower:
#                 if self.user_name != SystemConfig.DEFAULT_USER_NAME:
#                     return f"Your name is {self.user_name}, as you previously told me."
#                 return "I don't recall you telling me your name yet."
        
#         # Location queries
#         if any(word in user_lower for word in ["live", "stay", "from", "location"]):
#             if self.user_location:
#                 return f"You mentioned you live in {self.user_location}. Is that still correct?"
#             location_memories = self.memory.retrieve_similar("live", n_results=3)
#             for memory in location_memories:
#                 if any(word in memory['content'].lower() for word in ["live", "stay", "kathmandu", "nepal", "lalitpur"]):
#                     return f"I remember you mentioned you live/stay there."
#             return f"I don't have information about where you live, {self.user_name}."
        
#         # Brother/family queries
#         if "brother" in user_lower:
#             brother_memories = self.memory.retrieve_similar("brother", n_results=3)
#             for memory in brother_memories:
#                 if "brother" in memory['content'].lower() and "name" in memory['content'].lower():
#                     return f"Your brother's name is {memory['content'].split()[-1]}."
#             return "I don't have information about your brother."
        
#         # Greeting
#         if any(word in user_lower for word in ["hi", "hello", "hey", "greetings"]):
#             return f"Hello {self.user_name}! How can I assist you today?"
        
#         # General response
#         if context.strip():
#             return f"Hello {self.user_name}! Based on our conversations, {context.split('.')[-2] if '.' in context else 'I recall our previous discussions'}. How can I help?"
#         else:
#             return f"Hello {self.user_name}! I understand you're saying '{user_input[:50]}...'. How can I assist you today?"
    
#     def _update_knowledge_graph(self, user_input: str, response: str, analysis: Optional[Dict] = None):
#         """Update knowledge graph with new information (enhanced with query analysis)"""
#         # Check for name statements
#         name_match = re.search(r"my name is ([A-Za-z]+)", user_input, re.IGNORECASE)
#         if name_match:
#             name = name_match.group(1)
#             self.knowledge_graph.add_fact(name, "is_a", "user", confidence=1.0)
        
#         # Check for location statements
#         location_match = re.search(r"(?:live|stay) in ([A-Za-z\s,]+)", user_input, re.IGNORECASE)
#         if location_match:
#             location = location_match.group(1).strip()
#             self.knowledge_graph.add_fact(self.user_name, "lives_in", location, confidence=0.9)
        
#         # Check for family statements
#         brother_match = re.search(r"brother(?:'s)? name is ([A-Za-z]+)", user_input, re.IGNORECASE)
#         if brother_match:
#             brother_name = brother_match.group(1)
#             self.knowledge_graph.add_fact(self.user_name, "has_brother", brother_name, confidence=0.95)
        
#         # Handle corrections
#         if analysis and analysis.get("is_correction"):
#             logger.info("Correction detected - marking previous info for review")
    
#     def get_system_status(self) -> Dict[str, Any]:
#         """Get comprehensive system status"""
#         return {
#             "system_name": self.system_name,
#             "user_name": self.user_name,
#             "user_location": self.user_location,
#             "conversation_count": self.conversation_count,
#             "conversation_context": self.conversation_context.get_summary(),
#             "memory": self.memory.get_memory_statistics(),
#             "validation": self.validator.get_validation_statistics(),
#             "learning": self.learning_manager.get_learning_status(),
#             "knowledge_graph": self.knowledge_graph.get_graph_statistics(),
#             "model_loaded": self.model is not None,
#             "background_tasks": len(self.background_tasks)
#         }
    
#     def run_benchmark(self) -> Dict:
#         """Run benchmark suite"""
#         logger.info("Running benchmark suite...")
#         return self.benchmark.run_benchmark(self)
    
#     def export_research_data(self) -> str:
#         """Export all research data"""
#         # Export metrics
#         metrics_file = self.metrics.export_to_csv()
        
#         # Export report
#         report = self.metrics.generate_report()
#         os.makedirs(str(METRICS_DIR), exist_ok=True)
#         report_file = METRICS_DIR / "research_report.json"
#         with open(report_file, 'w') as f:
#             json.dump(report, f, indent=2)
        
#         logger.info(f"Exported research data to {metrics_file} and {report_file}")
#         return str(metrics_file)


# def main():
#     """Main function to run the cognitive architecture"""
#     print_config_summary()
    
#     # Initialize chatbot
#     chatbot = CognitiveChatbot()
    
#     print("\n" + "=" * 70)
#     print(f"  {chatbot.system_name} - Self-Evolving Cognitive Architecture")
#     print("  Research-Ready Implementation with:")
#     print("  • Ebbinghaus Forgetting Curve & RIF")
#     print("  • Post-Response Validation Pipeline")
#     print("  • Sleep-Phase Knowledge Consolidation")
#     print("  • Semantic Drift Detection")
#     print("  • Comprehensive Research Metrics")
#     print("  • Real-Time Data Fetching (Weather, PM, News)")
#     print("  • Conversation Context & Follow-up Questions")
#     print("  • LLM-Based Query Understanding")
#     print("=" * 70)
#     print("\nCommands:")
#     print("  'status'    - Show system status")
#     print("  'benchmark' - Run benchmark suite")
#     print("  'export'    - Export research data")
#     print("  'clear'     - Clear conversation context")
#     print("  'quit'      - Exit the system")
#     print("=" * 70)
    
#     while True:
#         try:
#             user_input = input(f"\n{chatbot.user_name}: ").strip()
            
#             if user_input.lower() == 'quit':
#                 print(f"\nShutting down {chatbot.system_name}...")
#                 chatbot.export_research_data()
#                 break
            
#             elif user_input.lower() == 'status':
#                 status = chatbot.get_system_status()
#                 print(f"\n{'='*50}")
#                 print(f"SYSTEM STATUS")
#                 print(f"{'='*50}")
#                 print(f"User: {status['user_name']}")
#                 print(f"Location: {status['user_location'] or 'Not set'}")
#                 print(f"Conversations: {status['conversation_count']}")
#                 print(f"Current Topic: {status['conversation_context']['current_topic']}")
#                 print(f"Model Loaded: {status['model_loaded']}")
#                 print(f"\nMemory:")
#                 print(f"  - Total Memories: {status['memory']['total_memories']}")
#                 print(f"  - Avg Strength: {status['memory']['average_memory_strength']:.3f}")
#                 print(f"\nValidation:")
#                 print(f"  - Verified: {status['validation']['verified_count']}")
#                 print(f"  - Success Rate: {status['validation']['success_rate']:.2%}")
#                 print(f"\nKnowledge Graph:")
#                 print(f"  - Nodes: {status['knowledge_graph']['nodes']}")
#                 print(f"  - Coherence: {status['knowledge_graph']['coherence_score']:.3f}")
#                 print(f"{'='*50}")
            
#             elif user_input.lower() == 'benchmark':
#                 results = chatbot.run_benchmark()
#                 print(f"\nBenchmark Results:")
#                 print(f"  - Avg Latency: {results['avg_latency']:.2f}s")
#                 print(f"  - Avg Accuracy: {results['avg_accuracy']:.2%}")
#                 print(f"  - Total Queries: {results['total_queries']}")
            
#             elif user_input.lower() == 'export':
#                 filepath = chatbot.export_research_data()
#                 print(f"\nResearch data exported to: {filepath}")
            
#             elif user_input.lower() == 'clear':
#                 chatbot.conversation_context.clear()
#                 print("\nConversation context cleared!")
            
#             else:
#                 response = chatbot.process_input(user_input)
#                 print(f"\n{chatbot.system_name}: {response}")
                
#         except KeyboardInterrupt:
#             print(f"\n\nShutting down {chatbot.system_name} gracefully...")
#             chatbot.export_research_data()
#             break
        
#         except Exception as e:
#             print(f"\nError: {e}")
#             logger.exception("Error in main loop")
#             continue


# if __name__ == "__main__":
#     main()