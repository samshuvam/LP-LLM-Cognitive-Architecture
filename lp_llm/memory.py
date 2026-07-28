"""
LP-LLM Cognitive Architecture Component
Authored by Shuvam (https://github.com/samshuvam)
"""

__author__ = "Shuvam (https://github.com/samshuvam)"

"""
Advanced Semantic Memory System
Production-Ready with All Features

Features:
- Ebbinghaus forgetting curve
- Retrieval-Induced Forgetting (RIF)
- Multi-tier memory (ephemeral, working, long-term)
- Concept frequency tracking
- Importance-based retention
- ChromaDB integration with proper metadata handling
- Auto-save and persistence
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import re
from collections import defaultdict
import pickle
from .config import MemoryConfig, MEMORY_DIR

logger = logging.getLogger(__name__)

class MemoryStrengthCalculator:
    """Calculate memory strength using Ebbinghaus-inspired decay"""
    
    @staticmethod
    def calculate(
        initial_strength: float = 1.0,
        time_elapsed_hours: float = 0,
        importance_score: float = 1.0,
        retrieval_count: int = 0,
        validation_success_count: int = 0,
        last_retrieved: Optional[datetime] = None
    ) -> float:
        """
        Calculate current memory strength using biologically-inspired formula
        
        Memory strength decays exponentially but is reinforced by:
        - High importance scores
        - Frequent retrieval (spacing effect)
        - Successful validations
        - Recent access
        """
        # Base exponential decay (Ebbinghaus curve)
        half_life = MemoryConfig.MEMORY_HALF_LIFE_HOURS
        base_decay = np.exp(-np.log(2) * time_elapsed_hours / half_life)
        
        # Importance reinforcement
        importance_factor = 1 + (MemoryConfig.IMPORTANCE_WEIGHT * importance_score)
        
        # Retrieval reinforcement (spacing effect)
        retrieval_factor = 1 + (MemoryConfig.RETRIEVAL_WEIGHT * min(retrieval_count, 20))
        
        # Validation reinforcement
        validation_factor = 1 + (MemoryConfig.VALIDATION_WEIGHT * validation_success_count)
        
        # Recency bonus
        recency_factor = 1.0
        if last_retrieved:
            hours_since_retrieval = (datetime.now() - last_retrieved).total_seconds() / 3600
            if hours_since_retrieval < 24:
                recency_factor = 1.2
            elif hours_since_retrieval < 72:
                recency_factor = 1.1
        
        # Calculate final strength
        strength = initial_strength * base_decay * importance_factor * retrieval_factor * validation_factor * recency_factor
        
        return min(1.0, max(0.0, strength))


class RetrievalInducedForgetting:
    """Implements Retrieval-Induced Forgetting (RIF) mechanism"""
    
    def __init__(self):
        self.suppressed_memories = {}
        self.suppression_duration = timedelta(hours=2)
    
    def suppress_competitors(self, retrieved_memory: Dict, all_memories: List[Dict]) -> List[str]:
        """When retrieving a memory, suppress competing memories"""
        suppressed_ids = []
        retrieved_concepts = set(retrieved_memory.get('metadata', {}).get('concepts', []))
        
        for memory in all_memories:
            if memory['id'] == retrieved_memory['id']:
                continue
            
            memory_concepts = set(memory.get('metadata', {}).get('concepts', []))
            
            if len(retrieved_concepts) > 0 and len(memory_concepts) > 0:
                overlap = len(retrieved_concepts & memory_concepts) / min(len(retrieved_concepts), len(memory_concepts))
                
                if overlap > MemoryConfig.RIF_CONCEPT_OVERLAP_THRESHOLD:
                    suppressed_ids.append(memory['id'])
                    self.suppressed_memories[memory['id']] = datetime.now() + self.suppression_duration
        
        return suppressed_ids
    
    def is_suppressed(self, memory_id: str) -> bool:
        """Check if a memory is currently suppressed"""
        if memory_id not in self.suppressed_memories:
            return False
        
        if datetime.now() > self.suppressed_memories[memory_id]:
            del self.suppressed_memories[memory_id]
            return False
        
        return True
    
    def get_suppression_status(self) -> Dict:
        """Get current suppression status"""
        return {
            "suppressed_count": len(self.suppressed_memories),
            "suppressed_ids": list(self.suppressed_memories.keys())
        }


class SemanticMemory:
    """Advanced semantic memory system with all cognitive features"""
    
    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_dir = persist_directory or str(MEMORY_DIR)
        
        # Initialize ChromaDB client with telemetry disabled
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        logger.debug(f"Loading embedding model: {MemoryConfig.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(MemoryConfig.EMBEDDING_MODEL)
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection(MemoryConfig.COLLECTION_NAME)
        except:
            self.collection = self.chroma_client.create_collection(
                name=MemoryConfig.COLLECTION_NAME,
                metadata={"hnsw:space": MemoryConfig.DISTANCE_METRIC}
            )
        
        # Initialize cognitive components
        self.strength_calculator = MemoryStrengthCalculator()
        self.rif = RetrievalInducedForgetting()
        
        # Load metadata
        self.metadata_file = os.path.join(self.persist_dir, "memory_metadata.json")
        self.concepts = self._load_concepts()
        self.memory_metadata = self._load_memory_metadata()
        
        logger.debug(f"Semantic memory initialized with {len(self.concepts)} tracked concepts")
    
    def _load_concepts(self) -> Dict:
        """Load concept tracking from file"""
        metadata_path = os.path.join(self.persist_dir, "concepts.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_concepts(self):
        """Save concept tracking to file"""
        metadata_path = os.path.join(self.persist_dir, "concepts.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.concepts, f, indent=2)
    
    def _load_memory_metadata(self) -> Dict:
        """Load per-memory metadata"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_memory_metadata(self):
        """Save per-memory metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.memory_metadata, f, indent=2)
    
    def add_memory(
        self, 
        text: str, 
        metadata: Optional[Dict] = None,
        tier: str = "working"
    ) -> str:
        """Add a new memory with ChromaDB-compatible metadata"""
        if metadata is None:
            metadata = {}
        
        try:
            # Generate embeddings
            embedding = self.embedding_model.encode([text])[0].tolist()
            
            # Extract concepts automatically
            concepts = self._extract_concepts(text)
            
            # Calculate initial importance score
            importance_score = self._calculate_importance(text, metadata)
            
            # Update concept frequency
            for concept in concepts:
                if concept not in self.concepts:
                    self.concepts[concept] = {
                        "count": 0,
                        "first_seen": datetime.now().isoformat(),
                        "last_seen": datetime.now().isoformat(),
                        "associated_memory_ids": []
                    }
                self.concepts[concept]["count"] += 1
                self.concepts[concept]["last_seen"] = datetime.now().isoformat()
            
            # Generate unique ID
            memory_id = hashlib.md5((text + str(datetime.now())).encode()).hexdigest()
            
            # Convert concepts list to comma-separated string (ChromaDB compatible)
            concepts_str = ",".join(concepts) if concepts else "general"
            
            # Ensure all metadata values are ChromaDB-compatible types
            storage_metadata = {
                "timestamp": datetime.now().isoformat(),
                "concepts": concepts_str,
                "source": metadata.get("source", "user") or "user",
                "importance": float(importance_score),
                "tier": tier,
                "initial_strength": 1.0,
                "current_strength": 1.0,
                "retrieval_count": 0,
                "validation_success_count": 0,
                "last_retrieved": datetime.now().isoformat(),
            }
            
            # Add only simple string/int/float values from user metadata
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    storage_metadata[key] = value
                elif value is not None:
                    storage_metadata[key] = str(value)
            
            # Store metadata locally (can be complex)
            self.memory_metadata[memory_id] = {
                "timestamp": datetime.now().isoformat(),
                "concepts": concepts,
                "source": metadata.get("source", "user"),
                "importance": importance_score,
                "tier": tier,
                "initial_strength": 1.0,
                "current_strength": 1.0,
                "retrieval_count": 0,
                "validation_success_count": 0,
                "last_retrieved": datetime.now().isoformat(),
                **metadata
            }
            
            # Add to ChromaDB
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[memory_id],
                metadatas=[storage_metadata]
            )
            
            # Update concept associations
            for concept in concepts:
                if memory_id not in self.concepts[concept]["associated_memory_ids"]:
                    self.concepts[concept]["associated_memory_ids"].append(memory_id)
            
            # Save metadata
            self._save_concepts()
            self._save_memory_metadata()
            
            logger.debug(f"Added {tier} memory: {text[:50]}... (ID: {memory_id[:8]})")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            raise
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text using multiple strategies"""
        concepts = []
        
        # Strategy 1: Capitalized words
        capitalized = re.findall(r'\b[A-Z][A-Za-z]*\b', text)
        concepts.extend(capitalized)
        
        # Strategy 2: Key phrases
        key_phrases = [
            r'(?:my name is|i am|i\'m)\s+([A-Za-z]+)',
            r'(?:live|stay|from)\s+in\s+([A-Za-z\s,]+)',
            r'(?:like|love|prefer|favorite)\s+(?:to\s+)?([A-Za-z\s]+)',
            r'(?:brother|sister|parent|friend)\s+(?:named|is)\s+([A-Za-z]+)'
        ]
        for pattern in key_phrases:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.extend(matches)
        
        # Strategy 3: Technical terms
        technical = re.findall(r'\b\d+\w*\b|\b\w+(?:ing|ed|tion|ity)\b', text)
        concepts.extend(technical)
        
        # Clean and deduplicate
        concepts = list(set([c.strip().lower() for c in concepts if len(c.strip()) > 2]))
        
        return concepts
    
    def _calculate_importance(self, text: str, metadata: Dict) -> float:
        """Calculate importance score based on content analysis"""
        importance = 1.0
        
        text_lower = text.lower()
        
        # Personal information
        personal_keywords = ['name', 'brother', 'sister', 'family', 'home', 'address', 'email', 'phone', 'birthday']
        if any(kw in text_lower for kw in personal_keywords):
            importance = max(importance, 5.0)
        
        # Preferences
        preference_keywords = ['like', 'dislike', 'prefer', 'favorite', 'love', 'hate', 'enjoy']
        if any(kw in text_lower for kw in preference_keywords):
            importance = max(importance, 4.0)
        
        # Facts about user
        fact_keywords = ['student', 'study', 'work', 'university', 'college', 'semester', 'major']
        if any(kw in text_lower for kw in fact_keywords):
            importance = max(importance, 3.0)
        
        # Source-based importance
        source = metadata.get('source', 'user')
        if source == 'user':
            importance = max(importance, 3.0)
        elif source == 'verified':
            importance = max(importance, 4.0)
        
        return importance
    
    def retrieve_similar(
        self, 
        query: str, 
        n_results: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """Retrieve similar memories with strength-based ranking and RIF"""
        n_results = n_results or MemoryConfig.DEFAULT_RETRIEVAL_COUNT
        threshold = threshold or MemoryConfig.RELEVANCE_THRESHOLD
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2
            )
            if results['documents'] is None or len(results['documents']) == 0:
                return []
            
            retrieved_memories = []
            for i in range(len(results['documents'][0])):
                memory_id = results['ids'][0][i] if results['ids'] else None
                if not memory_id: continue
                
                # Check RIF suppression
                if self.rif.is_suppressed(memory_id):
                    continue
                
                # Parse concepts string back to list - FIXED: Handle None metadata
                raw_metadata = results['metadatas'][0][i] if results['metadatas'] and len(results['metadatas'][0]) > i else {}
                if raw_metadata is None:
                    raw_metadata = {}
                
                concepts_str = raw_metadata.get('concepts', 'general')
                concepts_list = concepts_str.split(',') if concepts_str else ['general']
                raw_metadata['concepts'] = concepts_list
                
                memory = {
                    'id': memory_id,
                    'content': results['documents'][0][i],
                    'distance': results['distances'][0][i],
                    'metadata': raw_metadata
                }
                
                # Calculate current strength
                strength_data = self.memory_metadata.get(memory_id, {})
                time_elapsed = (datetime.now() - datetime.fromisoformat(strength_data.get('timestamp', datetime.now().isoformat()))).total_seconds() / 3600
                
                current_strength = self.strength_calculator.calculate(
                    initial_strength=strength_data.get('initial_strength', 1.0),
                    time_elapsed_hours=time_elapsed,
                    importance_score=strength_data.get('importance', 1.0),
                    retrieval_count=strength_data.get('retrieval_count', 0),
                    validation_success_count=strength_data.get('validation_success_count', 0),
                    last_retrieved=datetime.fromisoformat(strength_data.get('last_retrieved', datetime.now().isoformat())) if strength_data.get('last_retrieved') else None
                )
                
                memory['current_strength'] = current_strength
                
                # Filter by threshold
                if memory['distance'] < threshold:
                    retrieved_memories.append(memory)
            
            # Apply RIF to competitors
            if retrieved_memories:
                self.rif.suppress_competitors(retrieved_memories[0], retrieved_memories)
            
            # Update retrieval counts
            for memory in retrieved_memories[:n_results]:
                self._update_retrieval_stats(memory['id'])
            
            return retrieved_memories[:n_results]
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []
    
    def _update_retrieval_stats(self, memory_id: str):
        """Update retrieval statistics for a memory"""
        if memory_id in self.memory_metadata:
            self.memory_metadata[memory_id]['retrieval_count'] = self.memory_metadata[memory_id].get('retrieval_count', 0) + 1
            self.memory_metadata[memory_id]['last_retrieved'] = datetime.now().isoformat()
            self._save_memory_metadata()
    
    def update_validation_stats(self, memory_id: str, success: bool):
        """Update validation statistics for a memory"""
        if memory_id in self.memory_metadata:
            if success:
                self.memory_metadata[memory_id]['validation_success_count'] = self.memory_metadata[memory_id].get('validation_success_count', 0) + 1
            self._save_memory_metadata()
    
    def get_frequent_concepts(self, min_frequency: Optional[int] = None) -> List[Tuple[str, int]]:
        """Get frequently mentioned concepts"""
        min_freq = min_frequency or MemoryConfig.MIN_CONCEPT_FREQUENCY
        frequent = [(k, v['count']) for k, v in self.concepts.items() if v['count'] >= min_freq]
        return sorted(frequent, key=lambda x: x[1], reverse=True)
    
    def get_memories_by_strength(self, min_strength: float = 0.3) -> List[Dict]:
        """Get memories above a certain strength threshold"""
        all_ids = list(self.memory_metadata.keys())
        strong_memories = []
        
        for memory_id in all_ids:
            metadata = self.memory_metadata[memory_id]
            time_elapsed = (datetime.now() - datetime.fromisoformat(metadata.get('timestamp', datetime.now().isoformat()))).total_seconds() / 3600
            
            strength = self.strength_calculator.calculate(
                initial_strength=metadata.get('initial_strength', 1.0),
                time_elapsed_hours=time_elapsed,
                importance_score=metadata.get('importance', 1.0),
                retrieval_count=metadata.get('retrieval_count', 0),
                validation_success_count=metadata.get('validation_success_count', 0)
            )
            
            if strength >= min_strength:
                try:
                    result = self.collection.get(ids=[memory_id], include=['documents', 'metadatas'])
                    if result['documents']:
                        # FIXED: Handle None metadata properly
                        metadata_from_db = result['metadatas'][0] if result['metadatas'] and len(result['metadatas']) > 0 else {}
                        if metadata_from_db is None:
                            metadata_from_db = {}
                        
                        strong_memories.append({
                            'id': memory_id,
                            'content': result['documents'][0],
                            'metadata': metadata_from_db,
                            'strength': strength
                        })
                except:
                    continue
        
        return strong_memories
    
    def get_recent_memories(self, limit: int = 10) -> List[Dict]:
        """Get recent memories for review (Category 11)"""
        all_ids = list(self.memory_metadata.keys())[-limit:]
        recent_memories = []
        
        for memory_id in reversed(all_ids):
            try:
                result = self.collection.get(ids=[memory_id], include=['documents', 'metadatas'])
                if result['documents']:
                    metadata = self.memory_metadata.get(memory_id, {})
                    time_elapsed = (datetime.now() - datetime.fromisoformat(metadata.get('timestamp', datetime.now().isoformat()))).total_seconds() / 3600
                    
                    strength = self.strength_calculator.calculate(
                        initial_strength=metadata.get('initial_strength', 1.0),
                        time_elapsed_hours=time_elapsed,
                        importance_score=metadata.get('importance', 1.0),
                        retrieval_count=metadata.get('retrieval_count', 0),
                        validation_success_count=metadata.get('validation_success_count', 0)
                    )
                    
                    # FIXED: Handle None metadata properly
                    metadata_from_db = result['metadatas'][0] if result['metadatas'] and len(result['metadatas']) > 0 else {}
                    if metadata_from_db is None:
                        metadata_from_db = {}
                    
                    recent_memories.append({
                        'id': memory_id,
                        'content': result['documents'][0],
                        'metadata': metadata_from_db,
                        'strength': strength,
                        'timestamp': metadata.get('timestamp', 'unknown')
                    })
            except:
                continue
        
        return recent_memories
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory (Category 11)"""
        try:
            self.collection.delete(ids=[memory_id])
            if memory_id in self.memory_metadata:
                del self.memory_metadata[memory_id]
                self._save_memory_metadata()
            logger.info(f"Deleted memory: {memory_id[:8]}")
            return True
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return False
    
    def forget_weak_memories(self, threshold: float = 0.2) -> int:
        """Remove memories below strength threshold (Category 6)"""
        weak_memories = []
        
        for memory_id, metadata in self.memory_metadata.items():
            time_elapsed = (datetime.now() - datetime.fromisoformat(metadata.get('timestamp', datetime.now().isoformat()))).total_seconds() / 3600
            
            strength = self.strength_calculator.calculate(
                initial_strength=metadata.get('initial_strength', 1.0),
                time_elapsed_hours=time_elapsed,
                importance_score=metadata.get('importance', 1.0),
                retrieval_count=metadata.get('retrieval_count', 0),
                validation_success_count=metadata.get('validation_success_count', 0)
            )
            
            if strength < threshold:
                weak_memories.append(memory_id)
        
        if weak_memories:
            self.collection.delete(ids=weak_memories)
            
            for memory_id in weak_memories:
                del self.memory_metadata[memory_id]
            
            self._save_memory_metadata()
            logger.info(f"Forget operation: Removed {len(weak_memories)} weak memories")
        
        return len(weak_memories)
    
    def get_memory_statistics(self) -> Dict:
        """Get comprehensive memory statistics"""
        total_memories = len(self.memory_metadata)
        
        if self.memory_metadata:
            strengths = []
            for memory_id, metadata in self.memory_metadata.items():
                time_elapsed = (datetime.now() - datetime.fromisoformat(metadata.get('timestamp', datetime.now().isoformat()))).total_seconds() / 3600
                
                strength = self.strength_calculator.calculate(
                    initial_strength=metadata.get('initial_strength', 1.0),
                    time_elapsed_hours=time_elapsed,
                    importance_score=metadata.get('importance', 1.0),
                    retrieval_count=metadata.get('retrieval_count', 0),
                    validation_success_count=metadata.get('validation_success_count', 0)
                )
                strengths.append(strength)
            
            avg_strength = np.mean(strengths)
        else:
            avg_strength = 0
        
        return {
            "total_memories": total_memories,
            "tracked_concepts": len(self.concepts),
            "average_memory_strength": avg_strength,
            "suppressed_memories": self.rif.get_suppression_status()['suppressed_count'],
            "frequent_concepts": self.get_frequent_concepts()[:10]
        }
    













# """
# Advanced Semantic Memory System
# Features:
# - Ebbinghaus forgetting curve
# - Retrieval-Induced Forgetting (RIF)
# - Multi-tier memory (ephemeral, working, long-term)
# - Concept frequency tracking
# - Importance-based retention
# """
# from typing import Dict, List, Any, Optional, Tuple  # Added Optional

# import os
# import json
# import hashlib
# import logging
# from datetime import datetime, timedelta
# from typing import Dict, List, Any, Optional, Tuple
# import numpy as np
# import chromadb
# from chromadb.config import Settings
# from sentence_transformers import SentenceTransformer
# import re
# from collections import defaultdict
# import pickle

# from config import MemoryConfig, MEMORY_DIR

# logger = logging.getLogger(__name__)

# class MemoryStrengthCalculator:
#     """Calculate memory strength using Ebbinghaus-inspired decay"""
    
#     @staticmethod
#     def calculate(
#         initial_strength: float = 1.0,
#         time_elapsed_hours: float = 0,
#         importance_score: float = 1.0,
#         retrieval_count: int = 0,
#         validation_success_count: int = 0,
#         last_retrieved: Optional[datetime] = None
#     ) -> float:
#         """
#         Calculate current memory strength using biologically-inspired formula
        
#         Memory strength decays exponentially but is reinforced by:
#         - High importance scores
#         - Frequent retrieval (spacing effect)
#         - Successful validations
#         - Recent access
#         """
#         # Base exponential decay (Ebbinghaus curve)
#         half_life = MemoryConfig.MEMORY_HALF_LIFE_HOURS
#         base_decay = np.exp(-np.log(2) * time_elapsed_hours / half_life)
        
#         # Importance reinforcement
#         importance_factor = 1 + (MemoryConfig.IMPORTANCE_WEIGHT * importance_score)
        
#         # Retrieval reinforcement (spacing effect)
#         retrieval_factor = 1 + (MemoryConfig.RETRIEVAL_WEIGHT * min(retrieval_count, 20))
        
#         # Validation reinforcement
#         validation_factor = 1 + (MemoryConfig.VALIDATION_WEIGHT * validation_success_count)
        
#         # Recency bonus (memories accessed recently are stronger)
#         recency_factor = 1.0
#         if last_retrieved:
#             hours_since_retrieval = (datetime.now() - last_retrieved).total_seconds() / 3600
#             if hours_since_retrieval < 24:
#                 recency_factor = 1.2
#             elif hours_since_retrieval < 72:
#                 recency_factor = 1.1
        
#         # Calculate final strength
#         strength = initial_strength * base_decay * importance_factor * retrieval_factor * validation_factor * recency_factor
        
#         return min(1.0, max(0.0, strength))  # Clamp to [0, 1]


# class RetrievalInducedForgetting:
#     """Implements Retrieval-Induced Forgetting (RIF) mechanism"""
    
#     def __init__(self):
#         self.suppressed_memories = {}  # memory_id -> suppression_end_time
#         self.suppression_duration = timedelta(hours=2)
    
#     def suppress_competitors(self, retrieved_memory: Dict, all_memories: List[Dict]) -> List[str]:
#         """
#         When retrieving a memory, suppress competing memories that share concepts
#         but contain different/contradictory information
#         """
#         suppressed_ids = []
#         retrieved_concepts = set(retrieved_memory.get('metadata', {}).get('concepts', []))
        
#         for memory in all_memories:
#             if memory['id'] == retrieved_memory['id']:
#                 continue
            
#             memory_concepts = set(memory.get('metadata', {}).get('concepts', []))
            
#             # Calculate concept overlap
#             if len(retrieved_concepts) > 0 and len(memory_concepts) > 0:
#                 overlap = len(retrieved_concepts & memory_concepts) / min(len(retrieved_concepts), len(memory_concepts))
                
#                 # If high overlap but different content, suppress
#                 if overlap > MemoryConfig.RIF_CONCEPT_OVERLAP_THRESHOLD:
#                     suppressed_ids.append(memory['id'])
#                     self.suppressed_memories[memory['id']] = datetime.now() + self.suppression_duration
        
#         return suppressed_ids
    
#     def is_suppressed(self, memory_id: str) -> bool:
#         """Check if a memory is currently suppressed"""
#         if memory_id not in self.suppressed_memories:
#             return False
        
#         if datetime.now() > self.suppressed_memories[memory_id]:
#             # Suppression expired
#             del self.suppressed_memories[memory_id]
#             return False
        
#         return True
    
#     def get_suppression_status(self) -> Dict:
#         """Get current suppression status"""
#         return {
#             "suppressed_count": len(self.suppressed_memories),
#             "suppressed_ids": list(self.suppressed_memories.keys())
#         }


# class SemanticMemory:
#     """Advanced semantic memory system with all cognitive features"""
    
#     def __init__(self, persist_directory: Optional[str] = None):
#         self.persist_dir = persist_directory or str(MEMORY_DIR)
        
#         # Initialize ChromaDB client
#         self.chroma_client = chromadb.PersistentClient(
#             path=self.persist_dir,
#             settings=Settings(anonymized_telemetry=False)
#         )
        
#         # Initialize embedding model
#         logger.info(f"Loading embedding model: {MemoryConfig.EMBEDDING_MODEL}")
#         self.embedding_model = SentenceTransformer(MemoryConfig.EMBEDDING_MODEL)
        
#         # Create or get collection
#         try:
#             self.collection = self.chroma_client.get_collection(MemoryConfig.COLLECTION_NAME)
#         except:
#             self.collection = self.chroma_client.create_collection(
#                 name=MemoryConfig.COLLECTION_NAME,
#                 metadata={"hnsw:space": MemoryConfig.DISTANCE_METRIC}
#             )
        
#         # Initialize cognitive components
#         self.strength_calculator = MemoryStrengthCalculator()
#         self.rif = RetrievalInducedForgetting()
        
#         # Load metadata
#         self.metadata_file = os.path.join(self.persist_dir, "memory_metadata.json")
#         self.concepts = self._load_concepts()
#         self.memory_metadata = self._load_memory_metadata()
        
#         logger.info(f"Semantic memory initialized with {len(self.concepts)} tracked concepts")
    
#     def _load_concepts(self) -> Dict:
#         """Load concept tracking from file"""
#         metadata_path = os.path.join(self.persist_dir, "concepts.json")
#         if os.path.exists(metadata_path):
#             with open(metadata_path, 'r') as f:
#                 return json.load(f)
#         return {}
    
#     def _save_concepts(self):
#         """Save concept tracking to file"""
#         metadata_path = os.path.join(self.persist_dir, "concepts.json")
#         with open(metadata_path, 'w') as f:
#             json.dump(self.concepts, f, indent=2)
    
#     def _load_memory_metadata(self) -> Dict:
#         """Load per-memory metadata (strength, retrieval count, etc.)"""
#         if os.path.exists(self.metadata_file):
#             with open(self.metadata_file, 'r') as f:
#                 return json.load(f)
#         return {}
    
#     def _save_memory_metadata(self):
#         """Save per-memory metadata to file"""
#         with open(self.metadata_file, 'w') as f:
#             json.dump(self.memory_metadata, f, indent=2)
    
#     def add_memory(
#         self, 
#         text: str, 
#         metadata: Optional[Dict] = None,
#         tier: str = "working"
#     ) -> str:
#         """Add a new memory with ChromaDB-compatible metadata"""
#         if metadata is None:
#             metadata = {}
        
#         # Generate embeddings
#         embedding = self.embedding_model.encode([text])[0].tolist()
        
#         # Extract concepts automatically
#         concepts = self._extract_concepts(text)
        
#         # Calculate initial importance score
#         importance_score = self._calculate_importance(text, metadata)
        
#         # Update concept frequency
#         for concept in concepts:
#             if concept not in self.concepts:
#                 self.concepts[concept] = {
#                     "count": 0,
#                     "first_seen": datetime.now().isoformat(),
#                     "last_seen": datetime.now().isoformat(),
#                     "associated_memory_ids": []
#                 }
#             self.concepts[concept]["count"] += 1
#             self.concepts[concept]["last_seen"] = datetime.now().isoformat()
        
#         # Generate unique ID
#         memory_id = hashlib.md5((text + str(datetime.now())).encode()).hexdigest()
        
#         # FIX: Convert concepts list to comma-separated string (ChromaDB compatible)
#         concepts_str = ",".join(concepts) if concepts else "general"
        
#         # FIX: Ensure all metadata values are ChromaDB-compatible types (str, int, float, bool)
#         storage_metadata = {
#             "timestamp": datetime.now().isoformat(),
#             "concepts": concepts_str,  # String instead of list
#             "source": metadata.get("source", "user") or "user",  # No None values
#             "importance": float(importance_score),  # Ensure float
#             "tier": tier,
#             "initial_strength": 1.0,
#             "current_strength": 1.0,
#             "retrieval_count": 0,
#             "validation_success_count": 0,
#             "last_retrieved": datetime.now().isoformat(),  # String, not datetime object
#         }
        
#         # FIX: Add only simple string/int/float values from user metadata
#         for key, value in metadata.items():
#             if isinstance(value, (str, int, float, bool)):
#                 storage_metadata[key] = value
#             elif value is not None:
#                 # Convert other types to string
#                 storage_metadata[key] = str(value)
        
#         # Store metadata locally (can be complex)
#         self.memory_metadata[memory_id] = {
#             "timestamp": datetime.now().isoformat(),
#             "concepts": concepts,  # Keep as list for local use
#             "source": metadata.get("source", "user"),
#             "importance": importance_score,
#             "tier": tier,
#             "initial_strength": 1.0,
#             "current_strength": 1.0,
#             "retrieval_count": 0,
#             "validation_success_count": 0,
#             "last_retrieved": datetime.now().isoformat(),
#             **metadata
#         }
        
#         # Add to ChromaDB (with cleaned metadata)
#         self.collection.add(
#             documents=[text],
#             embeddings=[embedding],
#             ids=[memory_id],
#             metadatas=[storage_metadata]  # ChromaDB-compatible metadata
#         )
        
#         # Update concept associations
#         for concept in concepts:
#             if memory_id not in self.concepts[concept]["associated_memory_ids"]:
#                 self.concepts[concept]["associated_memory_ids"].append(memory_id)
        
#         # Save metadata
#         self._save_concepts()
#         self._save_memory_metadata()
        
#         logger.info(f"Added {tier} memory: {text[:50]}... (ID: {memory_id[:8]})")
#         return memory_id
    
#     def _extract_concepts(self, text: str) -> List[str]:
#         """Extract key concepts from text using multiple strategies"""
#         concepts = []
        
#         # Strategy 1: Capitalized words (proper nouns, acronyms)
#         capitalized = re.findall(r'\b[A-Z][A-Za-z]*\b', text)
#         concepts.extend(capitalized)
        
#         # Strategy 2: Key phrases (name, location, preference indicators)
#         key_phrases = [
#             r'(?:my name is|i am|i\'m)\s+([A-Za-z]+)',
#             r'(?:live|stay|from)\s+in\s+([A-Za-z\s,]+)',
#             r'(?:like|love|prefer|favorite)\s+(?:to\s+)?([A-Za-z\s]+)',
#             r'(?:brother|sister|parent|friend)\s+(?:named|is)\s+([A-Za-z]+)'
#         ]
#         for pattern in key_phrases:
#             matches = re.findall(pattern, text, re.IGNORECASE)
#             concepts.extend(matches)
        
#         # Strategy 3: Technical terms (numbers with units, technical vocabulary)
#         technical = re.findall(r'\b\d+\w*\b|\b\w+(?:ing|ed|tion|ity)\b', text)
#         concepts.extend(technical)
        
#         # Clean and deduplicate
#         concepts = list(set([c.strip().lower() for c in concepts if len(c.strip()) > 2]))
        
#         return concepts
    
#     def _calculate_importance(self, text: str, metadata: Dict) -> float:
#         """Calculate importance score based on content analysis"""
#         importance = 1.0  # Base importance
        
#         text_lower = text.lower()
        
#         # Personal information (highest importance)
#         personal_keywords = ['name', 'brother', 'sister', 'family', 'home', 'address', 'email', 'phone', 'birthday']
#         if any(kw in text_lower for kw in personal_keywords):
#             importance = max(importance, 5.0)
        
#         # Preferences
#         preference_keywords = ['like', 'dislike', 'prefer', 'favorite', 'love', 'hate', 'enjoy']
#         if any(kw in text_lower for kw in preference_keywords):
#             importance = max(importance, 4.0)
        
#         # Facts about user
#         fact_keywords = ['student', 'study', 'work', 'university', 'college', 'semester', 'major']
#         if any(kw in text_lower for kw in fact_keywords):
#             importance = max(importance, 3.0)
        
#         # Source-based importance
#         source = metadata.get('source', 'user')
#         if source == 'user':
#             importance = max(importance, 3.0)
#         elif source == 'verified':
#             importance = max(importance, 4.0)
        
#         return importance
    
#     def retrieve_similar(
#         self, 
#         query: str, 
#         n_results: Optional[int] = None,
#         threshold: Optional[float] = None
#     ) -> List[Dict]:
#         """Retrieve similar memories with strength-based ranking and RIF"""
#         n_results = n_results or MemoryConfig.DEFAULT_RETRIEVAL_COUNT
#         threshold = threshold or MemoryConfig.RELEVANCE_THRESHOLD
        
#         # Generate query embedding
#         query_embedding = self.embedding_model.encode([query])[0].tolist()
        
#         # Query ChromaDB
#         results = self.collection.query(
#             query_embeddings=[query_embedding],
#             n_results=n_results * 2
#         )
        
#         retrieved_memories = []
#         if results['documents'] is None or len(results['documents']) == 0:
#             return retrieved_memories
        
#         for i in range(len(results['documents'][0])):
#             memory_id = results['ids'][0][i]
            
#             # Check RIF suppression
#             if self.rif.is_suppressed(memory_id):
#                 continue
            
#             # FIX: Parse concepts string back to list for local use only
#             raw_metadata = results['metadatas'][0][i] if results['metadatas'] else {}
#             concepts_str = str(raw_metadata.get('concepts', 'general')) if raw_metadata.get('concepts') else 'general'
#             concepts_list = [c.strip() for c in concepts_str.split(',') if c.strip()]
            
#             # Create a new dictionary with the parsed concepts
#             processed_metadata = dict(raw_metadata)
#             # Keep concepts as string in metadata (ChromaDB requirement), but parse for local use
#             concepts_list_for_use = concepts_list
            
#             memory = {
#                 'id': memory_id,
#                 'content': results['documents'][0][i],
#                 'distance': results['distances'][0][i] if results['distances'] else 0.0,
#                 'metadata': processed_metadata
#             }
            
#             # Calculate current strength
#             strength_data = self.memory_metadata.get(memory_id, {})
#             time_elapsed = (datetime.now() - datetime.fromisoformat(strength_data.get('timestamp', datetime.now().isoformat()))).total_seconds() / 3600
            
#             current_strength = self.strength_calculator.calculate(
#                 initial_strength=strength_data.get('initial_strength', 1.0),
#                 time_elapsed_hours=time_elapsed,
#                 importance_score=strength_data.get('importance', 1.0),
#                 retrieval_count=strength_data.get('retrieval_count', 0),
#                 validation_success_count=strength_data.get('validation_success_count', 0),
#                 last_retrieved=datetime.fromisoformat(strength_data.get('last_retrieved', datetime.now().isoformat())) if strength_data.get('last_retrieved') else None
#             )
            
#             memory['current_strength'] = current_strength
            
#             # Filter by threshold
#             if memory['distance'] < threshold:
#                 retrieved_memories.append(memory)
            
#             # Apply RIF to competitors
#             if retrieved_memories:
#                 self.rif.suppress_competitors(retrieved_memories[0], retrieved_memories)
            
#             if len(retrieved_memories) >= n_results:
#                 break
        
#         # Update retrieval counts
#         for memory in retrieved_memories[:n_results]:
#             self._update_retrieval_stats(memory['id'])
        
#         return retrieved_memories[:n_results]
    
#     def _update_retrieval_stats(self, memory_id: str):
#         if memory_id in self.memory_metadata:
#             self.memory_metadata[memory_id]['retrieval_count'] = \
#                 self.memory_metadata[memory_id].get('retrieval_count', 0) + 1
#             self.memory_metadata[memory_id]['last_retrieved'] = datetime.now().isoformat()
#             self._save_memory_metadata()
    
#     def update_validation_stats(self, memory_id: str, success: bool):
#         """Update validation statistics for a memory"""
#         if memory_id in self.memory_metadata:
#             if success:
#                 self.memory_metadata[memory_id]['validation_success_count'] = self.memory_metadata[memory_id].get('validation_success_count', 0) + 1
#             self._save_memory_metadata()
    
#     def get_frequent_concepts(self, min_frequency: Optional[int] = None) -> List[Tuple[str, int]]:
#         """Get frequently mentioned concepts"""
#         min_freq = min_frequency or MemoryConfig.MIN_CONCEPT_FREQUENCY
#         frequent = [(k, v['count']) for k, v in self.concepts.items() if v['count'] >= min_freq]
#         return sorted(frequent, key=lambda x: x[1], reverse=True)
    
#     def get_memories_by_strength(self, min_strength: float = 0.3) -> List[Dict]:
#         """Get memories above a certain strength threshold"""
#         all_ids = list(self.memory_metadata.keys())
#         strong_memories = []
        
#         for memory_id in all_ids:
#             metadata = self.memory_metadata[memory_id]
#             time_elapsed = (datetime.now() - datetime.fromisoformat(metadata.get('timestamp', datetime.now().isoformat()))).total_seconds() / 3600
            
#             strength = self.strength_calculator.calculate(
#                 initial_strength=metadata.get('initial_strength', 1.0),
#                 time_elapsed_hours=time_elapsed,
#                 importance_score=metadata.get('importance', 1.0),
#                 retrieval_count=metadata.get('retrieval_count', 0),
#                 validation_success_count=metadata.get('validation_success_count', 0)
#             )
            
#             if strength >= min_strength:
#                 # Retrieve full memory content
#                 try:
#                     result = self.collection.get(ids=[memory_id], include=['documents', 'metadatas'])
#                     if result['documents'] and result['metadatas']:
#                         strong_memories.append({
#                             'id': memory_id,
#                             'content': result['documents'][0],
#                             'metadata': result['metadatas'][0],
#                             'strength': strength
#                         })
#                 except:
#                     continue
        
#         return strong_memories
    
#     def forget_weak_memories(self, threshold: float = 0.2) -> int:
#         """Remove memories below strength threshold"""
#         weak_memories = []
        
#         for memory_id, metadata in self.memory_metadata.items():
#             time_elapsed = (datetime.now() - datetime.fromisoformat(metadata.get('timestamp', datetime.now().isoformat()))).total_seconds() / 3600
            
#             strength = self.strength_calculator.calculate(
#                 initial_strength=metadata.get('initial_strength', 1.0),
#                 time_elapsed_hours=time_elapsed,
#                 importance_score=metadata.get('importance', 1.0),
#                 retrieval_count=metadata.get('retrieval_count', 0),
#                 validation_success_count=metadata.get('validation_success_count', 0)
#             )
            
#             if strength < threshold:
#                 weak_memories.append(memory_id)
        
#         # Delete from ChromaDB
#         if weak_memories:
#             self.collection.delete(ids=weak_memories)
            
#             # Remove from metadata
#             for memory_id in weak_memories:
#                 del self.memory_metadata[memory_id]
            
#             self._save_memory_metadata()
#             logger.info(f"Forget operation: Removed {len(weak_memories)} weak memories")
        
#         return len(weak_memories)
    
#     def get_memory_statistics(self) -> Dict:
#         """Get comprehensive memory statistics"""
#         total_memories = len(self.memory_metadata)
#         avg_strength = np.mean([
#             self.strength_calculator.calculate(
#                 initial_strength=m.get('initial_strength', 1.0),
#                 time_elapsed_hours=(datetime.now() - datetime.fromisoformat(m.get('timestamp', datetime.now().isoformat()))).total_seconds() / 3600,
#                 importance_score=m.get('importance', 1.0),
#                 retrieval_count=m.get('retrieval_count', 0),
#                 validation_success_count=m.get('validation_success_count', 0)
#             )
#             for m in self.memory_metadata.values()
#         ]) if self.memory_metadata else 0
        
#         return {
#             "total_memories": total_memories,
#             "tracked_concepts": len(self.concepts),
#             "average_memory_strength": avg_strength,
#             "suppressed_memories": self.rif.get_suppression_status()['suppressed_count'],
#             "frequent_concepts": self.get_frequent_concepts()[:10]
#         }