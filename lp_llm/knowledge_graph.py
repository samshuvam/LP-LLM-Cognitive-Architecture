"""
LP-LLM Cognitive Architecture Component
Authored by Shuvam (https://github.com/samshuvam)
"""

__author__ = "Shuvam (https://github.com/samshuvam)"

"""
Knowledge Graph & Semantic Drift Detection
Production-Ready with Real-Time Updates

Features:
- Concept relationship mapping
- Contradiction detection
- Knowledge coherence scoring
- Graph-based reasoning
- Real-time updates during conversations
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict
from .config import LEARNING_DIR

logger = logging.getLogger(__name__)

nx = None

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("networkx not available. Install with: pip install networkx")


class KnowledgeGraph:
    """Graph-based knowledge representation with drift detection (Category 7)"""
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else LEARNING_DIR
        self.graph_file = self.base_path / "knowledge_graph.json"
        
        # Initialize graph
        if NETWORKX_AVAILABLE and nx is not None:
            self.graph = nx.DiGraph()
        else:
            self.graph = None
        
        # Load existing graph
        self._load_graph()
        
        logger.debug(f"Knowledge graph initialized with {self.graph.number_of_nodes() if self.graph else 0} nodes")
    
    def _load_graph(self):
        """Load graph from file"""
        if self.graph_file.exists() and self.graph is not None:
            try:
                with open(self.graph_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct graph
                for node, attrs in data.get('nodes', {}).items():
                    self.graph.add_node(node, **attrs)
                for edge in data.get('edges', []):
                    self.graph.add_edge(edge['source'], edge['target'], **edge.get('attrs', {}))
                
                logger.debug(f"Loaded knowledge graph with {self.graph.number_of_nodes()} nodes")
            except Exception as e:
                logger.error(f"Error loading knowledge graph: {e}")
    
    def _save_graph(self):
        """Save graph to file"""
        if self.graph is None:
            return
        
        try:
            data = {
                "nodes": {node: dict(self.graph.nodes[node]) for node in self.graph.nodes()},
                "edges": [
                    {"source": u, "target": v, "attrs": dict(self.graph.edges[u, v])}
                    for u, v in self.graph.edges # type: ignore
                        ],
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.graph_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")
    
    def add_fact(self, subject: str, predicate: str, object_: str, confidence: float = 1.0):
        """Add a fact to the knowledge graph (Category 7)"""
        if self.graph is None:
            return
        
        try:
            # Add nodes
            self.graph.add_node(subject, type="entity", confidence=confidence, updated_at=datetime.now().isoformat())
            self.graph.add_node(object_, type="entity", confidence=confidence, updated_at=datetime.now().isoformat())
            
            # Add or update edge
            if self.graph.has_edge(subject, object_):
                self.graph.edges[subject, object_]['confidence'] = confidence
                self.graph.edges[subject, object_]['updated_at'] = datetime.now().isoformat()
            else:
                self.graph.add_edge(subject, object_, relation=predicate, confidence=confidence, created_at=datetime.now().isoformat())
            
            self._save_graph()
            logger.debug(f"Added fact: {subject} --{predicate}--> {object_}")
        except Exception as e:
            logger.error(f"Error adding fact: {e}")
    
    def detect_drift(self, subject: str, predicate: str, new_object: str) -> Dict:
        """Detect semantic drift/contradictions (Category 7)"""
        result = {
            "conflict_detected": False,
            "existing_facts": [],
            "confidence_comparison": {},
            "recommendation": "accept"
        }
        
        if self.graph is None:
            return result
        
        try:
            if self.graph.has_node(subject):
                for neighbor in self.graph.successors(subject):
                    edge_data = self.graph.edges[subject, neighbor]
                    if edge_data.get('relation') == predicate:
                        existing_fact = {
                            "object": neighbor,
                            "confidence": edge_data.get('confidence', 0),
                            "created_at": edge_data.get('created_at')
                        }
                        result["existing_facts"].append(existing_fact)
            
            if result["existing_facts"]:
                result["conflict_detected"] = True
                
                for existing in result["existing_facts"]:
                    if existing["object"] != new_object:
                        result["confidence_comparison"][existing["object"]] = {
                            "existing_confidence": existing["confidence"],
                            "new_confidence": 1.0
                        }
                
                max_existing_conf = max(f["confidence"] for f in result["existing_facts"])
                if max_existing_conf > 0.9:
                    result["recommendation"] = "review"
                else:
                    result["recommendation"] = "update"
        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
        
        return result
    
    def get_related_concepts(self, concept: str, depth: int = 2) -> List[Dict]:
        """Get concepts related to a given concept"""
        related = []
        
        if self.graph is None or not self.graph.has_node(concept):
            return related
        
        try:
            visited = set()
            queue = [(concept, 0)]
            
            while queue:
                current, current_depth = queue.pop(0)
                
                if current_depth > depth or current in visited:
                    continue
                
                visited.add(current)
                
                for neighbor in self.graph.neighbors(current):
                    edge_data = self.graph.edges[current, neighbor]
                    related.append({
                        "concept": neighbor,
                        "relation": edge_data.get('relation'),
                        "confidence": edge_data.get('confidence', 0),
                        "depth": current_depth + 1
                    })
                    queue.append((neighbor, current_depth + 1))
        except Exception as e:
            logger.error(f"Error getting related concepts: {e}")
        
        return related
    
    def calculate_coherence_score(self) -> float:
        """Calculate overall knowledge graph coherence (Category 7)"""
        if self.graph is None or self.graph.number_of_edges() == 0:
            return 1.0
        
        try:
            conflicts = 0
            total_edges = self.graph.number_of_edges()
            
            for node in self.graph.nodes():
                relations = defaultdict(list)
                for neighbor in self.graph.successors(node):
                    relation = self.graph.edges[node, neighbor].get('relation')
                    relations[relation].append(neighbor)
                
                for relation, objects in relations.items():
                    if len(set(objects)) > 1:
                        conflicts += len(set(objects)) - 1
            
            coherence = 1.0 - (conflicts / total_edges) if total_edges > 0 else 1.0
            return max(0.0, min(1.0, coherence))
        except Exception as e:
            logger.error(f"Error calculating coherence: {e}")
            return 0.5
    
    def get_graph_statistics(self) -> Dict:
        """Get graph statistics (Category 7)"""
        if self.graph is None:
            return {
                "nodes": 0,
                "edges": 0,
                "coherence_score": 0.0,
                "density": 0.0
            }
        
        try:
            return {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "coherence_score": self.calculate_coherence_score(),
                "density": nx.density(self.graph) if NETWORKX_AVAILABLE and nx is not None and self.graph.number_of_nodes() > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {
                "nodes": 0,
                "edges": 0,
                "coherence_score": 0.0,
                "density": 0.0
            }
    
    def check_contradiction(self, new_fact: Dict) -> Dict:
        """Check if new fact contradicts existing knowledge (Category 7)"""
        subject = new_fact.get('subject')
        predicate = new_fact.get('predicate')
        object_ = new_fact.get('object')
        
        if not isinstance(subject, str) or not isinstance(predicate, str) or not isinstance(object_, str):
            return {"contradiction": False, "message": "Incomplete or invalid fact"}
        
        drift_result = self.detect_drift(str(subject), str(predicate), str(object_))
        if drift_result["conflict_detected"]:
            return {
                "contradiction": True,
                "existing_facts": drift_result["existing_facts"],
                "recommendation": drift_result["recommendation"],
                "message": f"Contradiction detected: {subject} {predicate} {object_} conflicts with existing knowledge"
            }
        
        return {"contradiction": False, "message": "No contradiction detected"}



# """
# Knowledge Graph & Semantic Drift Detection
# Features:
# - Concept relationship mapping
# - Contradiction detection
# - Knowledge coherence scoring
# - Graph-based reasoning
# """
# from typing import Dict, List, Any, Optional  # Added Optional

# import os
# import json
# import logging
# from datetime import datetime
# from typing import Dict, List, Any, Optional, Set, Tuple
# from pathlib import Path
# from collections import defaultdict
# import networkx as nx

# from config import LEARNING_DIR

# logger = logging.getLogger(__name__)


# class KnowledgeGraph:
#     """Graph-based knowledge representation with drift detection"""
    
#     def __init__(self, base_path: Optional[str] = None):
#         self.base_path = Path(base_path) if base_path else LEARNING_DIR
#         self.graph_file = self.base_path / "knowledge_graph.json"
        
#         # Initialize graph
#         self.graph = nx.DiGraph()
        
#         # Load existing graph
#         self._load_graph()
        
#         logger.info(f"Knowledge graph initialized with {self.graph.number_of_nodes()} nodes")
    
#     def _load_graph(self):
#         """Load graph from file"""
#         if self.graph_file.exists():
#             with open(self.graph_file, 'r') as f:
#                 data = json.load(f)
#                 # Reconstruct graph
#                 for node, attrs in data.get('nodes', {}).items():
#                     self.graph.add_node(node, **attrs)
#                 for edge in data.get('edges', []):
#                     self.graph.add_edge(edge['source'], edge['target'], **edge.get('attrs', {}))
    
#     def _save_graph(self):
#         """Save graph to file"""
#         data = {
#             "nodes": {node: dict(self.graph.nodes[node]) for node in self.graph.nodes()},
#             "edges": [
#                 {"source": u, "target": v, "attrs": dict(self.graph.edges[u, v])}
#                 for u, v in self.graph.edges()
#             ],
#             "last_updated": datetime.now().isoformat()
#         }
#         with open(self.graph_file, 'w') as f:
#             json.dump(data, f, indent=2)
    
#     def add_fact(self, subject: str, predicate: str, object_: str, confidence: float = 1.0):
#         """
#         Add a fact to the knowledge graph
        
#         Args:
#             subject: The subject entity
#             predicate: The relationship
#             object_: The object entity
#             confidence: Confidence score for this fact
#         """
#         # Add nodes
#         self.graph.add_node(subject, type="entity", confidence=confidence, updated_at=datetime.now().isoformat())
#         self.graph.add_node(object_, type="entity", confidence=confidence, updated_at=datetime.now().isoformat())
        
#         # Add edge
#         edge_key = (subject, predicate, object_)
#         if self.graph.has_edge(subject, object_):
#             # Update existing edge
#             self.graph.edges[subject, object_]['confidence'] = confidence
#             self.graph.edges[subject, object_]['updated_at'] = datetime.now().isoformat()
#         else:
#             self.graph.add_edge(subject, object_, relation=predicate, confidence=confidence, created_at=datetime.now().isoformat())
        
#         self._save_graph()
#         logger.debug(f"Added fact: {subject} --{predicate}--> {object_}")
    
#     def detect_drift(self, subject: str, predicate: str, new_object: str) -> Dict:
#         """
#         Detect semantic drift/contradictions
        
#         Args:
#             subject: The subject entity
#             predicate: The relationship
#             new_object: The new object being asserted
        
#         Returns:
#             Dictionary with drift detection results
#         """
#         result = {
#             "conflict_detected": False,
#             "existing_facts": [],
#             "confidence_comparison": {},
#             "recommendation": "accept"
#         }
        
#         # Find existing facts with same subject and predicate
#         if self.graph.has_node(subject):
#             for neighbor in self.graph.successors(subject):
#                 edge_data = self.graph.edges[subject, neighbor]
#                 if edge_data.get('relation') == predicate:
#                     existing_fact = {
#                         "object": neighbor,
#                         "confidence": edge_data.get('confidence', 0),
#                         "created_at": edge_data.get('created_at')
#                     }
#                     result["existing_facts"].append(existing_fact)
        
#         # Check for contradictions
#         if result["existing_facts"]:
#             result["conflict_detected"] = True
            
#             # Compare confidences
#             for existing in result["existing_facts"]:
#                 if existing["object"] != new_object:
#                     result["confidence_comparison"][existing["object"]] = {
#                         "existing_confidence": existing["confidence"],
#                         "new_confidence": 1.0  # Assume new fact has high confidence
#                     }
            
#             # Make recommendation
#             max_existing_conf = max(f["confidence"] for f in result["existing_facts"])
#             if max_existing_conf > 0.9:
#                 result["recommendation"] = "review"  # High confidence existing fact
#             else:
#                 result["recommendation"] = "update"  # Low confidence, can update
        
#         return result
    
#     def get_related_concepts(self, concept: str, depth: int = 2) -> List[Dict]:
#         """Get concepts related to a given concept"""
#         related = []
        
#         if not self.graph.has_node(concept):
#             return related
        
#         # BFS to find related concepts
#         visited = set()
#         queue = [(concept, 0)]
        
#         while queue:
#             current, current_depth = queue.pop(0)
            
#             if current_depth > depth or current in visited:
#                 continue
            
#             visited.add(current)
            
#             # Get neighbors
#             for neighbor in self.graph.neighbors(current):
#                 edge_data = self.graph.edges[current, neighbor]
#                 related.append({
#                     "concept": neighbor,
#                     "relation": edge_data.get('relation'),
#                     "confidence": edge_data.get('confidence', 0),
#                     "depth": current_depth + 1
#                 })
#                 queue.append((neighbor, current_depth + 1))
            
#             # Also check predecessors
#             for predecessor in self.graph.predecessors(current):
#                 edge_data = self.graph.edges[predecessor, current]
#                 related.append({
#                     "concept": predecessor,
#                     "relation": f"inverse_{edge_data.get('relation')}",
#                     "confidence": edge_data.get('confidence', 0),
#                     "depth": current_depth + 1
#                 })
#                 queue.append((predecessor, current_depth + 1))
        
#         return related
    
#     def calculate_coherence_score(self) -> float:
#         """Calculate overall knowledge graph coherence"""
#         if self.graph.number_of_edges() == 0:
#             return 1.0  # No edges = no conflicts
        
#         # Count conflicting edges
#         conflicts = 0
#         total_edges = self.graph.number_of_edges()
        
#         # Simple heuristic: check for nodes with multiple outgoing edges of same relation type
#         for node in self.graph.nodes():
#             relations = defaultdict(list)
#             for neighbor in self.graph.successors(node):
#                 relation = self.graph.edges[node, neighbor].get('relation')
#                 relations[relation].append(neighbor)
            
#             # If same relation points to different objects, potential conflict
#             for relation, objects in relations.items():
#                 if len(set(objects)) > 1:
#                     conflicts += len(set(objects)) - 1
        
#         coherence = 1.0 - (conflicts / total_edges) if total_edges > 0 else 1.0
#         return max(0.0, min(1.0, coherence))
    
#     def get_graph_statistics(self) -> Dict:
#         """Get graph statistics"""
#         return {
#             "nodes": self.graph.number_of_nodes(),
#             "edges": self.graph.number_of_edges(),
#             "coherence_score": self.calculate_coherence_score(),
#             "density": nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0
#         }