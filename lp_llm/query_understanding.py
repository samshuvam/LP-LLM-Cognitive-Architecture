"""
LP-LLM Cognitive Architecture Component
Authored by Shuvam (https://github.com/samshuvam)
"""

__author__ = "Shuvam (https://github.com/samshuvam)"

"""
Query Understanding Module - Enhanced Version
Uses semantic analysis for accurate intent detection
"""

import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
from .config import QueryUnderstandingConfig

logger = logging.getLogger(__name__)

class QueryUnderstanding:
    """Enhanced query understanding with improved accuracy"""
    
    def __init__(self):
        self.intent_categories = QueryUnderstandingConfig.INTENT_CATEGORIES
        self.realtime_types = QueryUnderstandingConfig.REALTIME_TYPES
        self.min_confidence = QueryUnderstandingConfig.MIN_CONFIDENCE_FOR_ACTION
        self.high_confidence = QueryUnderstandingConfig.HIGH_CONFIDENCE_THRESHOLD
        self.max_context_turns = QueryUnderstandingConfig.MAX_CONTEXT_TURNS
    
    def analyze_query(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict]] = None,
        user_profile: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Main analysis method with improved accuracy"""
        analysis = {
            "original_query": query,
            "timestamp": datetime.now().isoformat(),
            "intent": self._classify_intent(query, conversation_history),
            "entities": self._extract_entities(query, user_profile),
            "needs_realtime": False,
            "realtime_type": "none",
            "is_follow_up": self._is_follow_up(query, conversation_history),
            "is_correction": self._is_correction(query, conversation_history),
            "is_question": self._is_question(query),
            "confidence": 0.0,
            "suggested_action": "",
            "context_references": []
        }
        
        # Determine realtime need
        analysis["needs_realtime"], analysis["realtime_type"] = self._check_realtime_need(
            query, analysis["intent"], analysis["entities"]
        )
        
        # Determine action
        analysis["suggested_action"] = self._determine_action(analysis)
        
        # Calculate confidence (improved)
        analysis["confidence"] = self._calculate_confidence(analysis, query)
        
        # Find context references
        if conversation_history:
            analysis["context_references"] = self._find_context_references(
                query, conversation_history
            )
        
        # Log at DEBUG level only (won't show in console)
        logger.debug(f"Query analysis: {analysis['intent']} (confidence: {analysis['confidence']:.2f})")
        
        return analysis
    
    def _classify_intent(self, query: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """Improved intent classification with better accuracy"""
        query_lower = query.lower().strip()
        
        # Clean query
        clean_query = query_lower
        fillers = ["please", "can you", "could you", "i want", "i need", 
                   "tell me", "let me know", "ok", "okay", "so", "well", "hey", "hi"]
        for filler in fillers:
            clean_query = clean_query.replace(filler, "")
        clean_query = clean_query.strip()
        
        # Scoring system for better accuracy
        intent_scores = {intent: 0 for intent in self.intent_categories}
        
        # === GREETING ===
        greeting_patterns = [
            r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening)",
            r"^(how are you|how's it going|what's up|how do you do)"
        ]
        for pattern in greeting_patterns:
            if re.search(pattern, clean_query):
                intent_scores["greeting"] += 10
        
        # === MEMORY STORAGE ===
        memory_patterns = [
            r"(remember|save|store|keep in mind|don't forget|note this|note that)",
            r"(will you remember|can you remember|please remember|remember this)",
            r"(i want you to remember|i need you to remember)"
        ]
        for pattern in memory_patterns:
            if re.search(pattern, clean_query):
                intent_scores["memory_storage"] += 10
        
        # === CORRECTION ===
        correction_patterns = [
            r"^(no|wrong|incorrect|that's not right|actually|nah|nope)",
            r"(it's not|it is not|that's wrong|you're wrong|incorrect|wrong information)",
            r"(the correct|actually it's|actually it is|the real|the actual)"
        ]
        for pattern in correction_patterns:
            if re.search(pattern, clean_query):
                intent_scores["correction"] += 10
        
        # === CONFIRMATION ===
        confirmation_patterns = [
            r"^(yes|yeah|yep|correct|right|exactly|that's right|that's correct)",
            r"^(no|nah|nope|wrong|incorrect|that's wrong|not really)"
        ]
        for pattern in confirmation_patterns:
            if re.search(pattern, clean_query):
                intent_scores["confirmation"] += 10
        
        # === PERSONAL INFO ===
        personal_patterns = [
            r"(my name|what's my name|whats my name|do you know my name)",
            r"(where do i live|where i live|my location|where am i from)",
            r"(my brother|my sister|my family|my friend|my mother|my father)",
            r"(who am i|what about me|tell me about me)"
        ]
        for pattern in personal_patterns:
            if re.search(pattern, clean_query):
                intent_scores["personal_info"] += 10
        
        # === MATH ===
        math_patterns = [
            r"\d+\s*[\+\-\*\/]\s*\d+",
            r"(what is|what's|whats|calculate|solve)\s*\d+",
            r"(plus|minus|times|divided by|multiplied by)"
        ]
        for pattern in math_patterns:
            if re.search(pattern, clean_query):
                intent_scores["math_calculation"] += 10
        
        # === INFORMATION REQUEST ===
        info_patterns = [
            r"^(what|who|when|where|why|how|which|whom|whose)",
            r"^(tell me about|explain|describe|what can you|what are you)",
            r"^(is|are|was|were|does|do|did|can|could|will|would)\s+\w+\s+\?",
            r"(i want to know|i need to know|do you know|can you tell me)"
        ]
        for pattern in info_patterns:
            if re.search(pattern, clean_query):
                intent_scores["information_request"] += 5
        
        # === COMMAND ===
        command_patterns = [
            r"^(show|display|open|run|execute|start|stop|create|delete|update)",
            r"(i want you to|please|can you|could you)\s+(show|tell|find|get|search)"
        ]
        for pattern in command_patterns:
            if re.search(pattern, clean_query):
                intent_scores["command"] += 5
        
        # === FOLLOW-UP ===
        if conversation_history and len(conversation_history) > 0:
            follow_up_indicators = [
                "what about", "how about", "and", "also", "too", "else",
                "what else", "tell me more", "continue", "go on",
                "there", "that", "those", "it", "they", "them", "this"
            ]
            if any(indicator in clean_query for indicator in follow_up_indicators):
                if len(query.split()) < 10:
                    intent_scores["follow_up"] += 8
        
        # Select highest scoring intent
        best_intent = max(intent_scores, key=lambda k: intent_scores[k])
        
        # If all scores are 0, default based on question structure
        if intent_scores[best_intent] == 0:
            if clean_query.endswith('?') or clean_query.startswith(('what', 'who', 'when', 'where', 'why', 'how')):
                best_intent = "information_request"
            else:
                best_intent = "casual_chat"
        
        logger.debug(f"Intent classification: {best_intent} (score: {intent_scores[best_intent]})")
        return best_intent
    
    def _extract_entities(self, query: str, user_profile: Optional[Dict] = None) -> Dict[str, Any]:
        """Extract entities from query"""
        entities = {
            "person_names": [],
            "locations": [],
            "organizations": [],
            "dates": [],
            "numbers": [],
            "topics": []
        }
        
        query_lower = query.lower()
        
        # Person names
        name_patterns = [
            r"(?:name is|named|called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"(?:pm|president|minister|cm)\s+(?:of\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        ]
        for pattern in name_patterns:
            matches = re.findall(pattern, query)
            entities["person_names"].extend(matches)
        
        # Locations
        location_patterns = [
            r"(?:in|at|from|of|for)\s+([A-Z][A-Za-z\s,]+?)(?:\?|\.|,|!|$)",
            r"(?:live|stay|reside)\s+(?:in\s+)?([A-Za-z\s,]+?)(?:\.|,|!|\?|$)"
        ]
        for pattern in location_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                location = match.strip()
                if location.lower() not in ['the', 'what', 'when', 'where', 'how', 'today']:
                    entities["locations"].append(location)
        
        if user_profile and user_profile.get('location'):
            entities["locations"].append(user_profile['location'])
        
        # Topics
        topic_keywords = [
            "weather", "time", "date", "news", "cricket", "match", "game",
            "politics", "election", "government", "minister", "president",
            "economy", "stock", "market", "sports", "movie", "music"
        ]
        for keyword in topic_keywords:
            if keyword in query_lower:
                entities["topics"].append(keyword)
        
        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _is_follow_up(self, query: str, conversation_history: Optional[List[Dict]] = None) -> bool:
        """Improved follow-up detection"""
        if not conversation_history or len(conversation_history) == 0:
            return False
        
        query_lower = query.lower()
        
        # Follow-up indicators
        follow_up_words = [
            "there", "that", "those", "it", "they", "them", "this",
            "what about", "how about", "and", "also", "too", "else",
            "what else", "tell me more", "continue", "go on"
        ]
        
        pronouns = ["he", "she", "it", "they", "him", "her", "them"]
        
        has_indicator = any(word in query_lower for word in follow_up_words)
        has_pronoun = any(pronoun in query_lower.split() for pronoun in pronouns)
        
        # Check topic continuity
        last_turn = conversation_history[-1] if conversation_history else None
        if last_turn:
            last_query_type = last_turn.get('query_type', 'general')
            current_topics = self._extract_entities(query, {}).get('topics', [])
            
            # If same topic, likely follow-up
            if last_query_type in current_topics:
                return True
        
        # Short queries with indicators are follow-ups
        if (has_indicator or has_pronoun) and len(query.split()) < 12:
            return True
        
        return False
    
    def _is_correction(self, query: str, conversation_history: Optional[List[Dict]] = None) -> bool:
        """Improved correction detection"""
        query_lower = query.lower()
        
        correction_words = [
            "no", "wrong", "incorrect", "that's not right", "actually",
            "nah", "nope", "it's not", "it is not", "that's wrong",
            "you're wrong", "the correct", "the actual", "the real"
        ]
        
        if any(word in query_lower for word in correction_words):
            return True
        
        # Check context for contradictions
        if conversation_history:
            last_turn = conversation_history[-1] if conversation_history else None
            if last_turn:
                last_response = last_turn.get('response', '').lower()
                query_words = set(query_lower.split())
                response_words = set(last_response.split())
                overlap = query_words & response_words
                
                # Negation with overlap = correction
                negation_words = ["not", "no", "wrong", "incorrect"]
                if any(neg in query_lower for neg in negation_words) and len(overlap) > 3:
                    return True
        
        return False
    
    def _is_question(self, query: str) -> bool:
        """Check if query is a question"""
        if query.strip().endswith('?'):
            return True
        
        question_words = ["what", "who", "when", "where", "why", "how", "which"]
        query_lower = query.lower()
        if any(query_lower.startswith(word + " ") for word in question_words):
            return True
        
        inversion_patterns = [
            r"^(is|are|was|were|does|do|did|can|could|will|would)\s+"
        ]
        for pattern in inversion_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def _check_realtime_need(self, query: str, intent: str, entities: Dict) -> tuple:
        """Determine if real-time data is needed"""
        query_lower = query.lower()
        
        if intent not in ["information_request", "follow_up"]:
            return False, "none"
        
        # Weather
        if any(word in query_lower for word in ["weather", "temperature", "degrees", "celsius", "forecast"]):
            return True, "weather"
        
        # Time
        if any(word in query_lower for word in ["what time", "current time", "time is it", "o'clock"]):
            return True, "time"
        
        # Date
        if any(word in query_lower for word in ["what date", "today's date", "what day", "what year", "what month"]):
            return True, "date"
        
        # PM
        if any(word in query_lower for word in ["prime minister", "pm of", "chief minister", "current pm"]):
            return True, "pm"
        
        # President
        if any(word in query_lower for word in ["president", "head of state"]):
            return True, "president"
        
        # News
        if any(word in query_lower for word in ["latest news", "what's happening", "recent news"]):
            return True, "news"
        
        return False, "none"
    
    def _determine_action(self, analysis: Dict) -> str:
        """Determine suggested action"""
        intent = analysis["intent"]
        
        action_map = {
            "greeting": "respond_greeting",
            "information_request": "generate_response",
            "memory_storage": "store_in_memory",
            "correction": "acknowledge_and_update",
            "follow_up": "continue_context",
            "personal_info": "check_personal_memory",
            "command": "execute_command",
            "confirmation": "acknowledge_confirmation",
            "casual_chat": "generate_response",
            "math_calculation": "calculate_and_respond"
        }
        
        if analysis["needs_realtime"]:
            return f"fetch_realtime_{analysis['realtime_type']}"
        
        return action_map.get(intent, "generate_response")
    
    def _calculate_confidence(self, analysis: Dict, query: str) -> float:
        """Improved confidence calculation"""
        confidence = 0.5
        
        # Boost for clear question structure
        if analysis["is_question"]:
            confidence += 0.15
        
        # Boost for detected entities
        entities = analysis["entities"]
        total_entities = sum(len(v) for v in entities.values())
        if total_entities > 0:
            confidence += min(0.25, total_entities * 0.05)
        
        # Boost for clear intent
        if analysis["intent"] not in ["casual_chat", "unknown"]:
            confidence += 0.15
        
        # Boost for follow-up detection with context
        if analysis["is_follow_up"] and analysis["context_references"]:
            confidence += 0.1
        
        # Reduce for very short queries
        if len(query.split()) < 3:
            confidence -= 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def _find_context_references(self, query: str, conversation_history: List[Dict]) -> List[Dict]:
        """Find references to previous conversation"""
        references = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for i, turn in enumerate(conversation_history[-5:]):
            turn_text = f"{turn.get('user_input', '')} {turn.get('response', '')}".lower()
            turn_words = set(turn_text.split())
            
            overlap = query_words & turn_words
            
            if len(overlap) >= 2:
                references.append({
                    "turn_index": len(conversation_history) - 5 + i,
                    "overlap_words": list(overlap)[:5],
                    "relevance_score": len(overlap) / len(query_words) if query_words else 0
                })
        
        return sorted(references, key=lambda x: x["relevance_score"], reverse=True)[:3]







# """
# Query Understanding Module
# Uses LLM to understand user intent BEFORE taking action

# This is the BRAIN of the system - analyzes what user actually wants,
# not just what keywords they used.
# """

# import logging
# import json
# import re
# from typing import Dict, Any, Optional, List
# from datetime import datetime

# logger = logging.getLogger(__name__)

# class QueryUnderstanding:
#     """
#     LLM-based query understanding system
    
#     Instead of keyword matching, we use the LLM to:
#     1. Classify intent (what does user want?)
#     2. Extract entities (what are they talking about?)
#     3. Determine if real-time data is needed
#     4. Check if this is a correction, question, or command
#     5. Understand context from conversation history
#     """
    
#     def __init__(self):
#         # Intent categories
#         self.intent_categories = [
#             "information_request",      # User wants to know something
#             "memory_storage",           # User wants system to remember something
#             "correction",               # User is correcting previous information
#             "follow_up",                # User is continuing previous topic
#             "greeting",                 # Hello, hi, etc.
#             "personal_info",            # About user (name, location, etc.)
#             "command",                  # User telling system to do something
#             "confirmation",             # User confirming or denying something
#             "casual_chat",              # General conversation
#             "math_calculation",         # Mathematical queries
#             "opinion_request",          # Asking for opinion/advice
#         ]
        
#         # Real-time data types
#         self.realtime_types = [
#             "weather", "time", "date", "news", "pm", "president", 
#             "sports", "stocks", "none"
#         ]
    
#     def analyze_query(
#         self, 
#         query: str, 
#         conversation_history: Optional[List[Dict]] = None,
#         user_profile: Optional[Dict] = None
#     ) -> Dict[str, Any]:
#         """
#         Main analysis method - uses rule-based + semantic understanding
        
#         Args:
#             query: User's input
#             conversation_history: Recent conversation turns
#             user_profile: Known user info (name, location, etc.)
        
#         Returns:
#             Comprehensive query analysis dictionary
#         """
#         analysis = {
#             "original_query": query,
#             "timestamp": datetime.now().isoformat(),
#             "intent": self._classify_intent(query, conversation_history),
#             "entities": self._extract_entities(query, user_profile),
#             "needs_realtime": False,
#             "realtime_type": "none",
#             "is_follow_up": self._is_follow_up(query, conversation_history),
#             "is_correction": self._is_correction(query, conversation_history),
#             "is_question": self._is_question(query),
#             "confidence": 0.0,
#             "suggested_action": "",
#             "context_references": []
#         }
        
#         # Determine if real-time data is actually needed
#         analysis["needs_realtime"], analysis["realtime_type"] = self._check_realtime_need(
#             query, analysis["intent"], analysis["entities"]
#         )
        
#         # Determine suggested action
#         analysis["suggested_action"] = self._determine_action(analysis)
        
#         # Calculate confidence score
#         analysis["confidence"] = self._calculate_confidence(analysis)
        
#         # Find context references from conversation history
#         if conversation_history:
#             analysis["context_references"] = self._find_context_references(
#                 query, conversation_history
#             )
        
#         logger.debug(f"Query analysis: {analysis['intent']} (confidence: {analysis['confidence']:.2f})")
        
#         return analysis
    
#     def _classify_intent(self, query: str, conversation_history: Optional[List[Dict]] = None) -> str:
#         """
#         Classify user intent using semantic analysis
        
#         This is smarter than keyword matching - looks at sentence structure,
#         question patterns, and context.
#         """
#         query_lower = query.lower().strip()
        
#         # Remove common fillers for analysis
#         clean_query = query_lower
#         fillers = ["please", "can you", "could you", "i want", "i need", 
#                    "tell me", "let me know", "ok", "okay", "so", "well", "hey", "hi"]
#         for filler in fillers:
#             clean_query = clean_query.replace(filler, "")
#         clean_query = clean_query.strip()
        
#         # === GREETING DETECTION ===
#         greeting_patterns = [
#             r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening)",
#             r"^(how are you|how's it going|what's up|how do you do)"
#         ]
#         for pattern in greeting_patterns:
#             if re.search(pattern, clean_query):
#                 return "greeting"
        
#         # === MEMORY STORAGE DETECTION ===
#         # User wants system to remember something
#         memory_patterns = [
#             r"(remember|save|store|keep in mind|don't forget|note this|note that)",
#             r"(will you remember|can you remember|please remember|remember this|remember that)",
#             r"(i want you to remember|i need you to remember)"
#         ]
#         for pattern in memory_patterns:
#             if re.search(pattern, clean_query):
#                 return "memory_storage"
        
#         # === CORRECTION DETECTION ===
#         # User is correcting previous information
#         correction_patterns = [
#             r"^(no|wrong|incorrect|that's not right|actually|nah|nope)",
#             r"(it's not|it is not|that's wrong|you're wrong|incorrect|wrong information)",
#             r"(the correct|actually it's|actually it is|the real|the actual)"
#         ]
#         for pattern in correction_patterns:
#             if re.search(pattern, clean_query):
#                 return "correction"
        
#         # === CONFIRMATION DETECTION ===
#         # User confirming or denying
#         confirmation_patterns = [
#             r"^(yes|yeah|yep|correct|right|exactly|that's right|that's correct)",
#             r"^(no|nah|nope|wrong|incorrect|that's wrong|not really)"
#         ]
#         for pattern in confirmation_patterns:
#             if re.search(pattern, clean_query):
#                 return "confirmation"
        
#         # === PERSONAL INFO DETECTION ===
#         # Questions about user's own information
#         personal_patterns = [
#             r"(my name|what's my name|whats my name|do you know my name)",
#             r"(where do i live|where i live|my location|where am i from)",
#             r"(my brother|my sister|my family|my friend|my mother|my father)",
#             r"(who am i|what about me|tell me about me)"
#         ]
#         for pattern in personal_patterns:
#             if re.search(pattern, clean_query):
#                 return "personal_info"
        
#         # === MATH/CALCULATION DETECTION ===
#         math_patterns = [
#             r"\d+\s*[\+\-\*\/]\s*\d+",  # 2+2, 5*3, etc.
#             r"(what is|what's|whats|calculate|solve)\s*\d+",
#             r"(plus|minus|times|divided by|multiplied by)"
#         ]
#         for pattern in math_patterns:
#             if re.search(pattern, clean_query):
#                 return "math_calculation"
        
#         # === INFORMATION REQUEST DETECTION ===
#         # User wants to know something (most common)
#         info_patterns = [
#             r"^(what|who|when|where|why|how|which|whom|whose)",
#             r"^(tell me about|explain|describe|what can you|what are you)",
#             r"^(is|are|was|were|does|do|did|can|could|will|would)\s+\w+\s+\?",
#             r"(i want to know|i need to know|do you know|can you tell me)"
#         ]
#         for pattern in info_patterns:
#             if re.search(pattern, clean_query):
#                 return "information_request"
        
#         # === COMMAND DETECTION ===
#         command_patterns = [
#             r"^(show|display|open|run|execute|start|stop|create|delete|update)",
#             r"(i want you to|please|can you|could you)\s+(show|tell|find|get|search)"
#         ]
#         for pattern in command_patterns:
#             if re.search(pattern, clean_query):
#                 return "command"
        
#         # === FOLLOW-UP DETECTION ===
#         # Check if this continues previous conversation
#         if conversation_history and len(conversation_history) > 0:
#             follow_up_indicators = [
#                 "what about", "how about", "and", "also", "too", "else",
#                 "what else", "tell me more", "continue", "go on",
#                 "there", "that", "those", "it", "they", "them", "this"
#             ]
#             if any(indicator in clean_query for indicator in follow_up_indicators):
#                 # Check if query is short (more likely follow-up)
#                 if len(query.split()) < 10:
#                     return "follow_up"
        
#         # === CASUAL CHAT (default) ===
#         return "casual_chat"
    
#     def _extract_entities(self, query: str, user_profile: Optional[Dict] = None) -> Dict[str, Any]:
#         """
#         Extract entities from query using pattern matching + context
        
#         Returns dictionary of found entities
#         """
#         entities = {
#             "person_names": [],
#             "locations": [],
#             "organizations": [],
#             "dates": [],
#             "numbers": [],
#             "topics": []
#         }
        
#         query_lower = query.lower()
        
#         # === PERSON NAMES ===
#         # Look for capitalized words (potential names)
#         name_patterns = [
#             r"(?:name is|named|called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
#             r"(?:pm|president|minister|cm)\s+(?:of\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
#             r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:is|was|became)"
#         ]
#         for pattern in name_patterns:
#             matches = re.findall(pattern, query)
#             entities["person_names"].extend(matches)
        
#         # === LOCATIONS ===
#         location_patterns = [
#             r"(?:in|at|from|of|for)\s+([A-Z][A-Za-z\s,]+?)(?:\?|\.|,|!|$)",
#             r"(?:live|stay|reside)\s+(?:in\s+)?([A-Za-z\s,]+?)(?:\.|,|!|\?|$)"
#         ]
#         for pattern in location_patterns:
#             matches = re.findall(pattern, query)
#             for match in matches:
#                 location = match.strip()
#                 # Filter false positives
#                 if location.lower() not in ['the', 'what', 'when', 'where', 'how', 'today']:
#                     entities["locations"].append(location)
        
#         # Add user's stored location
#         if user_profile and user_profile.get('location'):
#             entities["locations"].append(user_profile['location'])
        
#         # === ORGANIZATIONS ===
#         org_patterns = [
#             r"(?:company|organization|agency|department)\s+(?:of|called|named)?\s+([A-Z][A-Za-z\s]+)",
#             r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:Ltd|Inc|Corp|Company|University))"
#         ]
#         for pattern in org_patterns:
#             matches = re.findall(pattern, query)
#             entities["organizations"].extend(matches)
        
#         # === DATES ===
#         date_patterns = [
#             r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
#             r"(\w+\s+\d{1,2},?\s+\d{4})",
#             r"(today|tomorrow|yesterday|next week|last month)"
#         ]
#         for pattern in date_patterns:
#             matches = re.findall(pattern, query_lower)
#             entities["dates"].extend(matches)
        
#         # === NUMBERS ===
#         number_patterns = [
#             r"(\d+(?:\.\d+)?(?:\s*(?:percent|percentage|degrees|km|kg|lbs|hours|days))?)",
#         ]
#         matches = re.findall(number_patterns[0], query_lower)
#         entities["numbers"].extend(matches)
        
#         # === TOPICS ===
#         # Extract main topic words (nouns, typically)
#         topic_keywords = [
#             "weather", "time", "date", "news", "cricket", "match", "game",
#             "politics", "election", "government", "minister", "president",
#             "economy", "stock", "market", "sports", "movie", "music"
#         ]
#         for keyword in topic_keywords:
#             if keyword in query_lower:
#                 entities["topics"].append(keyword)
        
#         # Deduplicate
#         for key in entities:
#             entities[key] = list(set(entities[key]))
        
#         return entities
    
#     def _is_follow_up(self, query: str, conversation_history: Optional[List[Dict]] = None) -> bool:
#         """Check if query is a follow-up to previous conversation"""
#         if not conversation_history or len(conversation_history) == 0:
#             return False
        
#         query_lower = query.lower()
        
#         # Follow-up indicators
#         follow_up_words = [
#             "there", "that", "those", "it", "they", "them", "this",
#             "what about", "how about", "and", "also", "too", "else",
#             "what else", "tell me more", "continue", "go on",
#             "why", "because", "so", "then", "but", "however"
#         ]
        
#         # Check for pronouns without clear subject
#         pronouns = ["he", "she", "it", "they", "him", "her", "them", "his", "hers"]
        
#         has_follow_up_indicator = any(word in query_lower for word in follow_up_words)
#         has_pronoun = any(pronoun in query_lower.split() for pronoun in pronouns)
        
#         # Short queries with indicators are likely follow-ups
#         if (has_follow_up_indicator or has_pronoun) and len(query.split()) < 12:
#             return True
        
#         # Check if query references previous topic
#         last_turn = conversation_history[-1] if conversation_history else None
#         if last_turn:
#             last_topic = last_turn.get('query_type', 'general')
#             current_topics = self._extract_entities(query, {}).get('topics', [])
            
#             # If topics match, likely a follow-up
#             if last_topic in current_topics or any(t in query_lower for t in current_topics):
#                 return True
        
#         return False
    
#     def _is_correction(self, query: str, conversation_history: Optional[List[Dict]] = None) -> bool:
#         """Check if user is correcting previous information"""
#         query_lower = query.lower()
        
#         # Correction indicators
#         correction_words = [
#             "no", "wrong", "incorrect", "that's not right", "actually",
#             "nah", "nope", "it's not", "it is not", "that's wrong",
#             "you're wrong", "the correct", "the actual", "the real"
#         ]
        
#         if any(word in query_lower for word in correction_words):
#             return True
        
#         # Check if contradicting previous response
#         if conversation_history:
#             last_turn = conversation_history[-1] if conversation_history else None
#             if last_turn:
#                 last_response = last_turn.get('response', '').lower()
#                 # If query contains negation of something in last response
#                 negation_words = ["not", "no", "wrong", "incorrect"]
#                 if any(neg in query_lower for neg in negation_words):
#                     # Check for overlapping content
#                     query_words = set(query_lower.split())
#                     response_words = set(last_response.split())
#                     overlap = query_words & response_words
#                     if len(overlap) > 3:  # Significant overlap with negation
#                         return True
        
#         return False
    
#     def _is_question(self, query: str) -> bool:
#         """Check if query is a question"""
#         query_stripped = query.strip()
        
#         # Ends with question mark
#         if query_stripped.endswith('?'):
#             return True
        
#         # Starts with question words
#         question_words = ["what", "who", "when", "where", "why", "how", "which", "whom", "whose"]
#         query_lower = query_lower = query.lower()
#         if any(query_lower.startswith(word + " ") for word in question_words):
#             return True
        
#         # Inverted question structure
#         inversion_patterns = [
#             r"^(is|are|was|were|does|do|did|can|could|will|would|should|may|might)\s+",
#             r"^(do|does|did)\s+you\s+(know|have|want|need|think)"
#         ]
#         for pattern in inversion_patterns:
#             if re.search(pattern, query_lower):
#                 return True
        
#         return False
    
#     def _check_realtime_need(
#         self, 
#         query: str, 
#         intent: str, 
#         entities: Dict[str, Any]
#     ) -> tuple:
#         """
#         Determine if real-time data is actually needed
        
#         This is smarter than keyword matching - considers intent AND entities
#         """
#         query_lower = query.lower()
        
#         # If intent is not information request, probably don't need realtime
#         if intent not in ["information_request", "follow_up"]:
#             return False, "none"
        
#         # === WEATHER ===
#         weather_indicators = ["weather", "temperature", "degrees", "celsius", 
#                              "fahrenheit", "forecast", "how's the weather",
#                              "what's the weather", "is it raining", "is it sunny"]
#         if any(ind in query_lower for ind in weather_indicators):
#             return True, "weather"
        
#         # === TIME ===
#         time_indicators = ["what time", "current time", "time is it", 
#                           "what's the time", "whats the time", "o'clock"]
#         if any(ind in query_lower for ind in time_indicators):
#             return True, "time"
        
#         # === DATE ===
#         date_indicators = ["what date", "what's the date", "whats the date",
#                           "today's date", "todays date", "what day is it",
#                           "what is today", "what year", "what month"]
#         if any(ind in query_lower for ind in date_indicators):
#             return True, "date"
        
#         # === PRIME MINISTER / POLITICS ===
#         pm_indicators = ["prime minister", "pm of", "chief minister", 
#                         "cm of", "who is the pm", "current pm", "current minister"]
#         if any(ind in query_lower for ind in pm_indicators):
#             return True, "pm"
        
#         # === PRESIDENT ===
#         president_indicators = ["president", "head of state", "who is the president"]
#         if any(ind in query_lower for ind in president_indicators):
#             return True, "president"
        
#         # === NEWS / CURRENT EVENTS ===
#         news_indicators = ["latest news", "current news", "what's happening",
#                           "whats happening", "recent news", "news about",
#                           "what happened", "what's new"]
#         if any(ind in query_lower for ind in news_indicators):
#             return True, "news"
        
#         # === SPORTS ===
#         sports_indicators = ["cricket match", "football match", "game today",
#                             "match today", "sports news", "score", "who won"]
#         if any(ind in query_lower for ind in sports_indicators):
#             return True, "sports"
        
#         return False, "none"
    
#     def _determine_action(self, analysis: Dict[str, Any]) -> str:
#         """Determine what action to take based on analysis"""
#         intent = analysis["intent"]
        
#         action_map = {
#             "greeting": "respond_greeting",
#             "information_request": "generate_response",
#             "memory_storage": "store_in_memory",
#             "correction": "acknowledge_and_update",
#             "follow_up": "continue_context",
#             "personal_info": "check_personal_memory",
#             "command": "execute_command",
#             "confirmation": "acknowledge_confirmation",
#             "casual_chat": "generate_response",
#             "math_calculation": "calculate_and_respond"
#         }
        
#         # If needs realtime, override action
#         if analysis["needs_realtime"]:
#             return f"fetch_realtime_{analysis['realtime_type']}"
        
#         return action_map.get(intent, "generate_response")
    
#     def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
#         """Calculate confidence score for the analysis"""
#         confidence = 0.5  # Base confidence
        
#         # Boost for clear question structure
#         if analysis["is_question"]:
#             confidence += 0.1
        
#         # Boost for detected entities
#         entities = analysis["entities"]
#         total_entities = sum(len(v) for v in entities.values())
#         if total_entities > 0:
#             confidence += min(0.2, total_entities * 0.05)
        
#         # Boost for clear intent
#         if analysis["intent"] not in ["casual_chat", "unknown"]:
#             confidence += 0.1
        
#         # Reduce for ambiguous queries
#         if len(analysis["original_query"].split()) < 3:
#             confidence -= 0.1
        
#         return min(1.0, max(0.0, confidence))
    
#     def _find_context_references(
#         self, 
#         query: str, 
#         conversation_history: List[Dict]
#     ) -> List[Dict]:
#         """Find references to previous conversation turns"""
#         references = []
#         query_lower = query.lower()
#         query_words = set(query_lower.split())
        
#         # Look at last 5 turns
#         for i, turn in enumerate(conversation_history[-5:]):
#             turn_text = f"{turn.get('user_input', '')} {turn.get('response', '')}".lower()
#             turn_words = set(turn_text.split())
            
#             # Calculate overlap
#             overlap = query_words & turn_words
            
#             # If significant overlap, this turn is referenced
#             if len(overlap) >= 3:
#                 references.append({
#                     "turn_index": len(conversation_history) - 5 + i,
#                     "overlap_words": list(overlap)[:5],
#                     "relevance_score": len(overlap) / len(query_words) if query_words else 0
#                 })
        
#         return sorted(references, key=lambda x: x["relevance_score"], reverse=True)[:3]


# # Test function
# if __name__ == "__main__":
#     analyzer = QueryUnderstanding()
    
#     print("=" * 70)
#     print("Query Understanding Test Suite")
#     print("=" * 70)
    
#     test_queries = [
#         # (query, expected_intent, expected_realtime)
#         ("who is the current PM of Nepal?", "information_request", True),
#         ("remember my name is Shuvam", "memory_storage", False),
#         ("no, that's wrong", "correction", False),
#         ("what about the weather there?", "follow_up", True),
#         ("hi, how are you?", "greeting", False),
#         ("what's my name?", "personal_info", False),
#         ("will you remember this for me?", "memory_storage", False),
#         ("yes, that's correct", "confirmation", False),
#         ("what is 2+2?", "math_calculation", False),
#         ("tell me about yourself", "information_request", False),
#         ("no, its sushila karki na!", "correction", False),
#         ("current time in Nepal?", "information_request", True),
#     ]
    
#     for query, expected_intent, expected_realtime in test_queries:
#         analysis = analyzer.analyze_query(query)
        
#         intent_match = "✓" if analysis["intent"] == expected_intent else "✗"
#         realtime_match = "✓" if analysis["needs_realtime"] == expected_realtime else "✗"
        
#         print(f"\nQuery: {query}")
#         print(f"  Intent: {analysis['intent']} {intent_match}")
#         print(f"  Needs Realtime: {analysis['needs_realtime']} {realtime_match}")
#         print(f"  Entities: {analysis['entities']}")
#         print(f"  Is Follow-up: {analysis['is_follow_up']}")
#         print(f"  Is Correction: {analysis['is_correction']}")
#         print(f"  Suggested Action: {analysis['suggested_action']}")
#         print(f"  Confidence: {analysis['confidence']:.2f}")
    
#     print("\n" + "=" * 70)