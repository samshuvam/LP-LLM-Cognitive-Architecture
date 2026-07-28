"""
LP-LLM Cognitive Architecture Component
Authored by Shuvam (https://github.com/samshuvam)
"""

__author__ = "Shuvam (https://github.com/samshuvam)"

"""
Post-Response Validation Pipeline
Production-Ready with Background Worker

Features:
- Async validation (answer first, verify after)
- Separate storage for pending/verified/flagged responses
- Google Search API integration
- Audit trail for all validations
- Training candidate identification
- Background worker with retry mechanism
"""

import os
import json
import hashlib
import logging
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from googleapiclient.discovery import build
from .config import ValidationConfig, VALIDATION_DIR, GOOGLE_API_KEY, SEARCH_ENGINE_ID

logger = logging.getLogger(__name__)

class PostResponseValidator:
    """Post-response validation with async processing and audit storage"""
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else VALIDATION_DIR
        self.pending_dir = self.base_path / "pending"
        self.verified_dir = self.base_path / "verified"
        self.flagged_dir = self.base_path / "flagged"
        self.audit_file = self.base_path / "audit_log.json"
        
        # Create directories
        for d in [self.pending_dir, self.verified_dir, self.flagged_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Initialize Google Search API with error handling
        try:
            self.google_service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
            logger.debug("Google Search API initialized")
        except Exception as e:
            logger.error(f"Google Search API initialization failed: {e}")
            self.google_service = None
        
        # Load audit log
        self.audit_log = self._load_audit_log()
        
        logger.debug(f"Post-response validator initialized at {self.base_path}")
    
    def _load_audit_log(self) -> List[Dict]:
        """Load audit log from file"""
        if self.audit_file.exists():
            try:
                with open(self.audit_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_audit_log(self):
        """Save audit log to file"""
        try:
            with open(self.audit_file, 'w') as f:
                json.dump(self.audit_log, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving audit log: {e}")
    
    def should_validate(self, response: str) -> bool:
        """Determine if a response needs validation"""
        if len(response) >= ValidationConfig.MIN_RESPONSE_LENGTH:
            return True
        
        response_lower = response.lower()
        for indicator in ValidationConfig.CLAIM_INDICATORS:
            if indicator in response_lower:
                return True
        
        return False
    
    def store_pending(
        self,
        query: str,
        response: str,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:

        entry_id = hashlib.md5(f"{query}{response}{datetime.now()}".encode()).hexdigest()

        entry = {
            "id": entry_id,
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
            "metadata": metadata or {},
            "validation_result": None,
            "validated_at": None,
            "eligible_for_training": False
        }

        filepath = self.pending_dir / f"{entry_id}.json"

        try:
            with open(filepath, 'w') as f:
                json.dump(entry, f, indent=2, default=str)

            self.audit_log.append(entry)
            self._save_audit_log()

            logger.debug(f"Stored pending validation: {entry_id[:8]}")
            return entry_id

        except Exception as e:
            logger.error(f"Error storing pending validation: {e}")
            return None
    
    async def validate_entry(self, entry_id: str) -> Optional[Dict]:
        """Validate a pending entry and move to appropriate folder"""
        pending_file = self.pending_dir / f"{entry_id}.json"
        if not pending_file.exists():
            logger.debug(f"Pending entry not found: {entry_id}")
            return None
        
        try:
            with open(pending_file, 'r') as f:
                entry = json.load(f)
            
            # Run validation with retry
            validation_result = await self._validate_fact_with_retry(entry['response'], entry['query'])
            
            entry['validation_result'] = validation_result
            entry['validated_at'] = datetime.now().isoformat()
            
            confidence = validation_result.get('confidence', 0)
            
            if validation_result.get('is_valid', False) and confidence >= ValidationConfig.CONFIDENCE_THRESHOLD:
                entry['status'] = 'verified'
                entry['eligible_for_training'] = confidence >= ValidationConfig.HIGH_CONFIDENCE_THRESHOLD
                target_dir = self.verified_dir
            else:
                entry['status'] = 'flagged'
                entry['eligible_for_training'] = False
                target_dir = self.flagged_dir
            
            target_file = target_dir / f"{entry_id}.json"
            with open(target_file, 'w') as f:
                json.dump(entry, f, indent=2, default=str)
            
            pending_file.unlink()
            
            for i, log_entry in enumerate(self.audit_log):
                if log_entry['id'] == entry_id:
                    self.audit_log[i] = entry
                    break
            self._save_audit_log()
            
            logger.debug(f"Validated entry {entry_id[:8]}: {entry['status']}")
            return entry
            
        except Exception as e:
            logger.error(f"Error validating entry: {e}")
            return None
    
    async def _validate_fact_with_retry(self, fact: str, context: str = "", max_retries: int = 3) -> Dict[str, Any]:
        """Validate with retry mechanism (Category 10)"""
        for attempt in range(max_retries):
            try:
                return await self._validate_fact(fact, context)
            except Exception as e:
                logger.warning(f"Validation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(ValidationConfig.RETRY_DELAY_SECONDS)
        
        return {
            "is_valid": False,
            "confidence": 0.0,
            "sources": [],
            "analysis": f"Validation failed after {max_retries} attempts"
        }
    
    async def _validate_fact(self, fact: str, context: str = "") -> Dict[str, Any]:
        """Validate a fact using Google Search API"""
        try:
            if self.google_service is None:
                return {
                    "is_valid": False,
                    "confidence": 0.0,
                    "sources": [],
                    "analysis": "Google Search API not available"
                }
            
            search_query = f"{fact} {context}"
            search_results = await self._search_google(search_query)
            confidence = await self._analyze_search_results(search_results, fact)
            
            return {
                "is_valid": confidence >= ValidationConfig.CONFIDENCE_THRESHOLD,
                "confidence": confidence,
                "sources": search_results[:ValidationConfig.MAX_SEARCH_RESULTS],
                "analysis": f"Fact validation score: {confidence:.2f}",
                "search_query": search_query
            }
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                "is_valid": False,
                "confidence": 0.0,
                "sources": [],
                "analysis": f"Validation failed: {str(e)}"
            }
    
    async def _search_google(self, query: str) -> List[Dict[str, str]]:
        """Perform Google search using Custom Search API"""
        try:
            if not self.google_service:
                return []

            result = self.google_service.cse().list(
                q=query,
                cx=SEARCH_ENGINE_ID,
                num=ValidationConfig.MAX_SEARCH_RESULTS
            ).execute()
            
            search_results = []
            if 'items' in result:
                for item in result['items']:
                    search_results.append({
                        "title": item.get('title', ''),
                        "url": item.get('link', ''),
                        "snippet": item.get('snippet', '')
                    })
            return search_results
            
        except Exception as e:
            logger.error(f"Google search error: {e}")
            return []
    
    async def _analyze_search_results(self, results: List[Dict], target_fact: str) -> float:
        """Analyze search results to determine fact validity"""
        if not results:
            return 0.0
        
        target_lower = target_fact.lower()
        confidence_scores = []
        
        for result in results:
            content = (result.get('title', '') + ' ' + result.get('snippet', '')).lower()
            
            if target_lower in content:
                confidence_scores.append(0.9)
            elif len(set(target_lower.split()) & set(content.split())) > 2:
                confidence_scores.append(0.7)
            elif any(word in content for word in target_lower.split() if len(word) > 3):
                confidence_scores.append(0.5)
            else:
                confidence_scores.append(0.2)
        
        return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    
    def get_training_candidates(self, min_confidence: Optional[float] = None) -> List[Dict]:
        """Get verified entries eligible for LoRA training"""
        min_conf = min_confidence or ValidationConfig.HIGH_CONFIDENCE_THRESHOLD
        candidates = []
        
        for filepath in self.verified_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    entry = json.load(f)
                
                if entry.get('eligible_for_training', False):
                    confidence = entry.get('validation_result', {}).get('confidence', 0)
                    if confidence >= min_conf:
                        candidates.append({
                            "instruction": entry['query'],
                            "output": entry['response'],
                            "concept": entry.get('metadata', {}).get('concept', 'general'),
                            "confidence": confidence,
                            "entry_id": entry['id'],
                            "validated_at": entry.get('validated_at')
                        })
            except:
                continue
        
        return candidates
    
    def get_validation_statistics(self) -> Dict:
        """Get validation statistics (Category 5)"""
        pending_count = len(list(self.pending_dir.glob("*.json")))
        verified_count = len(list(self.verified_dir.glob("*.json")))
        flagged_count = len(list(self.flagged_dir.glob("*.json")))
        
        total_validated = verified_count + flagged_count
        success_rate = verified_count / total_validated if total_validated > 0 else 0
        
        confidences = []
        for filepath in self.verified_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    entry = json.load(f)
                conf = entry.get('validation_result', {}).get('confidence', 0)
                confidences.append(conf)
            except:
                continue
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "pending_count": pending_count,
            "verified_count": verified_count,
            "flagged_count": flagged_count,
            "success_rate": success_rate,
            "average_confidence": avg_confidence,
            "training_candidates": len(self.get_training_candidates())
        }
    
    async def process_pending_batch(self, batch_size: Optional[int] = None) -> int:
        """Process a batch of pending validations (Category 5)"""
        batch_size = batch_size or ValidationConfig.VALIDATION_BATCH_SIZE
        
        pending_files = list(self.pending_dir.glob("*.json"))[:batch_size]
        
        if not pending_files:
            return 0
        
        tasks = [self.validate_entry(pf.stem) for pf in pending_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))
        
        logger.debug(f"Processed {successful}/{len(pending_files)} pending validations")
        return successful


class ValidationScheduler:
    """Background scheduler for async validation processing (Category 5)"""
    
    def __init__(self, validator: PostResponseValidator):
        self.validator = validator
        self.running = False
        self.task = None
        self.check_interval = ValidationConfig.VALIDATION_CHECK_INTERVAL_SECONDS
    
    async def start(self, check_interval: Optional[int] = None):
        """Start background validation processing"""
        self.check_interval = check_interval or self.check_interval
        self.running = True
        
        logger.debug(f"Validation scheduler started (interval: {self.check_interval}s)")
        
        while self.running:
            try:
                await self.validator.process_pending_batch()
            except Exception as e:
                logger.error(f"Validation scheduler error: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    def stop(self):
        """Stop background validation processing"""
        self.running = False
        if self.task:
            self.task.cancel()

















# """
# Post-Response Validation Pipeline
# Features:
# - Async validation (answer first, verify after)
# - Separate storage for pending/verified/flagged responses
# - Google Search API integration
# - Audit trail for all validations
# - Training candidate identification
# """
# from typing import Dict, List, Any, Optional  # Added Optional

# import os
# import json
# import hashlib
# import logging
# import asyncio
# import aiohttp
# from datetime import datetime
# from typing import Dict, List, Any, Optional
# from pathlib import Path
# from googleapiclient.discovery import build

# from config import ValidationConfig, VALIDATION_DIR, GOOGLE_API_KEY, SEARCH_ENGINE_ID

# logger = logging.getLogger(__name__)

# class PostResponseValidator:
#     """Post-response validation with async processing and audit storage"""
    
#     def __init__(self, base_path: Optional[str] = None):
#         self.base_path = Path(base_path) if base_path else VALIDATION_DIR
#         self.pending_dir = self.base_path / "pending"
#         self.verified_dir = self.base_path / "verified"
#         self.flagged_dir = self.base_path / "flagged"
#         self.audit_file = self.base_path / "audit_log.json"
        
#         # Create directories
#         for d in [self.pending_dir, self.verified_dir, self.flagged_dir]:
#             d.mkdir(parents=True, exist_ok=True)
        
#         # Initialize Google Search API
#         self.google_service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        
#         # Load audit log
#         self.audit_log = self._load_audit_log()
        
#         logger.info(f"Post-response validator initialized at {self.base_path}")
    
#     def _load_audit_log(self) -> List[Dict]:
#         """Load audit log from file"""
#         if self.audit_file.exists():
#             with open(self.audit_file, 'r') as f:
#                 return json.load(f)
#         return []
    
#     def _save_audit_log(self):
#         """Save audit log to file"""
#         with open(self.audit_file, 'w') as f:
#             json.dump(self.audit_log, f, indent=2, default=str)
    
#     def should_validate(self, response: str) -> bool:
#         """Determine if a response needs validation"""
#         # Validate if response is long enough
#         if len(response) >= ValidationConfig.MIN_RESPONSE_LENGTH:
#             return True
        
#         # Check for claim-making phrases
#         response_lower = response.lower()
#         for indicator in ValidationConfig.CLAIM_INDICATORS:
#             if indicator in response_lower:
#                 return True
        
#         return False
    
#     def store_pending(
#         self, 
#         query: str, 
#         response: str, 
#         metadata: Optional[Dict] = None
#     ) -> str:
#         """
#         Store response awaiting validation
        
#         Args:
#             query: User's query
#             response: AI's response
#             meta Additional metadata
        
#         Returns:
#             entry_id: Unique ID for this validation entry
#         """
#         entry_id = hashlib.md5(f"{query}{response}{datetime.now()}".encode()).hexdigest()
        
#         entry = {
#             "id": entry_id,
#             "query": query,
#             "response": response,
#             "timestamp": datetime.now().isoformat(),
#             "status": "pending",
#             "metadata": metadata or {},
#             "validation_result": None,
#             "validated_at": None,
#             "eligible_for_training": False
#         }
        
#         # Save to pending directory
#         filepath = self.pending_dir / f"{entry_id}.json"
#         with open(filepath, 'w') as f:
#             json.dump(entry, f, indent=2, default=str)
        
#         # Add to audit log
#         self.audit_log.append(entry)
#         self._save_audit_log()
        
#         logger.debug(f"Stored pending validation: {entry_id[:8]}")
#         return entry_id
    
#     async def validate_entry(self, entry_id: str) -> Optional[Dict]:
#         """
#         Validate a pending entry and move to appropriate folder
        
#         Args:
#             entry_id: The ID of the entry to validate
        
#         Returns:
#             Updated entry dictionary or None if not found
#         """
#         # Find the entry
#         pending_file = self.pending_dir / f"{entry_id}.json"
#         if not pending_file.exists():
#             logger.warning(f"Pending entry not found: {entry_id}")
#             return None
        
#         # Load entry
#         with open(pending_file, 'r') as f:
#             entry = json.load(f)
        
#         # Run validation
#         validation_result = await self._validate_fact(
#             entry['response'],
#             entry['query']
#         )
        
#         # Update entry
#         entry['validation_result'] = validation_result
#         entry['validated_at'] = datetime.now().isoformat()
        
#         # Determine destination based on validation result
#         confidence = validation_result.get('confidence', 0)
        
#         if validation_result.get('is_valid', False) and confidence >= ValidationConfig.CONFIDENCE_THRESHOLD:
#             # Move to verified
#             entry['status'] = 'verified'
#             entry['eligible_for_training'] = confidence >= ValidationConfig.HIGH_CONFIDENCE_THRESHOLD
#             target_dir = self.verified_dir
#         else:
#             # Move to flagged
#             entry['status'] = 'flagged'
#             entry['eligible_for_training'] = False
#             target_dir = self.flagged_dir
        
#         # Save to new location
#         target_file = target_dir / f"{entry_id}.json"
#         with open(target_file, 'w') as f:
#             json.dump(entry, f, indent=2, default=str)
        
#         # Remove from pending
#         pending_file.unlink()
        
#         # Update audit log
#         for i, log_entry in enumerate(self.audit_log):
#             if log_entry['id'] == entry_id:
#                 self.audit_log[i] = entry
#                 break
#         self._save_audit_log()
        
#         logger.info(f"Validated entry {entry_id[:8]}: {entry['status']}")
#         return entry
    
#     async def _validate_fact(self, fact: str, context: str = "") -> Dict[str, Any]:
#         """Validate a fact using Google Search API"""
#         try:
#             # Perform Google search
#             search_query = f"{fact} {context}"
#             search_results = await self._search_google(search_query)
            
#             # Analyze results
#             confidence = await self._analyze_search_results(search_results, fact)
            
#             return {
#                 "is_valid": confidence >= ValidationConfig.CONFIDENCE_THRESHOLD,
#                 "confidence": confidence,
#                 "sources": search_results[:ValidationConfig.MAX_SEARCH_RESULTS],
#                 "analysis": f"Fact validation score: {confidence:.2f}",
#                 "search_query": search_query
#             }
#         except Exception as e:
#             logger.error(f"Validation error: {e}")
#             return {
#                 "is_valid": False,
#                 "confidence": 0.0,
#                 "sources": [],
#                 "analysis": f"Validation failed: {str(e)}"
#             }
    
#     async def _search_google(self, query: str) -> List[Dict[str, str]]:
#         """Perform Google search using Custom Search API"""
#         try:
#             result = self.google_service.cse().list(
#                 q=query,
#                 cx=SEARCH_ENGINE_ID,
#                 num=ValidationConfig.MAX_SEARCH_RESULTS
#             ).execute()
            
#             search_results = []
#             if 'items' in result:
#                 for item in result['items']:
#                     search_results.append({
#                         "title": item.get('title', ''),
#                         "url": item.get('link', ''),
#                         "snippet": item.get('snippet', '')
#                     })
            
#             return search_results
            
#         except Exception as e:
#             logger.error(f"Google search error: {e}")
#             return []
    
#     async def _analyze_search_results(self, results: List[Dict], target_fact: str) -> float:
#         """Analyze search results to determine fact validity"""
#         if not results:
#             return 0.0
        
#         target_lower = target_fact.lower()
#         confidence_scores = []
        
#         for result in results:
#             content = (result.get('title', '') + ' ' + result.get('snippet', '')).lower()
            
#             # Exact phrase match
#             if target_lower in content:
#                 confidence_scores.append(0.9)
#             # Partial word matches
#             elif len(set(target_lower.split()) & set(content.split())) > 2:
#                 confidence_scores.append(0.7)
#             # Single word match
#             elif any(word in content for word in target_lower.split() if len(word) > 3):
#                 confidence_scores.append(0.5)
#             else:
#                 confidence_scores.append(0.2)
        
#         return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    
#     def get_training_candidates(self, min_confidence: Optional[float] = None) -> List[Dict]:
#         """Get verified entries eligible for LoRA training"""
#         min_conf = min_confidence or ValidationConfig.HIGH_CONFIDENCE_THRESHOLD
#         candidates = []
        
#         for filepath in self.verified_dir.glob("*.json"):
#             with open(filepath, 'r') as f:
#                 entry = json.load(f)
            
#             if entry.get('eligible_for_training', False):
#                 confidence = entry.get('validation_result', {}).get('confidence', 0)
#                 if confidence >= min_conf:
#                     candidates.append({
#                         "instruction": entry['query'],
#                         "output": entry['response'],
#                         "concept": entry.get('metadata', {}).get('concept', 'general'),
#                         "confidence": confidence,
#                         "entry_id": entry['id'],
#                         "validated_at": entry.get('validated_at')
#                     })
        
#         return candidates
    
#     def get_validation_statistics(self) -> Dict:
#         """Get validation statistics"""
#         pending_count = len(list(self.pending_dir.glob("*.json")))
#         verified_count = len(list(self.verified_dir.glob("*.json")))
#         flagged_count = len(list(self.flagged_dir.glob("*.json")))
        
#         # Calculate success rate
#         total_validated = verified_count + flagged_count
#         success_rate = verified_count / total_validated if total_validated > 0 else 0
        
#         # Get average confidence
#         confidences = []
#         for filepath in self.verified_dir.glob("*.json"):
#             with open(filepath, 'r') as f:
#                 entry = json.load(f)
#             conf = entry.get('validation_result', {}).get('confidence', 0)
#             confidences.append(conf)
        
#         avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
#         return {
#             "pending_count": pending_count,
#             "verified_count": verified_count,
#             "flagged_count": flagged_count,
#             "success_rate": success_rate,
#             "average_confidence": avg_confidence,
#             "training_candidates": len(self.get_training_candidates())
#         }
    
#     async def process_pending_batch(self, batch_size: Optional[int] = None) -> int:
#         """Process a batch of pending validations"""
#         batch_size = batch_size or ValidationConfig.VALIDATION_BATCH_SIZE
        
#         pending_files = list(self.pending_dir.glob("*.json"))[:batch_size]
        
#         if not pending_files:
#             return 0
        
#         # Process in parallel
#         tasks = [self.validate_entry(pf.stem) for pf in pending_files]
#         results = await asyncio.gather(*tasks, return_exceptions=True)
        
#         successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))
        
#         logger.info(f"Processed {successful}/{len(pending_files)} pending validations")
#         return successful


# class ValidationScheduler:
#     """Background scheduler for async validation processing"""
    
#     def __init__(self, validator: PostResponseValidator):
#         self.validator = validator
#         self.running = False
#         self.task = None
    
#     async def start(self, check_interval: Optional[int] = None):
#         """Start background validation processing"""
#         check_interval = check_interval or ValidationConfig.VALIDATION_CHECK_INTERVAL_SECONDS
#         self.running = True
        
#         logger.info(f"Starting validation scheduler (interval: {check_interval}s)")
        
#         while self.running:
#             try:
#                 await self.validator.process_pending_batch()
#             except Exception as e:
#                 logger.error(f"Validation scheduler error: {e}")
            
#             await asyncio.sleep(check_interval)
    
#     def stop(self):
#         """Stop background validation processing"""
#         self.running = False
#         if self.task:
#             self.task.cancel()