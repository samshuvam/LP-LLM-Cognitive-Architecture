"""
LP-LLM Cognitive Architecture Component
Authored by Shuvam (https://github.com/samshuvam)
"""

__author__ = "Shuvam (https://github.com/samshuvam)"

"""
User Profile Management Module
Handles auto-save/load of user information across sessions
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from .config import USER_PROFILE_DIR, MemoryConfig

logger = logging.getLogger(__name__)

class UserProfile:
    """Manages user profile with auto-persistence"""
    
    def __init__(self, profile_name: str = "default"):
        self.profile_name = profile_name
        self.profile_file = USER_PROFILE_DIR / f"{profile_name}_profile.json"
        self.data = self._load_profile()
        
        # Auto-save timer
        self.last_save_time = datetime.now()
        self.auto_save_interval = MemoryConfig.AUTO_SAVE_INTERVAL_SECONDS
    
    def _load_profile(self) -> Dict[str, Any]:
        """Load profile from file"""
        if self.profile_file.exists():
            try:
                with open(self.profile_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded user profile: {self.profile_name}")
                return data
            except Exception as e:
                logger.error(f"Error loading profile: {e}")
                return self._create_default_profile()
        else:
            logger.info("Creating new user profile")
            return self._create_default_profile()
    
    def _create_default_profile(self) -> Dict[str, Any]:
        """Create default profile structure"""
        return {
            "profile_name": self.profile_name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "personal_info": {
                "name": None,
                "location": None,
                "email": None,
                "phone": None,
                "birthday": None
            },
            "preferences": {
                "language": "en",
                "timezone": "Asia/Kathmandu",
                "notification_enabled": True
            },
            "family": {
                "brother_name": None,
                "sister_name": None,
                "parent_names": []
            },
            "education": {
                "institution": None,
                "major": None,
                "semester": None,
                "graduation_year": None
            },
            "conversation_stats": {
                "total_conversations": 0,
                "total_messages": 0,
                "first_interaction": None,
                "last_interaction": None
            },
            "learned_facts": [],
            "corrections": []
        }
    
    def _save_profile(self):
        """Save profile to file"""
        try:
            self.data["updated_at"] = datetime.now().isoformat()
            
            # Ensure directory exists
            self.profile_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.profile_file, 'w') as f:
                json.dump(self.data, f, indent=2)
            
            self.last_save_time = datetime.now()
            logger.debug(f"Saved user profile: {self.profile_name}")
            
        except Exception as e:
            logger.error(f"Error saving profile: {e}")
    
    def auto_save_if_needed(self):
        """Auto-save if interval has passed"""
        elapsed = (datetime.now() - self.last_save_time).total_seconds()
        if elapsed >= self.auto_save_interval:
            self._save_profile()
    
    def update_personal_info(self, field: str, value: str):
        """Update personal information field"""
        if field in self.data["personal_info"]:
            old_value = self.data["personal_info"][field]
            self.data["personal_info"][field] = value
            
            # Log change
            if old_value != value:
                logger.info(f"Updated {field}: {old_value} -> {value}")
            
            self._save_profile()
            return True
        return False
    
    def update_family_info(self, field: str, value: str):
        """Update family information field"""
        if field in self.data["family"]:
            old_value = self.data["family"][field]
            self.data["family"][field] = value
            
            if old_value != value:
                logger.info(f"Updated family.{field}: {old_value} -> {value}")
            
            self._save_profile()
            return True
        return False
    
    def update_education_info(self, field: str, value: str):
        """Update education information field"""
        if field in self.data["education"]:
            old_value = self.data["education"][field]
            self.data["education"][field] = value
            
            if old_value != value:
                logger.info(f"Updated education.{field}: {old_value} -> {value}")
            
            self._save_profile()
            return True
        return False
    
    def add_learned_fact(self, fact: str, source: str = "conversation"):
        """Add a learned fact to profile"""
        fact_entry = {
            "fact": fact,
            "source": source,
            "learned_at": datetime.now().isoformat(),
            "verified": False
        }
        self.data["learned_facts"].append(fact_entry)
        self._save_profile()
        logger.debug(f"Added learned fact: {fact[:50]}...")
    
    def add_correction(self, original: str, corrected: str):
        """Record a correction made by user"""
        correction_entry = {
            "original": original,
            "corrected": corrected,
            "corrected_at": datetime.now().isoformat()
        }
        self.data["corrections"].append(correction_entry)
        
        # Keep only last 50 corrections
        if len(self.data["corrections"]) > 50:
            self.data["corrections"] = self.data["corrections"][-50:]
        
        self._save_profile()
        logger.info(f"Recorded correction: {original[:30]}... -> {corrected[:30]}...")
    
    def increment_conversation_stats(self):
        """Increment conversation statistics"""
        self.data["conversation_stats"]["total_conversations"] += 1
        self.data["conversation_stats"]["total_messages"] += 1
        
        now = datetime.now().isoformat()
        if not self.data["conversation_stats"]["first_interaction"]:
            self.data["conversation_stats"]["first_interaction"] = now
        self.data["conversation_stats"]["last_interaction"] = now
        
        self._save_profile()
    
    def get_name(self) -> Optional[str]:
        """Get user's name"""
        return self.data["personal_info"].get("name")
    
    def get_location(self) -> Optional[str]:
        """Get user's location"""
        return self.data["personal_info"].get("location")
    
    def get_all_info(self) -> Dict[str, Any]:
        """Get all profile information"""
        return self.data.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profile summary for status display"""
        return {
            "name": self.get_name(),
            "location": self.get_location(),
            "total_conversations": self.data["conversation_stats"]["total_conversations"],
            "learned_facts_count": len(self.data["learned_facts"]),
            "corrections_count": len(self.data["corrections"]),
            "last_updated": self.data["updated_at"]
        }
    
    def export_profile(self, export_path: Optional[str] = None) -> str:
        """Export profile to file"""
        if export_path is None:
            export_path = f"user_profile_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_file = Path(export_path)
        with open(export_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        logger.info(f"Exported profile to {export_file}")
        return str(export_file)
    
    def clear_profile(self):
        """Clear all profile data (keep structure)"""
        self.data = self._create_default_profile()
        self._save_profile()
        logger.warning("User profile cleared")