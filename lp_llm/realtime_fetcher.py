"""
LP-LLM Cognitive Architecture Component
Authored by Shuvam (https://github.com/samshuvam)
"""

__author__ = "Shuvam (https://github.com/samshuvam)"

"""
Real-Time Information Fetcher
Production-Ready with Google Search API

Features:
- Weather, time, date, PM, news fetching
- Multiple fallback strategies
- Comprehensive error handling
- Timezone support
"""

import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
import pytz
from .config import GOOGLE_API_KEY, SEARCH_ENGINE_ID
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)


class RealTimeFetcher:
    """Fetch real-time information using Google Search API"""
    
    def __init__(self):
        try:
            self.service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
            logger.debug("Google Search API initialized")
        except Exception as e:
            logger.error(f"Google Search API initialization failed: {e}")
            self.service = None
        
        self.query_patterns = {
            "weather": {
                "primary": ["weather", "temperature", "degrees", "celsius", "fahrenheit", "forecast"],
                "phrases": ["what's the weather", "whats the weather", "how's the weather"]
            },
            "time": {
                "primary": ["what time", "current time", "time is it", "o'clock"],
                "phrases": ["what's the time", "whats the time"]
            },
            "date": {
                "primary": ["what date", "today's date", "todays date", "what day", "what year", "what month"],
                "phrases": ["what is today", "whats today"]
            },
            "pm": {
                "primary": ["prime minister", "pm of", "chief minister"],
                "phrases": ["who is the prime minister", "current pm"]
            },
            "news": {
                "primary": ["latest news", "current news", "what's happening"],
                "phrases": ["what's new", "recent news"]
            }
        }
    
    def detect_query_type(self, query: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """Detect what type of real-time information is being requested"""
        query_lower = query.lower().strip()
        
        fillers = ["my name is", "i am", "i'm", "do you know", "can you", "please", "tell me"]
        for filler in fillers:
            query_lower = query_lower.replace(filler, "")
        query_lower = query_lower.strip()
        
        for query_type, patterns in self.query_patterns.items():
            for phrase in patterns.get('phrases', []):
                if phrase in query_lower:
                    logger.debug(f"Query type '{query_type}' detected by phrase: {phrase}")
                    return query_type
        
        for query_type, patterns in self.query_patterns.items():
            for keyword in patterns.get('primary', []):
                if keyword in query_lower:
                    logger.debug(f"Query type '{query_type}' detected by keyword: {keyword}")
                    return query_type
        
        logger.debug(f"Query classified as 'general': {query:.50}...")
        return "general"
    
    def extract_location(self, query: str, user_location: Optional[str] = None) -> str:
        """Extract location from query or use user's stored location"""
        location_patterns = [
            r"in\s+([A-Z][A-Za-z\s,]+?)(?:\?|\.|,|!|$)",
            r"of\s+([A-Z][A-Za-z\s,]+?)(?:\?|\.|,|!|$)",
            r"for\s+([A-Z][A-Za-z\s,]+?)(?:\?|\.|,|!|$)",
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, query)
            if match:
                location = match.group(1).strip()
                false_positives = ['the', 'what', 'when', 'where', 'how', 'today', 'tomorrow', 'current', 'latest']
                if location.lower() not in false_positives and len(location) > 2:
                    logger.debug(f"Location extracted from query: {location}")
                    return location
        
        if user_location:
            logger.debug(f"Using user's stored location: {user_location}")
            return user_location
        
        logger.debug("Using default location: Nepal")
        return "Nepal"
    
    def search(self, query: str, num_results: int = 10) -> List[Dict[str, str]]:
        """Perform Google search"""
        if self.service is None:
            logger.warning("Google Search API not available")
            return []
        
        try:
            result = self.service.cse().list(
                q=query,
                cx=SEARCH_ENGINE_ID,
                num=num_results
            ).execute()
            
            if 'items' in result:
                results = [
                    {
                        "title": item.get('title', ''),
                        "url": item.get('link', ''),
                        "snippet": item.get('snippet', '')
                    }
                    for item in result['items']
                ]
                logger.debug(f"Search found {len(results)} results for: {query:.50}...")
                return results
            
            return []
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def fetch_weather(self, location: str) -> Dict[str, Any]:
        """Fetch current weather for a location"""
        query = f"current weather temperature in {location} celsius today"
        results = self.search(query)
        
        if results:
            for result in results:
                snippet = result['snippet']
                temp_match = re.search(r'(\d+)\s*°?\s*[CF]?', snippet)
                if temp_match:
                    temp = temp_match.group(1)
                    if 'F' in snippet.upper():
                        temp = round((int(temp) - 32) * 5/9)
                    
                    condition = "Unknown"
                    for word in ["sunny", "cloudy", "rainy", "partly cloudy", "clear", "overcast"]:
                        if word in snippet.lower():
                            condition = word.title()
                            break
                    
                    return {
                        "success": True,
                        "location": location,
                        "temperature": f"{temp}°C",
                        "condition": condition,
                        "source": result['url'].split('//')[1].split('/')[0] if result.get('url') else "unknown",
                        "timestamp": datetime.now().isoformat()
                    }
        
        return {
            "success": False,
            "error": "Could not fetch weather data",
            "fallback": f"I couldn't fetch current weather for {location}"
        }
    
    def fetch_time(self, location: str) -> Dict[str, Any]:
        """Fetch current time - ALWAYS use system time with timezone"""
        try:
            timezone_map = {
                "nepal": "Asia/Kathmandu",
                "kathmandu": "Asia/Kathmandu",
                "lalitpur": "Asia/Kathmandu",
                "janakpur": "Asia/Kathmandu",
                "pokhara": "Asia/Kathmandu",
                "india": "Asia/Kolkata",
                "delhi": "Asia/Kolkata",
                "usa": "America/New_York",
                "uk": "Europe/London",
                "london": "Europe/London",
            }
            
            location_lower = location.lower()
            tz_name = None
            
            for loc, tz in timezone_map.items():
                if loc in location_lower:
                    tz_name = tz
                    break
            
            if tz_name:
                tz = pytz.timezone(tz_name)
                current_time = datetime.now(tz).strftime("%I:%M %p")
                source = f"{tz_name} timezone"
            else:
                tz = pytz.timezone('Asia/Kathmandu')
                current_time = datetime.now(tz).strftime("%I:%M %p")
                source = "Asia/Kathmandu timezone"
            
            return {
                "success": True,
                "location": location,
                "time": current_time,
                "source": source,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Time fetch error: {e}")
            return {
                "success": True,
                "location": location,
                "time": datetime.now().strftime("%I:%M %p"),
                "source": "system time",
                "timestamp": datetime.now().isoformat()
            }
    
    def fetch_date(self) -> Dict[str, Any]:
        """Fetch current date - ALWAYS use system date (NEVER hallucinate)"""
        today = datetime.now()
        return {
            "success": True,
            "date": today.strftime("%B %d, %Y"),
            "day": today.strftime("%A"),
            "formatted": today.strftime("%A, %B %d, %Y"),
            "year": today.strftime("%Y"),
            "month": today.strftime("%B"),
            "source": "system date",
            "timestamp": datetime.now().isoformat()
        }
    
    def fetch_pm(self, country: str) -> Dict[str, Any]:
        """Fetch current prime minister/leader information with cross-referencing"""
        # Append site preferences to query for higher quality results
        query = f"who is the current prime minister of {country} 2026 (site:wikipedia.org OR site:reuters.com OR site:apnews.com OR site:bbc.com)"
        
        results = self.search(query, num_results=10)
        
        # Fallback if strict query fails
        if not results:
             query = f"current prime minister of {country} 2026"
             results = self.search(query, num_results=10)
             
        if results:
            # Prioritize Wikipedia or trusted news sources if available
            best_result = results[0]
            trusted_domains = ['wikipedia.org', 'reuters.com', 'apnews.com', 'bbc.com', 'cnn.com', 'aljazeera.com']
            
            for result in results:
                url = result.get('url', '').lower()
                if any(domain in url for domain in trusted_domains):
                    best_result = result
                    break
                    
            snippet = best_result['snippet']
            title = best_result['title']
            
            # Combine title and snippet for better extraction context
            context = f"{title}. {snippet}"
            
            return {
                "success": True,
                "country": country,
                "pm_name": context[:200] + "..." if len(context) > 200 else context,
                "source": best_result['url'].split('//')[1].split('/')[0] if best_result.get('url') else "unknown",
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "success": False,
            "error": "Could not fetch PM information",
            "fallback": f"I couldn't fetch current PM information for {country}"
        }
    
    def fetch_news(self, topic: str) -> Dict[str, Any]:
        """Fetch latest news with trusted source prioritization"""
        query = f"latest news about {topic} today"
        results = self.search(query, num_results=10)
        
        if results:
            trusted_domains = ['reuters.com', 'apnews.com', 'bbc.com', 'cnn.com', 'aljazeera.com', 'bloomberg.com', 'wsj.com']
            verified_results = []
            
            # Identify trusted sources first
            for result in results:
                url = result.get('url', '').lower()
                if any(domain in url for domain in trusted_domains):
                    verified_results.append(result)
            
            # Fallback to general results if no trusted ones found
            if not verified_results:
                verified_results = results
                
            best_result = verified_results[0]
            
            return {
                "success": True,
                "topic": topic,
                "results": verified_results[:3],
                "summary": best_result['snippet'],
                "source": best_result['url'].split('//')[1].split('/')[0] if best_result.get('url') else "unknown",
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "success": False,
            "error": "No news found",
            "fallback": f"I couldn't fetch current news for {topic}."
        }
    
    def fetch(self, query: str, user_location: Optional[str] = None, 
              conversation_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Main fetch method"""
        query_type = self.detect_query_type(query, conversation_context)
        logger.info(f"Fetching {query_type} data for: {query:.50}...")
        
        if query_type == "weather":
            location = self.extract_location(query, user_location)
            return self.fetch_weather(location)
        
        elif query_type == "time":
            location = self.extract_location(query, user_location)
            return self.fetch_time(location)
        
        elif query_type == "date":
            return self.fetch_date()
        
        elif query_type == "pm":
            country_match = re.search(r'of\s+([A-Za-z]+)', query, re.IGNORECASE)
            country = country_match.group(1) if country_match else "Nepal"
            return self.fetch_pm(country)
        
        elif query_type == "news":
            return self.fetch_news(query)
        
        else:
            return {"success": False, "type": "general", "query_type": query_type}












            




# """
# Real-Time Information Fetcher
# Uses Google Search API + System Time for accurate real-time data
# Features:
# - Smart query type detection with context awareness
# - Multiple fallback strategies
# - Comprehensive error handling
# """

# import logging
# import re
# from typing import Dict, Any, Optional, List, Tuple
# from datetime import datetime
# import pytz
# from config import GOOGLE_API_KEY, SEARCH_ENGINE_ID
# from googleapiclient.discovery import build

# logger = logging.getLogger(__name__)

# class RealTimeFetcher:
#     """Fetch real-time information using Google Search API"""
    
#     def __init__(self):
#         self.service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        
#         # Comprehensive keyword mappings for query detection
#         self.query_patterns = {
#             "weather": {
#                 "primary": ["weather", "temperature", "degrees", "celsius", "fahrenheit", "forecast"],
#                 "secondary": ["hot", "cold", "rain", "sunny", "cloudy", "storm", "wind"],
#                 "phrases": ["what's the weather", "whats the weather", "how's the weather", 
#                            "how is the weather", "weather like", "temperature outside"]
#             },
#             "time": {
#                 "primary": ["what time", "current time", "time is it", "o'clock", "clock"],
#                 "secondary": ["am", "pm", "hour", "minutes", "now"],
#                 "phrases": ["what's the time", "whats the time", "tell me the time", 
#                            "can you tell me the time", "do you know the time"]
#             },
#             "date": {
#                 "primary": ["what date", "today's date", "todays date", "what day", "current date"],
#                 "secondary": ["year", "month", "tomorrow", "yesterday", "weekday"],
#                 "phrases": ["what is today", "whats today", "what's today", "what day is it",
#                            "what year is it", "what month is it", "what is the date"]
#             },
#             "pm": {
#                 "primary": ["prime minister", "pm of", "chief minister", "governor"],
#                 "secondary": ["minister", "government", "leader"],
#                 "phrases": ["who is the prime minister", "whos the prime minister", 
#                            "who's the pm", "current pm", "present pm"]
#             },
#             "president": {
#                 "primary": ["president", "head of state", "president of"],
#                 "secondary": ["leader", "country leader"],
#                 "phrases": ["who is the president", "whos the president", "current president"]
#             },
#             "news": {
#                 "primary": ["latest news", "current news", "recent news", "breaking news"],
#                 "secondary": ["happening", "trending", "headline"],
#                 "phrases": ["what's happening", "whats happening", "what is happening",
#                            "any news", "news about", "recent events"]
#             }
#         }
    
#     def detect_query_type(self, query: str, conversation_context: Optional[List[Dict]] = None) -> str:
#         """
#         Detect what type of real-time information is being requested
#         Uses multiple strategies for accuracy
        
#         Args:
#             query: User's query
#             conversation_context: Recent conversation history for context
        
#         Returns:
#             Query type string (weather, time, date, pm, president, news, general)
#         """
#         query_lower = query.lower().strip()
        
#         # Remove common conversational fillers
#         fillers = [
#             "my name is", "i am", "i'm", "do you know", "can you",
#             "could you", "please", "tell me", "i want to know",
#             "i need to know", "what is your", "what are you",
#             "hey", "hi", "hello", "ok", "okay", "so", "well"
#         ]
#         for filler in fillers:
#             query_lower = query_lower.replace(filler, "")
#         query_lower = query_lower.strip()
        
#         # Check for follow-up questions (pronouns, references)
#         if conversation_context:
#             follow_up_indicators = ["there", "that", "those", "it", "they", "them", 
#                                    "what about", "how about", "and", "also", "too"]
#             is_follow_up = any(indicator in query_lower for indicator in follow_up_indicators)
            
#             if is_follow_up:
#                 # Inherit query type from recent context if this is a follow-up
#                 recent_types = [ctx.get('query_type', 'general') for ctx in conversation_context[-3:]]
#                 if recent_types.count(recent_types[0]) >= 2:
#                     logger.debug(f"Follow-up detected, inheriting query type: {recent_types[0]}")
#                     return recent_types[0]
        
#         # Strategy 1: Check exact phrases first (most accurate)
#         for query_type, patterns in self.query_patterns.items():
#             for phrase in patterns.get('phrases', []):
#                 if phrase in query_lower:
#                     logger.debug(f"Query type '{query_type}' detected by phrase: {phrase}")
#                     return query_type
        
#         # Strategy 2: Check primary keywords
#         for query_type, patterns in self.query_patterns.items():
#             for keyword in patterns.get('primary', []):
#                 if keyword in query_lower:
#                     logger.debug(f"Query type '{query_type}' detected by primary keyword: {keyword}")
#                     return query_type
        
#         # Strategy 3: Check secondary keywords (with higher confidence threshold)
#         for query_type, patterns in self.query_patterns.items():
#             for keyword in patterns.get('secondary', []):
#                 if keyword in query_lower:
#                     # Require additional context for secondary keywords
#                     if len(query_lower) > 10:  # Longer queries more likely to be genuine
#                         logger.debug(f"Query type '{query_type}' detected by secondary keyword: {keyword}")
#                         return query_type
        
#         # Strategy 4: Check for question patterns
#         question_patterns = [
#             (r"what\s+(is|'s|are|was|were)\s+\w+\s+(weather|time|date|day|year|month)", "date"),
#             (r"how\s+(is|does|about)\s+\w+\s+(weather|climate|temperature)", "weather"),
#             (r"who\s+(is|'s|was|were)\s+\w+\s+(minister|president|leader)", "pm"),
#         ]
#         for pattern, query_type in question_patterns:
#             if re.search(pattern, query_lower):
#                 logger.debug(f"Query type '{query_type}' detected by pattern: {pattern}")
#                 return query_type
        
#         # Default: No real-time data needed
#         logger.debug(f"Query classified as 'general': {query[:50]}...")
#         return "general"
    
#     def extract_location(self, query: str, user_location: Optional[str] = None) -> str:
#         """
#         Extract location from query or use user's stored location
        
#         Args:
#             query: User's query
#             user_location: User's stored location from profile
        
#         Returns:
#             Location string
#         """
#         # Priority 1: Explicit location in query
#         location_patterns = [
#             (r"in\s+([A-Z][A-Za-z\s,]+?)(?:\?|\.|,|!|$)", "in"),
#             (r"of\s+([A-Z][A-Za-z\s,]+?)(?:\?|\.|,|!|$)", "of"),
#             (r"for\s+([A-Z][A-Za-z\s,]+?)(?:\?|\.|,|!|$)", "for"),
#             (r"at\s+([A-Z][A-Za-z\s,]+?)(?:\?|\.|,|!|$)", "at"),
#             (r"around\s+([A-Z][A-Za-z\s,]+?)(?:\?|\.|,|!|$)", "around"),
#             (r"near\s+([A-Z][A-Za-z\s,]+?)(?:\?|\.|,|!|$)", "near"),
#         ]
        
#         for pattern, prefix in location_patterns:
#             match = re.search(pattern, query)
#             if match:
#                 location = match.group(1).strip()
#                 # Filter common false positives
#                 false_positives = ['the', 'what', 'when', 'where', 'how', 'today', 
#                                   'tomorrow', 'current', 'latest', 'this', 'that', 'there']
#                 if location.lower() not in false_positives and len(location) > 2:
#                     logger.debug(f"Location extracted from query: {location}")
#                     return location
        
#         # Priority 2: User's stored location
#         if user_location:
#             logger.debug(f"Using user's stored location: {user_location}")
#             return user_location
        
#         # Priority 3: Default fallback
#         logger.debug("Using default location: Nepal")
#         return "Nepal"
    
#     def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
#         """
#         Perform Google search using Custom Search API
        
#         Args:
#             query: Search query
#             num_results: Number of results to fetch
        
#         Returns:
#             List of search results
#         """
#         try:
#             result = self.service.cse().list(
#                 q=query,
#                 cx=SEARCH_ENGINE_ID,
#                 num=num_results
#             ).execute()
            
#             if 'items' in result:
#                 results = [
#                     {
#                         "title": item.get('title', ''),
#                         "url": item.get('link', ''),
#                         "snippet": item.get('snippet', '')
#                     }
#                     for item in result['items']
#                 ]
#                 logger.debug(f"Search found {len(results)} results for: {query[:50]}...")
#                 return results
            
#             logger.warning(f"No search results for: {query[:50]}...")
#             return []
            
#         except Exception as e:
#             logger.error(f"Search error: {e}")
#             return []
    
#     def fetch_weather(self, location: str) -> Dict[str, Any]:
#         """Fetch current weather for a location"""
#         query = f"current weather temperature in {location} celsius today"
#         results = self.search(query)
        
#         if results:
#             for result in results:
#                 snippet = result['snippet']
                
#                 # Extract temperature
#                 temp_match = re.search(r'(\d+)\s*°?\s*[CF]?', snippet)
#                 if temp_match:
#                     temp = temp_match.group(1)
                    
#                     # Convert Fahrenheit to Celsius if needed
#                     if 'F' in snippet.upper() or 'fahrenheit' in snippet.lower():
#                         temp = round((int(temp) - 32) * 5/9)
                    
#                     # Extract weather condition
#                     condition = "Unknown"
#                     condition_keywords = {
#                         "sunny": ["sunny", "clear", "bright"],
#                         "cloudy": ["cloudy", "overcast", "grey"],
#                         "rainy": ["rainy", "rain", "showers", "wet"],
#                         "partly cloudy": ["partly cloudy", "partly sunny", "mixed"],
#                         "thunderstorm": ["thunderstorm", "storm", "lightning"],
#                         "foggy": ["foggy", "fog", "mist", "hazy"],
#                         "windy": ["windy", "wind", "breezy"]
#                     }
                    
#                     for cond, keywords in condition_keywords.items():
#                         if any(kw in snippet.lower() for kw in keywords):
#                             condition = cond.title()
#                             break
                    
#                     return {
#                         "success": True,
#                         "location": location,
#                         "temperature": f"{temp}°C",
#                         "condition": condition,
#                         "source": result['url'].split('//')[1].split('/')[0] if result.get('url') else "unknown",
#                         "timestamp": datetime.now().isoformat()
#                     }
        
#         return {
#             "success": False,
#             "error": "Could not fetch weather data",
#             "fallback": f"I couldn't fetch current weather for {location}. Please check a weather website."
#         }
    
#     def fetch_time(self, location: str) -> Dict[str, Any]:
#         """Fetch current time for a location - ALWAYS use system time with timezone"""
#         try:
#             # Map common locations to timezones
#             timezone_map = {
#                 "nepal": "Asia/Kathmandu",
#                 "kathmandu": "Asia/Kathmandu",
#                 "lalitpur": "Asia/Kathmandu",
#                 "janakpur": "Asia/Kathmandu",
#                 "pokhara": "Asia/Kathmandu",
#                 "india": "Asia/Kolkata",
#                 "delhi": "Asia/Kolkata",
#                 "mumbai": "Asia/Kolkata",
#                 "usa": "America/New_York",
#                 "uk": "Europe/London",
#                 "london": "Europe/London",
#                 "australia": "Australia/Sydney",
#                 "sydney": "Australia/Sydney",
#                 "japan": "Asia/Tokyo",
#                 "tokyo": "Asia/Tokyo",
#                 "china": "Asia/Shanghai",
#                 "beijing": "Asia/Shanghai",
#             }
            
#             location_lower = location.lower()
#             tz_name = None
            
#             for loc, tz in timezone_map.items():
#                 if loc in location_lower:
#                     tz_name = tz
#                     break
            
#             if tz_name:
#                 tz = pytz.timezone(tz_name)
#                 current_time = datetime.now(tz).strftime("%I:%M %p")
#                 source = f"{tz_name} timezone"
#             else:
#                 # Default to Nepal time
#                 tz = pytz.timezone('Asia/Kathmandu')
#                 current_time = datetime.now(tz).strftime("%I:%M %p")
#                 source = "Asia/Kathmandu timezone"
            
#             return {
#                 "success": True,
#                 "location": location,
#                 "time": current_time,
#                 "source": source,
#                 "timestamp": datetime.now().isoformat()
#             }
            
#         except Exception as e:
#             logger.error(f"Time fetch error: {e}")
#             # Ultimate fallback
#             return {
#                 "success": True,
#                 "location": location,
#                 "time": datetime.now().strftime("%I:%M %p"),
#                 "source": "system time",
#                 "timestamp": datetime.now().isoformat()
#             }
    
#     def fetch_date(self) -> Dict[str, Any]:
#         """Fetch current date - ALWAYS use system date (NEVER hallucinate)"""
#         today = datetime.now()
#         return {
#             "success": True,
#             "date": today.strftime("%B %d, %Y"),
#             "day": today.strftime("%A"),
#             "formatted": today.strftime("%A, %B %d, %Y"),
#             "year": today.strftime("%Y"),
#             "month": today.strftime("%B"),
#             "day_number": today.strftime("%d"),
#             "source": "system date",
#             "timestamp": datetime.now().isoformat()
#         }
    
#     def fetch_pm(self, country: str) -> Dict[str, Any]:
#         """Fetch current prime minister information"""
#         query = f"current prime minister of {country} 2026"
#         results = self.search(query)
        
#         if results:
#             snippet = results[0]['snippet']
            
#             # Try to extract name using patterns
#             name_patterns = [
#                 r'(?:is|was|named|called|served)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
#                 r'(?:prime minister|pm)\s+(?:is|was)?\s+([A-Z][A-Za-z\s]+?)(?:\.|,|$)',
#                 r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:became|appointed|selected)',
#             ]
            
#             for pattern in name_patterns:
#                 match = re.search(pattern, snippet)
#                 if match:
#                     return {
#                         "success": True,
#                         "country": country,
#                         "pm_name": match.group(1).strip(),
#                         "source": results[0]['url'].split('//')[1].split('/')[0] if results[0].get('url') else "unknown",
#                         "timestamp": datetime.now().isoformat()
#                     }
            
#             # Fallback: return snippet
#             return {
#                 "success": True,
#                 "country": country,
#                 "pm_name": snippet[:200],
#                 "source": results[0]['url'].split('//')[1].split('/')[0] if results[0].get('url') else "unknown",
#                 "timestamp": datetime.now().isoformat()
#             }
        
#         return {
#             "success": False,
#             "error": "Could not fetch PM information",
#             "fallback": f"I couldn't fetch current PM information for {country}"
#         }
    
#     def fetch_news(self, topic: str) -> Dict[str, Any]:
#         """Fetch latest news"""
#         query = f"latest news about {topic} today"
#         results = self.search(query)
        
#         if results:
#             return {
#                 "success": True,
#                 "topic": topic,
#                 "results": results[:3],
#                 "summary": results[0]['snippet'] if results else "",
#                 "source": results[0]['url'].split('//')[1].split('/')[0] if results[0].get('url') else "unknown",
#                 "timestamp": datetime.now().isoformat()
#             }
        
#         return {
#             "success": False,
#             "error": "No news found",
#             "fallback": "I couldn't fetch current news."
#         }
    
#     def fetch(self, query: str, user_location: Optional[str] = None, 
#               conversation_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
#         """
#         Main fetch method - detects query type and fetches appropriate data
        
#         Args:
#             query: User's query
#             user_location: User's stored location (optional)
#             conversation_context: Recent conversation history (optional)
        
#         Returns:
#             Dictionary with fetched data or error
#         """
#         # Detect query type with context awareness
#         query_type = self.detect_query_type(query, conversation_context)
#         logger.info(f"Fetching {query_type} data for: {query[:50]}...")
        
#         if query_type == "weather":
#             location = self.extract_location(query, user_location)
#             return self.fetch_weather(location)
        
#         elif query_type == "time":
#             location = self.extract_location(query, user_location)
#             return self.fetch_time(location)
        
#         elif query_type == "date":
#             return self.fetch_date()
        
#         elif query_type == "pm":
#             country_match = re.search(r'of\s+([A-Za-z]+)', query, re.IGNORECASE)
#             country = country_match.group(1) if country_match else "Nepal"
#             return self.fetch_pm(country)
        
#         elif query_type == "president":
#             country_match = re.search(r'of\s+([A-Za-z]+)', query, re.IGNORECASE)
#             country = country_match.group(1) if country_match else "Nepal"
#             return self.fetch_pm(country)
        
#         elif query_type == "news":
#             return self.fetch_news(query)
        
#         else:
#             # General query - don't fetch anything
#             return {"success": False, "type": "general", "query_type": query_type}


# # Test function
# if __name__ == "__main__":
#     fetcher = RealTimeFetcher()
    
#     print("=" * 60)
#     print("Testing RealTimeFetcher")
#     print("=" * 60)
    
#     test_queries = [
#         "what is today's date",
#         "what time is it in Nepal",
#         "weather in Lalitpur Nepal",
#         "who is the prime minister of Nepal",
#         "what year is it",
#         "do you know my name",
#         "what is 2+2",
#         "my name is shuvam"
#     ]
    
#     for query in test_queries:
#         print(f"\nQuery: {query}")
#         query_type = fetcher.detect_query_type(query)
#         print(f"Detected type: {query_type}")
        
#         if query_type != "general":
#             result = fetcher.fetch(query)
#             print(f"Result: {result.get('success', False)}")
    
#     print("\n" + "=" * 60)