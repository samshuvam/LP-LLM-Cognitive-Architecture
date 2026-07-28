"""
LP-LLM Cognitive Architecture Component
Authored by Shuvam (https://github.com/samshuvam)
"""

__author__ = "Shuvam (https://github.com/samshuvam)"

"""
External API Tools for Real-Time Information
Weather, News, Search, etc.
"""

import requests
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================================
# WEATHER API CONFIGURATION
# ============================================================================
# Get free API key from: https://openweathermap.org/api
WEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"  # Replace with actual key
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

class WeatherTool:
    """Fetch real-time weather data"""
    
    @staticmethod
    def get_weather(location: str, units: str = "metric") -> Dict[str, Any]:
        """
        Get current weather for a location
        
        Args:
            location: City name (e.g., "Lalitpur,NP")
            units: "metric" for Celsius, "imperial" for Fahrenheit
        
        Returns:
            Dictionary with weather information
        """
        try:
            params = {
                "q": location,
                "appid": WEATHER_API_KEY,
                "units": units
            }
            
            response = requests.get(WEATHER_API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                "success": True,
                "location": data.get("name", location),
                "country": data.get("sys", {}).get("country", ""),
                "temperature": data.get("main", {}).get("temp", "N/A"),
                "feels_like": data.get("main", {}).get("feels_like", "N/A"),
                "humidity": data.get("main", {}).get("humidity", "N/A"),
                "description": data.get("weather", [{}])[0].get("description", "N/A"),
                "wind_speed": data.get("wind", {}).get("speed", "N/A"),
                "timestamp": datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Weather API error: {e}")
            return {
                "success": False,
                "error": str(e),
                "location": location
            }
    
    @staticmethod
    def format_weather_response(weather_data: Dict) -> str:
        """Format weather data into natural language response"""
        if not weather_data.get("success"):
            return f"I couldn't fetch weather data for {weather_data.get('location', 'that location')}. Error: {weather_data.get('error', 'Unknown error')}"
        
        return (
            f"🌤️ **Current Weather in {weather_data['location']}, {weather_data['country']}**\n\n"
            f"• Temperature: {weather_data['temperature']}°C (feels like {weather_data['feels_like']}°C)\n"
            f"• Conditions: {weather_data['description'].title()}\n"
            f"• Humidity: {weather_data['humidity']}%\n"
            f"• Wind Speed: {weather_data['wind_speed']} m/s\n"
            f"\n_Data fetched at {weather_data['timestamp']}_ "
        )


class LocationTool:
    """Geolocation and distance tools"""
    
    @staticmethod
    def get_coordinates(city: str, country: str = "") -> Optional[Dict]:
        """Get latitude/longitude for a city"""
        try:
            # Using OpenStreetMap Nominatim (free, no API key needed)
            query = f"{city}, {country}" if country else city
            url = "https://nominatim.openstreetmap.org/search"
            params = {"q": query, "format": "json", "limit": 1}
            headers = {"User-Agent": "Sentrix/1.0"}
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()
            
            if data:
                return {
                    "latitude": float(data[0]["lat"]),
                    "longitude": float(data[0]["lon"]),
                    "display_name": data[0]["display_name"]
                }
            return None
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
            return None


# ============================================================================
# TOOL REGISTRY
# ============================================================================

class ToolRegistry:
    """Central registry for all available tools"""
    
    def __init__(self):
        self.tools = {
            "get_weather": WeatherTool.get_weather,
            "get_coordinates": LocationTool.get_coordinates
        }
    
    def execute(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool by name"""
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            return self.tools[tool_name](**kwargs)
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {"error": str(e)}
    
    def get_available_tools(self) -> list:
        """Get list of available tool names"""
        return list(self.tools.keys())


# Create global registry instance
tool_registry = ToolRegistry()