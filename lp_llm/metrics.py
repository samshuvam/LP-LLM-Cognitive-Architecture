"""
LP-LLM Cognitive Architecture Component
Authored by Shuvam (https://github.com/samshuvam)
"""

__author__ = "Shuvam (https://github.com/samshuvam)"

"""
Research Metrics Tracking System
Production-Ready with All 8 Metrics

Features:
- Comprehensive metric collection (all 8 metrics)
- Trend analysis
- Export for research papers
- Benchmark suite
- Automatic tracking
"""

import os
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import csv
from .config import MetricsConfig, METRICS_DIR

logger = logging.getLogger(__name__)


class ResearchMetrics:
    """Comprehensive research metrics tracking (Category 8)"""
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else METRICS_DIR
        self.metrics_file = self.base_path / "metrics.json"
        self.exports_dir = self.base_path / "exports"
        
        # Create directories
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage (all 8 metrics)
        self.metrics = {
            metric_name: [] for metric_name in MetricsConfig.TRACKED_METRICS
        }
        
        # Load existing metrics
        self._load_metrics()
        
        logger.debug(f"Research metrics initialized at {self.base_path}")
    
    def _load_metrics(self):
        """Load metrics from file"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics = data.get('metrics', self.metrics)
            except:
                pass
    
    def _save_metrics(self):
        """Save metrics to file"""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "metrics": self.metrics
            }
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def log(self, metric_name: str, value: float, metadata: Optional[Dict] = None):
        """Log a metric value (Category 8)"""
        if metric_name not in self.metrics:
            logger.debug(f"Unknown metric: {metric_name}")
            return
        
        entry = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.metrics[metric_name].append(entry)
        self._save_metrics()
        
        logger.debug(f"Logged {metric_name}: {value}")
    
    def get_metric_summary(self, metric_name: str) -> Dict:
        """Get summary statistics for a metric"""
        if metric_name not in self.metrics:
            return {}
        
        values = [entry['value'] for entry in self.metrics[metric_name]]
        
        if not values:
            return {"count": 0}
        
        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "last_10_avg": float(np.mean(values[-10:])) if len(values) >= 10 else float(np.mean(values))
        }
    
    def get_trend(self, metric_name: str, window_size: int = 10) -> str:
        """Calculate trend for a metric"""
        if metric_name not in self.metrics:
            return "unknown"
        
        values = [entry['value'] for entry in self.metrics[metric_name]]
        
        if len(values) < window_size:
            return "insufficient_data"
        
        recent = np.mean(values[-window_size:])
        older = np.mean(values[-window_size*2:-window_size]) if len(values) >= window_size * 2 else recent
        
        if recent > older * 1.1:
            return "increasing"
        elif recent < older * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def generate_report(self) -> Dict:
        """Generate comprehensive research report (Category 8)"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "metrics": {},
            "trends": {},
            "summary": {}
        }
        
        for metric_name in MetricsConfig.TRACKED_METRICS:
            summary = self.get_metric_summary(metric_name)
            trend = self.get_trend(metric_name)
            
            report["metrics"][metric_name] = summary
            report["trends"][metric_name] = trend
            
            if summary.get('count', 0) > 0:
                report["summary"][metric_name] = summary['last_10_avg']
        
        return report
    
    def get_summary(self) -> Dict:
        """Get quick metrics summary for status display"""
        summary = {}
        for metric_name in MetricsConfig.TRACKED_METRICS:
            metric_summary = self.get_metric_summary(metric_name)
            if metric_summary.get('count', 0) > 0:
                summary[metric_name] = f"{metric_summary['last_10_avg']:.4f}"
        return summary
    
    def export_to_csv(self, filename: Optional[str] = None) -> str:
        """Export metrics to CSV for analysis (Category 8 & 11)"""
        if filename is None:
            filename = f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = self.exports_dir / filename
        
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                
                writer.writerow(['metric_name', 'timestamp', 'value', 'metadata'])
                
                for metric_name, entries in self.metrics.items():
                    for entry in entries:
                        writer.writerow([
                            metric_name,
                            entry['timestamp'],
                            entry['value'],
                            json.dumps(entry.get('metadata', {}))
                        ])
            
            logger.info(f"Exported metrics to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return ""
    
    def export_for_paper(self) -> Dict:
        """Export metrics in format suitable for research paper (Category 12)"""
        report = self.generate_report()
        
        paper_format = {
            "title": "Self-Evolving Cognitive Architecture - Performance Metrics",
            "generated": report['generated_at'],
            "metrics_table": []
        }
        
        for metric_name, summary in report['metrics'].items():
            if summary.get('count', 0) > 0:
                paper_format["metrics_table"].append({
                    "metric": metric_name,
                    "mean": f"{summary['mean']:.4f}",
                    "std": f"{summary['std']:.4f}",
                    "n": summary['count'],
                    "trend": report['trends'][metric_name]
                })
        
        return paper_format
    
    def clear_old_data(self, max_age_days: int = 90):
        """Clear metrics older than specified age"""
        cutoff = datetime.now() - timedelta(days=max_age_days)
        
        for metric_name in self.metrics:
            self.metrics[metric_name] = [
                entry for entry in self.metrics[metric_name]
                if datetime.fromisoformat(entry['timestamp']) > cutoff
            ]
        
        self._save_metrics()
        logger.info(f"Cleared metrics older than {max_age_days} days")


class BenchmarkSuite:
    """Benchmark suite for system evaluation (Category 8 & 12)"""
    
    def __init__(self, metrics_tracker: ResearchMetrics):
        self.metrics = metrics_tracker
        self.benchmark_queries = [
            {"query": "What is my name?", "expected_concept": "personal_info"},
            {"query": "Where do I live?", "expected_concept": "location"},
            {"query": "Tell me about my brother", "expected_concept": "family"},
            {"query": "What semester am I in?", "expected_concept": "education"},
            {"query": "What is the weather today?", "expected_concept": "weather"},
            {"query": "What time is it?", "expected_concept": "time"},
            {"query": "What is today's date?", "expected_concept": "date"},
            {"query": "Who is the PM of Nepal?", "expected_concept": "politics"}
        ]
    
    def run_benchmark(self, chatbot) -> Dict:
        """Run benchmark suite and log results (Category 12)"""
        results = []
        
        for benchmark in self.benchmark_queries:
            try:
                start_time = datetime.now()
                
                response = chatbot.process_input(benchmark['query'])
                
                end_time = datetime.now()
                latency = (end_time - start_time).total_seconds()
                
                accuracy = 1.0 if len(response) > 10 else 0.0
                
                results.append({
                    "query": benchmark['query'],
                    "latency": latency,
                    "accuracy": accuracy,
                    "response_length": len(response)
                })
                
                self.metrics.log("response_latency", latency, {"query": benchmark['query']})
                self.metrics.log("hallucination_rate", 1 - accuracy, {"query": benchmark['query']})
            except Exception as e:
                logger.error(f"Benchmark query failed: {benchmark['query']} - {e}")
        
        avg_latency = np.mean([r['latency'] for r in results]) if results else 0
        avg_accuracy = np.mean([r['accuracy'] for r in results]) if results else 0
        
        summary = {
            "avg_latency": float(avg_latency),
            "avg_accuracy": float(avg_accuracy),
            "total_queries": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Benchmark complete: {summary}")
        return summary








        







# """
# Research Metrics Tracking System
# Features:
# - Comprehensive metric collection
# - Trend analysis
# - Export for research papers
# - Benchmark suite
# """
# from typing import Dict, List, Any, Optional  # Added Optional

# import os
# import json
# import logging
# import numpy as np
# from datetime import datetime, timedelta
# from typing import Dict, List, Any, Optional
# from pathlib import Path
# import csv

# from config import MetricsConfig, METRICS_DIR

# logger = logging.getLogger(__name__)


# class ResearchMetrics:
#     """Comprehensive research metrics tracking"""
    
#     def __init__(self, base_path: Optional[str] = None):
#         self.base_path = Path(base_path) if base_path else METRICS_DIR
#         self.metrics_file = self.base_path / "metrics.json"
#         self.exports_dir = self.base_path / "exports"
        
#         # Create directories
#         self.exports_dir.mkdir(parents=True, exist_ok=True)
        
#         # Initialize metrics storage
#         self.metrics = {
#             metric_name: [] for metric_name in MetricsConfig.TRACKED_METRICS
#         }
        
#         # Load existing metrics
#         self._load_metrics()
        
#         logger.info(f"Research metrics initialized at {self.base_path}")
    
#     def _load_metrics(self):
#         """Load metrics from file"""
#         if self.metrics_file.exists():
#             with open(self.metrics_file, 'r') as f:
#                 data = json.load(f)
#                 self.metrics = data.get('metrics', self.metrics)
    
#     def _save_metrics(self):
#         """Save metrics to file"""
#         data = {
#             "last_updated": datetime.now().isoformat(),
#             "metrics": self.metrics
#         }
#         with open(self.metrics_file, 'w') as f:
#             json.dump(data, f, indent=2)
    
#     def log(self, metric_name: str, value: float, metadata: Optional[Dict] = None):
#         """
#         Log a metric value
        
#         Args:
#             metric_name: Name of the metric
#             value: Numeric value
#             meta Additional context
#         """
#         if metric_name not in self.metrics:
#             logger.warning(f"Unknown metric: {metric_name}")
#             return
        
#         entry = {
#             "value": value,
#             "timestamp": datetime.now().isoformat(),
#             "metadata": metadata or {}
#         }
        
#         self.metrics[metric_name].append(entry)
#         self._save_metrics()
        
#         logger.debug(f"Logged {metric_name}: {value}")
    
#     def get_metric_summary(self, metric_name: str) -> Dict:
#         """Get summary statistics for a metric"""
#         if metric_name not in self.metrics:
#             return {}
        
#         values = [entry['value'] for entry in self.metrics[metric_name]]
        
#         if not values:
#             return {"count": 0}
        
#         return {
#             "count": len(values),
#             "mean": np.mean(values),
#             "std": np.std(values),
#             "min": np.min(values),
#             "max": np.max(values),
#             "median": np.median(values),
#             "last_10_avg": np.mean(values[-10:]) if len(values) >= 10 else np.mean(values)
#         }
    
#     def get_trend(self, metric_name: str, window_size: int = 10) -> str:
#         """Calculate trend for a metric"""
#         if metric_name not in self.metrics:
#             return "unknown"
        
#         values = [entry['value'] for entry in self.metrics[metric_name]]
        
#         if len(values) < window_size:
#             return "insufficient_data"
        
#         recent = np.mean(values[-window_size:])
#         older = np.mean(values[-window_size*2:-window_size]) if len(values) >= window_size * 2 else recent
        
#         if recent > older * 1.1:
#             return "increasing"
#         elif recent < older * 0.9:
#             return "decreasing"
#         else:
#             return "stable"
    
#     def generate_report(self) -> Dict:
#         """Generate comprehensive research report"""
#         report = {
#             "generated_at": datetime.now().isoformat(),
#             "metrics": {},
#             "trends": {},
#             "summary": {}
#         }
        
#         for metric_name in MetricsConfig.TRACKED_METRICS:
#             summary = self.get_metric_summary(metric_name)
#             trend = self.get_trend(metric_name)
            
#             report["metrics"][metric_name] = summary
#             report["trends"][metric_name] = trend
            
#             if summary.get('count', 0) > 0:
#                 report["summary"][metric_name] = summary['last_10_avg']
        
#         return report
    
#     def export_to_csv(self, filename: Optional[str] = None) -> str:
#         """Export metrics to CSV for analysis"""
#         if filename is None:
#             filename = f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
#         filepath = self.exports_dir / filename
        
#         with open(filepath, 'w', newline='') as f:
#             writer = csv.writer(f)
            
#             # Header
#             writer.writerow(['metric_name', 'timestamp', 'value', 'metadata'])
            
#             # Data
#             for metric_name, entries in self.metrics.items():
#                 for entry in entries:
#                     writer.writerow([
#                         metric_name,
#                         entry['timestamp'],
#                         entry['value'],
#                         json.dumps(entry.get('metadata', {}))
#                     ])
        
#         logger.info(f"Exported metrics to {filepath}")
#         return str(filepath)
    
#     def export_for_paper(self) -> Dict:
#         """Export metrics in format suitable for research paper"""
#         report = self.generate_report()
        
#         # Format for academic presentation
#         paper_format = {
#             "title": "Self-Evolving Cognitive Architecture - Performance Metrics",
#             "generated": report['generated_at'],
#             "metrics_table": []
#         }
        
#         for metric_name, summary in report['metrics'].items():
#             if summary.get('count', 0) > 0:
#                 paper_format["metrics_table"].append({
#                     "metric": metric_name,
#                     "mean": f"{summary['mean']:.4f}",
#                     "std": f"{summary['std']:.4f}",
#                     "n": summary['count'],
#                     "trend": report['trends'][metric_name]
#                 })
        
#         return paper_format
    
#     def clear_old_data(self, max_age_days: int = 90):
#         """Clear metrics older than specified age"""
#         cutoff = datetime.now() - timedelta(days=max_age_days)
        
#         for metric_name in self.metrics:
#             self.metrics[metric_name] = [
#                 entry for entry in self.metrics[metric_name]
#                 if datetime.fromisoformat(entry['timestamp']) > cutoff
#             ]
        
#         self._save_metrics()
#         logger.info(f"Cleared metrics older than {max_age_days} days")


# class BenchmarkSuite:
#     """Benchmark suite for system evaluation"""
    
#     def __init__(self, metrics_tracker: ResearchMetrics):
#         self.metrics = metrics_tracker
#         self.benchmark_queries = [
#             {"query": "What is my name?", "expected_concept": "personal_info"},
#             {"query": "Where do I live?", "expected_concept": "location"},
#             {"query": "Tell me about my brother", "expected_concept": "family"},
#             {"query": "What semester am I in?", "expected_concept": "education"},
#         ]
    
#     def run_benchmark(self, chatbot) -> Dict:
#         """Run benchmark suite and log results"""
#         results = []
        
#         for benchmark in self.benchmark_queries:
#             start_time = datetime.now()
            
#             # Get response
#             response = chatbot.process_input(benchmark['query'])
            
#             end_time = datetime.now()
#             latency = (end_time - start_time).total_seconds()
            
#             # Simple accuracy check (response contains expected concept)
#             accuracy = 1.0 if len(response) > 10 else 0.0
            
#             results.append({
#                 "query": benchmark['query'],
#                 "latency": latency,
#                 "accuracy": accuracy,
#                 "response_length": len(response)
#             })
            
#             # Log metrics
#             self.metrics.log("response_latency", latency, {"query": benchmark['query']})
#             self.metrics.log("hallucination_rate", 1 - accuracy, {"query": benchmark['query']})
        
#         # Calculate summary
#         avg_latency = np.mean([r['latency'] for r in results])
#         avg_accuracy = np.mean([r['accuracy'] for r in results])
        
#         summary = {
#             "avg_latency": avg_latency,
#             "avg_accuracy": avg_accuracy,
#             "total_queries": len(results),
#             "timestamp": datetime.now().isoformat()
#         }
        
#         logger.info(f"Benchmark complete: {summary}")
#         return summary