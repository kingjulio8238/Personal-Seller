"""
Engagement Tracking Package for Social Darwin Gödel Machine
Provides metrics collection, conversion tracking, and reward calculation
"""

from .metrics_collector import MetricsCollector
from .conversion_tracker import ConversionTracker, ConversionData
from .reward_calculator import RewardCalculator

__all__ = [
    'MetricsCollector',
    'ConversionTracker',
    'ConversionData',
    'RewardCalculator'
]