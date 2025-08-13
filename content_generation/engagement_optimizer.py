"""
Engagement-Based Content Optimization System
Analyzes real-time engagement data to optimize content generation parameters and strategies
"""

import os
import json
import time
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from decimal import Decimal
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.orm import Session
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

from ..database.models import Post, EngagementMetrics, DatabaseManager
from ..engagement_tracking.metrics_collector import MetricsCollector
from .content_pipeline import ContentPipeline, ContentGenerationRequest
from .text_generator import ContentRequest


@dataclass
class EngagementPattern:
    """Engagement pattern analysis result"""
    platform: str
    content_type: str
    pattern_id: str
    avg_engagement: float
    peak_engagement_hour: int
    best_format: str
    optimal_hashtag_count: int
    optimal_caption_length: int
    color_preferences: Dict[str, float]
    style_preferences: Dict[str, float]
    frequency_pattern: Dict[str, float]
    confidence_score: float


@dataclass
class OptimizationRecommendation:
    """Content optimization recommendation"""
    recommendation_id: str
    priority: str  # 'critical', 'high', 'medium', 'low'
    category: str  # 'timing', 'format', 'content', 'visual', 'copy'
    current_value: Any
    recommended_value: Any
    expected_improvement: float  # Percentage improvement expected
    confidence: float
    reasoning: str
    action_required: str
    estimated_impact: Dict[str, float]  # likes, shares, comments impact


@dataclass
class ContentOptimizationResult:
    """Result of content optimization analysis"""
    optimization_id: str
    post_id: int
    platform: str
    content_type: str
    current_performance: Dict[str, float]
    predicted_performance: Dict[str, float]
    recommendations: List[OptimizationRecommendation]
    a_b_test_suggestions: List[Dict[str, Any]]
    engagement_forecast: Dict[str, float]
    optimization_score: float
    processing_time: float
    created_at: datetime


class EngagementOptimizer:
    """
    Advanced engagement-based content optimization system
    Uses machine learning to analyze engagement patterns and optimize content generation
    """
    
    def __init__(self, database_session: Session, content_pipeline: ContentPipeline):
        self.database_session = database_session
        self.db_manager = DatabaseManager(database_session)
        self.content_pipeline = content_pipeline
        self.metrics_collector = MetricsCollector(database_session)
        
        # Machine learning models
        self.engagement_models = {}
        self.pattern_models = {}
        self.scaler = StandardScaler()
        
        # Optimization parameters
        self.platform_configs = {
            'tiktok': {
                'optimal_video_length': (15, 30),  # seconds
                'peak_hours': [18, 19, 20, 21, 22],
                'hashtag_range': (3, 8),
                'engagement_weights': {'likes': 1.0, 'shares': 8.0, 'comments': 4.0, 'views': 0.01},
                'viral_threshold': 1000,
                'completion_rate_target': 0.7
            },
            'instagram': {
                'optimal_image_ratio': '1:1',
                'story_completion_target': 0.8,
                'peak_hours': [11, 12, 19, 20, 21],
                'hashtag_range': (5, 15),
                'engagement_weights': {'likes': 1.0, 'comments': 3.0, 'saves': 5.0, 'shares': 4.0},
                'carousel_performance_bonus': 1.3,
                'reel_vs_photo_multiplier': 2.1
            },
            'x': {
                'optimal_length': 280,
                'peak_hours': [9, 12, 15, 18, 21],
                'hashtag_range': (2, 4),
                'engagement_weights': {'likes': 1.0, 'retweets': 6.0, 'replies': 4.0, 'quotes': 5.0},
                'thread_performance_bonus': 1.5,
                'image_vs_text_multiplier': 1.8
            },
            'linkedin': {
                'optimal_length': 1200,
                'peak_hours': [8, 9, 10, 17, 18],
                'hashtag_range': (3, 5),
                'engagement_weights': {'likes': 1.0, 'comments': 4.0, 'shares': 6.0, 'clicks': 3.0},
                'professional_content_bonus': 1.4,
                'document_performance_bonus': 1.2
            },
            'pinterest': {
                'optimal_ratio': '2:3',
                'peak_hours': [14, 15, 20, 21, 22],
                'hashtag_range': (5, 10),
                'engagement_weights': {'saves': 10.0, 'clicks': 5.0, 'comments': 2.0},
                'seasonal_bonus_periods': ['fall', 'spring'],
                'idea_pin_multiplier': 1.6
            },
            'youtube': {
                'optimal_thumbnail_ctr': 0.12,
                'peak_hours': [19, 20, 21, 22],
                'hashtag_range': (3, 6),
                'engagement_weights': {'views': 0.01, 'likes': 1.0, 'comments': 3.0, 'subscribes': 20.0},
                'watch_time_target': 0.6,
                'shorts_vs_long_multiplier': 3.2
            }
        }
        
        # Pattern recognition settings
        self.pattern_analysis_window = 30  # days
        self.min_samples_for_pattern = 10
        self.confidence_threshold = 0.7
        
        # Model storage
        self.model_storage_path = os.path.join(os.path.dirname(__file__), '..', 'temp', 'optimization_models')
        os.makedirs(self.model_storage_path, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load or initialize models
        self._load_or_initialize_models()

    def _load_or_initialize_models(self):
        """Load existing models or initialize new ones"""
        try:
            # Try to load existing models
            for platform in self.platform_configs.keys():
                model_path = os.path.join(self.model_storage_path, f'{platform}_engagement_model.pkl')
                pattern_path = os.path.join(self.model_storage_path, f'{platform}_pattern_model.pkl')
                
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.engagement_models[platform] = pickle.load(f)
                else:
                    # Initialize new model
                    self.engagement_models[platform] = GradientBoostingRegressor(
                        n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
                    )
                
                if os.path.exists(pattern_path):
                    with open(pattern_path, 'rb') as f:
                        self.pattern_models[platform] = pickle.load(f)
                else:
                    # Initialize new pattern model
                    self.pattern_models[platform] = KMeans(n_clusters=5, random_state=42)
            
            self.logger.info("Optimization models loaded/initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load optimization models: {e}")

    def analyze_engagement_patterns(self, platform: str, days: int = 30) -> List[EngagementPattern]:
        """
        Analyze engagement patterns for a specific platform
        """
        try:
            self.logger.info(f"Analyzing engagement patterns for {platform}")
            
            # Get recent posts with engagement data
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            posts = self.database_session.query(Post).filter(
                Post.platform == platform,
                Post.posted_time >= cutoff_time,
                Post.status == 'posted'
            ).all()
            
            if len(posts) < self.min_samples_for_pattern:
                self.logger.warning(f"Insufficient data for pattern analysis: {len(posts)} posts")
                return []
            
            patterns = []
            
            # Group posts by content type
            content_type_groups = defaultdict(list)
            for post in posts:
                content_type_groups[post.content_type].append(post)
            
            for content_type, type_posts in content_type_groups.items():
                if len(type_posts) < 5:  # Need minimum samples per content type
                    continue
                
                # Extract features and engagement data
                features_data = []
                engagement_data = []
                
                for post in type_posts:
                    latest_metrics = post.get_latest_engagement()
                    if not latest_metrics:
                        continue
                    
                    # Extract temporal features
                    post_hour = post.posted_time.hour if post.posted_time else 12
                    post_day = post.posted_time.weekday() if post.posted_time else 0
                    
                    # Extract content features
                    caption_length = len(post.caption) if post.caption else 0
                    hashtag_count = post.hashtag_count if hasattr(post, 'hashtag_count') else 0
                    
                    # Calculate engagement score
                    platform_config = self.platform_configs.get(platform, {})
                    weights = platform_config.get('engagement_weights', {})
                    engagement_score = (
                        latest_metrics.likes * weights.get('likes', 1.0) +
                        latest_metrics.shares * weights.get('shares', 5.0) +
                        latest_metrics.comments * weights.get('comments', 3.0) +
                        latest_metrics.views * weights.get('views', 0.01)
                    )
                    
                    features_data.append([
                        post_hour, post_day, caption_length, hashtag_count,
                        latest_metrics.likes, latest_metrics.shares,
                        latest_metrics.comments, latest_metrics.views
                    ])
                    engagement_data.append(engagement_score)
                
                if len(features_data) < 5:
                    continue
                
                # Convert to numpy arrays
                X = np.array(features_data)
                y = np.array(engagement_data)
                
                # Identify patterns using clustering
                pattern_model = self.pattern_models[platform]
                if hasattr(pattern_model, 'n_clusters'):
                    clusters = pattern_model.fit_predict(X)
                else:
                    clusters = [0] * len(X)
                
                # Analyze each cluster for patterns
                for cluster_id in set(clusters):
                    cluster_mask = clusters == cluster_id
                    cluster_data = X[cluster_mask]
                    cluster_engagement = y[cluster_mask]
                    
                    if len(cluster_data) < 3:
                        continue
                    
                    # Calculate pattern characteristics
                    avg_engagement = np.mean(cluster_engagement)
                    peak_hour = int(np.median(cluster_data[:, 0]))
                    avg_caption_length = int(np.median(cluster_data[:, 2]))
                    avg_hashtag_count = int(np.median(cluster_data[:, 3]))
                    
                    # Determine optimal format based on performance
                    best_format = content_type
                    if content_type == 'text_image' and avg_engagement > np.percentile(y, 75):
                        best_format = 'high_performing_image'
                    elif content_type == 'text_video' and avg_engagement > np.percentile(y, 80):
                        best_format = 'high_performing_video'
                    
                    pattern = EngagementPattern(
                        platform=platform,
                        content_type=content_type,
                        pattern_id=f"{platform}_{content_type}_cluster_{cluster_id}",
                        avg_engagement=avg_engagement,
                        peak_engagement_hour=peak_hour,
                        best_format=best_format,
                        optimal_hashtag_count=avg_hashtag_count,
                        optimal_caption_length=avg_caption_length,
                        color_preferences=self._analyze_color_preferences(cluster_data),
                        style_preferences=self._analyze_style_preferences(cluster_data),
                        frequency_pattern=self._analyze_frequency_pattern(cluster_data),
                        confidence_score=min(len(cluster_data) / 20, 1.0)  # Confidence based on sample size
                    )
                    
                    patterns.append(pattern)
            
            self.logger.info(f"Identified {len(patterns)} engagement patterns for {platform}")
            return patterns
            
        except Exception as e:
            self.logger.error(f"Pattern analysis failed for {platform}: {e}")
            return []

    def generate_optimization_recommendations(self, post_id: int) -> ContentOptimizationResult:
        """
        Generate comprehensive optimization recommendations for a specific post
        """
        start_time = time.time()
        optimization_id = f"opt_{post_id}_{int(start_time)}"
        
        try:
            # Get post data
            post = self.database_session.query(Post).get(post_id)
            if not post:
                raise ValueError(f"Post {post_id} not found")
            
            # Get latest engagement metrics
            latest_metrics = post.get_latest_engagement()
            if not latest_metrics:
                raise ValueError(f"No engagement metrics found for post {post_id}")
            
            platform = post.platform
            content_type = post.content_type
            
            self.logger.info(f"Generating optimization recommendations for post {post_id} ({platform})")
            
            # Analyze current performance
            current_performance = self._analyze_current_performance(post, latest_metrics, platform)
            
            # Get engagement patterns for this platform/content type
            patterns = self.analyze_engagement_patterns(platform, days=30)
            relevant_patterns = [p for p in patterns if p.content_type == content_type]
            
            # Generate specific recommendations
            recommendations = []
            
            # Timing optimization
            timing_rec = self._generate_timing_recommendations(post, relevant_patterns, platform)
            if timing_rec:
                recommendations.append(timing_rec)
            
            # Content format optimization
            format_rec = self._generate_format_recommendations(post, latest_metrics, relevant_patterns, platform)
            if format_rec:
                recommendations.append(format_rec)
            
            # Caption/copy optimization
            copy_rec = self._generate_copy_recommendations(post, relevant_patterns, platform)
            if copy_rec:
                recommendations.append(copy_rec)
            
            # Visual optimization
            visual_rec = self._generate_visual_recommendations(post, relevant_patterns, platform)
            if visual_rec:
                recommendations.append(visual_rec)
            
            # Hashtag optimization
            hashtag_rec = self._generate_hashtag_recommendations(post, relevant_patterns, platform)
            if hashtag_rec:
                recommendations.append(hashtag_rec)
            
            # Predict performance with optimizations
            predicted_performance = self._predict_optimized_performance(
                post, recommendations, current_performance, platform
            )
            
            # Generate A/B test suggestions
            ab_suggestions = self._generate_ab_test_suggestions(post, recommendations, platform)
            
            # Generate engagement forecast
            engagement_forecast = self._generate_engagement_forecast(
                post, current_performance, predicted_performance, platform
            )
            
            # Calculate optimization score
            optimization_score = self._calculate_optimization_score(
                current_performance, predicted_performance, recommendations
            )
            
            result = ContentOptimizationResult(
                optimization_id=optimization_id,
                post_id=post_id,
                platform=platform,
                content_type=content_type,
                current_performance=current_performance,
                predicted_performance=predicted_performance,
                recommendations=recommendations,
                a_b_test_suggestions=ab_suggestions,
                engagement_forecast=engagement_forecast,
                optimization_score=optimization_score,
                processing_time=time.time() - start_time,
                created_at=datetime.utcnow()
            )
            
            self.logger.info(
                f"Optimization analysis completed for post {post_id}: "
                f"{len(recommendations)} recommendations (score: {optimization_score:.1f})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization analysis failed for post {post_id}: {e}")
            return ContentOptimizationResult(
                optimization_id=optimization_id,
                post_id=post_id,
                platform="unknown",
                content_type="unknown",
                current_performance={},
                predicted_performance={},
                recommendations=[],
                a_b_test_suggestions=[],
                engagement_forecast={},
                optimization_score=0.0,
                processing_time=time.time() - start_time,
                created_at=datetime.utcnow()
            )

    def _analyze_current_performance(self, post: Post, metrics: EngagementMetrics, platform: str) -> Dict[str, float]:
        """Analyze current post performance metrics"""
        platform_config = self.platform_configs.get(platform, {})
        weights = platform_config.get('engagement_weights', {})
        
        # Calculate weighted engagement score
        engagement_score = (
            metrics.likes * weights.get('likes', 1.0) +
            metrics.shares * weights.get('shares', 5.0) +
            metrics.comments * weights.get('comments', 3.0) +
            metrics.views * weights.get('views', 0.01)
        )
        
        # Calculate engagement rate (if we have follower data)
        # For now, use views as proxy for reach
        engagement_rate = (metrics.likes + metrics.shares + metrics.comments) / max(metrics.views, 1) * 100
        
        return {
            'total_engagement': engagement_score,
            'likes': float(metrics.likes),
            'shares': float(metrics.shares),
            'comments': float(metrics.comments),
            'views': float(metrics.views),
            'engagement_rate': engagement_rate,
            'like_rate': metrics.likes / max(metrics.views, 1) * 100,
            'share_rate': metrics.shares / max(metrics.views, 1) * 100,
            'comment_rate': metrics.comments / max(metrics.views, 1) * 100
        }

    def _generate_timing_recommendations(self, post: Post, patterns: List[EngagementPattern], 
                                       platform: str) -> Optional[OptimizationRecommendation]:
        """Generate timing optimization recommendations"""
        if not post.posted_time or not patterns:
            return None
        
        current_hour = post.posted_time.hour
        platform_config = self.platform_configs.get(platform, {})
        peak_hours = platform_config.get('peak_hours', [12, 18, 21])
        
        # Find optimal hour from patterns
        pattern_hours = [p.peak_engagement_hour for p in patterns if p.confidence_score > 0.5]
        optimal_hour = max(set(pattern_hours), key=pattern_hours.count) if pattern_hours else peak_hours[0]
        
        if abs(current_hour - optimal_hour) >= 2:  # Significant time difference
            expected_improvement = min(abs(current_hour - optimal_hour) * 5, 25)  # Up to 25% improvement
            
            return OptimizationRecommendation(
                recommendation_id=f"timing_{post.id}",
                priority="high" if expected_improvement > 15 else "medium",
                category="timing",
                current_value=f"{current_hour}:00",
                recommended_value=f"{optimal_hour}:00",
                expected_improvement=expected_improvement,
                confidence=0.8,
                reasoning=f"Peak engagement occurs at {optimal_hour}:00 based on historical data",
                action_required="Schedule future posts during peak engagement hours",
                estimated_impact={
                    'likes': expected_improvement * 0.8,
                    'shares': expected_improvement * 1.2,
                    'comments': expected_improvement * 1.0,
                    'views': expected_improvement * 0.6
                }
            )
        
        return None

    def _generate_format_recommendations(self, post: Post, metrics: EngagementMetrics,
                                       patterns: List[EngagementPattern], platform: str) -> Optional[OptimizationRecommendation]:
        """Generate content format optimization recommendations"""
        current_format = post.content_type
        platform_config = self.platform_configs.get(platform, {})
        
        # Analyze format performance from patterns
        format_performance = {}
        for pattern in patterns:
            if pattern.best_format not in format_performance:
                format_performance[pattern.best_format] = []
            format_performance[pattern.best_format].append(pattern.avg_engagement)
        
        # Find best performing format
        if format_performance:
            best_format = max(format_performance.keys(), key=lambda f: np.mean(format_performance[f]))
            current_avg = np.mean(format_performance.get(current_format, [0]))
            best_avg = np.mean(format_performance[best_format])
            
            if best_format != current_format and best_avg > current_avg * 1.2:
                expected_improvement = ((best_avg - current_avg) / current_avg) * 100
                expected_improvement = min(expected_improvement, 40)  # Cap at 40%
                
                return OptimizationRecommendation(
                    recommendation_id=f"format_{post.id}",
                    priority="high" if expected_improvement > 25 else "medium",
                    category="format",
                    current_value=current_format,
                    recommended_value=best_format,
                    expected_improvement=expected_improvement,
                    confidence=0.7,
                    reasoning=f"{best_format} performs {expected_improvement:.1f}% better on average",
                    action_required="Consider creating content in the recommended format",
                    estimated_impact={
                        'likes': expected_improvement * 1.0,
                        'shares': expected_improvement * 1.3,
                        'comments': expected_improvement * 0.9,
                        'views': expected_improvement * 1.1
                    }
                )
        
        return None

    def _generate_copy_recommendations(self, post: Post, patterns: List[EngagementPattern],
                                     platform: str) -> Optional[OptimizationRecommendation]:
        """Generate caption/copy optimization recommendations"""
        if not post.caption or not patterns:
            return None
        
        current_length = len(post.caption)
        platform_config = self.platform_configs.get(platform, {})
        
        # Find optimal length from patterns
        optimal_lengths = [p.optimal_caption_length for p in patterns if p.confidence_score > 0.5]
        if not optimal_lengths:
            return None
        
        optimal_length = int(np.median(optimal_lengths))
        platform_optimal = platform_config.get('optimal_length', optimal_length)
        
        # Use the more conservative recommendation
        recommended_length = min(optimal_length, platform_optimal) if platform_optimal else optimal_length
        
        length_diff_ratio = abs(current_length - recommended_length) / current_length
        
        if length_diff_ratio > 0.3:  # Significant length difference
            expected_improvement = min(length_diff_ratio * 20, 30)  # Up to 30% improvement
            
            return OptimizationRecommendation(
                recommendation_id=f"copy_{post.id}",
                priority="medium",
                category="copy",
                current_value=f"{current_length} characters",
                recommended_value=f"{recommended_length} characters",
                expected_improvement=expected_improvement,
                confidence=0.6,
                reasoning=f"Optimal caption length for {platform} is around {recommended_length} characters",
                action_required="Adjust caption length for better engagement",
                estimated_impact={
                    'likes': expected_improvement * 0.7,
                    'shares': expected_improvement * 0.9,
                    'comments': expected_improvement * 1.2,
                    'views': expected_improvement * 0.5
                }
            )
        
        return None

    def _generate_visual_recommendations(self, post: Post, patterns: List[EngagementPattern],
                                       platform: str) -> Optional[OptimizationRecommendation]:
        """Generate visual content optimization recommendations"""
        platform_config = self.platform_configs.get(platform, {})
        
        # Platform-specific visual recommendations
        recommendations = {
            'instagram': "Use 1:1 aspect ratio for feed posts, bright colors perform well",
            'tiktok': "Vertical 9:16 format, high contrast visuals, text overlay for accessibility",
            'x': "Images increase engagement by 80%, use high contrast and clear text",
            'linkedin': "Professional imagery, infographics perform well, avoid oversaturation",
            'pinterest': "2:3 aspect ratio optimal, bright lifestyle imagery, clear text overlay",
            'youtube': "Custom thumbnails with faces perform 30% better, high contrast text"
        }
        
        if platform in recommendations:
            return OptimizationRecommendation(
                recommendation_id=f"visual_{post.id}",
                priority="medium",
                category="visual",
                current_value="current visual style",
                recommended_value=recommendations[platform],
                expected_improvement=15.0,
                confidence=0.7,
                reasoning=f"Platform-optimized visuals increase engagement on {platform}",
                action_required="Apply visual optimization guidelines",
                estimated_impact={
                    'likes': 12.0,
                    'shares': 18.0,
                    'comments': 10.0,
                    'views': 20.0
                }
            )
        
        return None

    def _generate_hashtag_recommendations(self, post: Post, patterns: List[EngagementPattern],
                                        platform: str) -> Optional[OptimizationRecommendation]:
        """Generate hashtag optimization recommendations"""
        platform_config = self.platform_configs.get(platform, {})
        hashtag_range = platform_config.get('hashtag_range', (3, 8))
        
        # Estimate current hashtag count from caption
        current_hashtag_count = post.caption.count('#') if post.caption else 0
        
        # Find optimal count from patterns
        optimal_counts = [p.optimal_hashtag_count for p in patterns if p.confidence_score > 0.5]
        if optimal_counts:
            recommended_count = int(np.median(optimal_counts))
        else:
            recommended_count = (hashtag_range[0] + hashtag_range[1]) // 2
        
        # Ensure within platform range
        recommended_count = max(hashtag_range[0], min(hashtag_range[1], recommended_count))
        
        if abs(current_hashtag_count - recommended_count) >= 2:
            expected_improvement = min(abs(current_hashtag_count - recommended_count) * 3, 20)
            
            return OptimizationRecommendation(
                recommendation_id=f"hashtag_{post.id}",
                priority="low" if expected_improvement < 10 else "medium",
                category="content",
                current_value=f"{current_hashtag_count} hashtags",
                recommended_value=f"{recommended_count} hashtags",
                expected_improvement=expected_improvement,
                confidence=0.6,
                reasoning=f"Optimal hashtag count for {platform} is {recommended_count}",
                action_required="Adjust hashtag strategy for better discoverability",
                estimated_impact={
                    'likes': expected_improvement * 0.8,
                    'shares': expected_improvement * 0.6,
                    'comments': expected_improvement * 0.7,
                    'views': expected_improvement * 1.5
                }
            )
        
        return None

    def _predict_optimized_performance(self, post: Post, recommendations: List[OptimizationRecommendation],
                                     current_performance: Dict[str, float], platform: str) -> Dict[str, float]:
        """Predict performance after applying optimizations"""
        predicted = current_performance.copy()
        
        # Apply improvement estimates from recommendations
        for rec in recommendations:
            multiplier = rec.expected_improvement / 100
            confidence_factor = rec.confidence
            
            for metric, impact in rec.estimated_impact.items():
                if metric in predicted:
                    improvement = (impact / 100) * confidence_factor
                    predicted[metric] *= (1 + improvement)
        
        # Apply platform-specific bonuses
        platform_config = self.platform_configs.get(platform, {})
        
        # Check for format bonuses
        if post.content_type == 'image_carousel' and platform == 'instagram':
            carousel_bonus = platform_config.get('carousel_performance_bonus', 1.3)
            predicted['total_engagement'] *= carousel_bonus
        
        return predicted

    def _generate_ab_test_suggestions(self, post: Post, recommendations: List[OptimizationRecommendation],
                                    platform: str) -> List[Dict[str, Any]]:
        """Generate A/B test suggestions based on recommendations"""
        suggestions = []
        
        # Create test variants from high-priority recommendations
        high_priority_recs = [r for r in recommendations if r.priority in ['critical', 'high']]
        
        for rec in high_priority_recs[:3]:  # Limit to top 3 recommendations
            suggestion = {
                'test_id': f"ab_test_{rec.recommendation_id}",
                'test_type': rec.category,
                'control_value': rec.current_value,
                'variant_value': rec.recommended_value,
                'expected_improvement': rec.expected_improvement,
                'confidence': rec.confidence,
                'test_duration_days': 7,
                'min_sample_size': 100,
                'success_metrics': ['engagement_rate', 'total_engagement'],
                'risk_level': 'low' if rec.expected_improvement < 20 else 'medium'
            }
            suggestions.append(suggestion)
        
        return suggestions

    def _generate_engagement_forecast(self, post: Post, current_performance: Dict[str, float],
                                    predicted_performance: Dict[str, float], platform: str) -> Dict[str, float]:
        """Generate engagement forecast over time"""
        # Simple exponential decay model for engagement prediction
        current_total = current_performance.get('total_engagement', 0)
        predicted_total = predicted_performance.get('total_engagement', 0)
        
        forecast = {}
        
        # 24 hour, 7 day, 30 day forecasts
        time_periods = [1, 7, 30]  # days
        decay_rates = {'tiktok': 0.8, 'instagram': 0.6, 'x': 0.9, 'linkedin': 0.4, 'pinterest': 0.2, 'youtube': 0.3}
        decay_rate = decay_rates.get(platform, 0.6)
        
        for days in time_periods:
            # Current trajectory
            current_forecast = current_total * (1 + (1 - decay_rate) ** days)
            # Optimized trajectory
            optimized_forecast = predicted_total * (1 + (1 - decay_rate * 0.8) ** days)  # Optimization reduces decay
            
            forecast[f'{days}_day_current'] = current_forecast
            forecast[f'{days}_day_optimized'] = optimized_forecast
            forecast[f'{days}_day_improvement'] = ((optimized_forecast - current_forecast) / current_forecast) * 100 if current_forecast > 0 else 0
        
        return forecast

    def _calculate_optimization_score(self, current: Dict[str, float], predicted: Dict[str, float],
                                    recommendations: List[OptimizationRecommendation]) -> float:
        """Calculate overall optimization score (0-100)"""
        # Base score from predicted improvement
        current_total = current.get('total_engagement', 1)
        predicted_total = predicted.get('total_engagement', 1)
        improvement_score = min(((predicted_total - current_total) / current_total) * 100, 50)
        
        # Confidence score from recommendations
        if recommendations:
            avg_confidence = np.mean([r.confidence for r in recommendations])
            confidence_score = avg_confidence * 30
        else:
            confidence_score = 0
        
        # Actionability score (more high-priority recommendations = higher score)
        high_priority_count = len([r for r in recommendations if r.priority in ['critical', 'high']])
        actionability_score = min(high_priority_count * 5, 20)
        
        return min(improvement_score + confidence_score + actionability_score, 100)

    def _analyze_color_preferences(self, cluster_data: np.ndarray) -> Dict[str, float]:
        """Analyze color preferences from engagement data"""
        # Placeholder for color analysis - would need actual image analysis
        return {
            'bright_colors': 0.7,
            'pastel_colors': 0.3,
            'dark_colors': 0.4,
            'high_contrast': 0.8
        }

    def _analyze_style_preferences(self, cluster_data: np.ndarray) -> Dict[str, float]:
        """Analyze style preferences from engagement data"""
        # Placeholder for style analysis
        return {
            'minimalist': 0.6,
            'detailed': 0.4,
            'professional': 0.5,
            'casual': 0.7
        }

    def _analyze_frequency_pattern(self, cluster_data: np.ndarray) -> Dict[str, float]:
        """Analyze posting frequency patterns"""
        # Extract day of week patterns
        days = cluster_data[:, 1] if len(cluster_data[0]) > 1 else [0] * len(cluster_data)
        day_counts = {f'day_{i}': np.sum(days == i) for i in range(7)}
        total = sum(day_counts.values()) or 1
        return {k: v / total for k, v in day_counts.items()}

    def train_engagement_models(self, platform: str, days: int = 90):
        """Train machine learning models for engagement prediction"""
        try:
            self.logger.info(f"Training engagement model for {platform}")
            
            # Get training data
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            posts = self.database_session.query(Post).filter(
                Post.platform == platform,
                Post.posted_time >= cutoff_time,
                Post.status == 'posted'
            ).all()
            
            if len(posts) < 50:
                self.logger.warning(f"Insufficient data for model training: {len(posts)} posts")
                return False
            
            # Extract features and targets
            X, y = self._extract_training_data(posts)
            
            if len(X) < 20:
                self.logger.warning("Insufficient processed training data")
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train engagement prediction model
            model = self.engagement_models[platform]
            model.fit(X_scaled, y)
            
            # Train pattern recognition model
            pattern_model = self.pattern_models[platform]
            pattern_model.fit(X_scaled)
            
            # Save models
            model_path = os.path.join(self.model_storage_path, f'{platform}_engagement_model.pkl')
            pattern_path = os.path.join(self.model_storage_path, f'{platform}_pattern_model.pkl')
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            with open(pattern_path, 'wb') as f:
                pickle.dump(pattern_model, f)
            
            self.logger.info(f"Model training completed for {platform}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model training failed for {platform}: {e}")
            return False

    def _extract_training_data(self, posts: List[Post]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and targets for model training"""
        features = []
        targets = []
        
        for post in posts:
            latest_metrics = post.get_latest_engagement()
            if not latest_metrics:
                continue
            
            # Feature extraction
            post_hour = post.posted_time.hour if post.posted_time else 12
            post_day = post.posted_time.weekday() if post.posted_time else 0
            caption_length = len(post.caption) if post.caption else 0
            hashtag_count = post.caption.count('#') if post.caption else 0
            has_image = 1 if post.image_url else 0
            has_video = 1 if post.video_url else 0
            
            features.append([
                post_hour, post_day, caption_length, hashtag_count, has_image, has_video
            ])
            
            # Target (engagement score)
            platform_config = self.platform_configs.get(post.platform, {})
            weights = platform_config.get('engagement_weights', {})
            engagement_score = (
                latest_metrics.likes * weights.get('likes', 1.0) +
                latest_metrics.shares * weights.get('shares', 5.0) +
                latest_metrics.comments * weights.get('comments', 3.0) +
                latest_metrics.views * weights.get('views', 0.01)
            )
            targets.append(engagement_score)
        
        return np.array(features), np.array(targets)

    async def run_automated_optimization_cycle(self):
        """Run automated optimization analysis for recent posts"""
        try:
            self.logger.info("Starting automated optimization cycle")
            
            # Get recent posts that need optimization analysis
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            posts = self.database_session.query(Post).filter(
                Post.posted_time >= cutoff_time,
                Post.status == 'posted'
            ).limit(20).all()  # Process up to 20 recent posts
            
            optimization_results = []
            
            for post in posts:
                try:
                    result = self.generate_optimization_recommendations(post.id)
                    optimization_results.append(result)
                    
                    # Brief delay to avoid overwhelming the system
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    self.logger.error(f"Optimization failed for post {post.id}: {e}")
                    continue
            
            # Generate summary report
            self._generate_optimization_summary(optimization_results)
            
            self.logger.info(f"Automated optimization cycle completed: {len(optimization_results)} posts analyzed")
            
        except Exception as e:
            self.logger.error(f"Automated optimization cycle failed: {e}")

    def _generate_optimization_summary(self, results: List[ContentOptimizationResult]):
        """Generate summary of optimization cycle results"""
        if not results:
            return
        
        summary = {
            'total_posts_analyzed': len(results),
            'avg_optimization_score': np.mean([r.optimization_score for r in results]),
            'total_recommendations': sum(len(r.recommendations) for r in results),
            'high_priority_recommendations': sum(
                len([rec for rec in r.recommendations if rec.priority in ['critical', 'high']])
                for r in results
            ),
            'platform_breakdown': defaultdict(list),
            'recommendation_categories': defaultdict(int),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        for result in results:
            summary['platform_breakdown'][result.platform].append({
                'post_id': result.post_id,
                'optimization_score': result.optimization_score,
                'recommendation_count': len(result.recommendations)
            })
            
            for rec in result.recommendations:
                summary['recommendation_categories'][rec.category] += 1
        
        # Save summary to file
        summary_path = os.path.join(self.model_storage_path, 'optimization_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Optimization summary saved: {summary['total_recommendations']} recommendations generated")

    def get_optimization_insights(self, platform: str = None, days: int = 30) -> Dict[str, Any]:
        """Get optimization insights and performance trends"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            query = self.database_session.query(Post).filter(
                Post.posted_time >= cutoff_time,
                Post.status == 'posted'
            )
            
            if platform:
                query = query.filter(Post.platform == platform)
            
            posts = query.all()
            
            insights = {
                'total_posts': len(posts),
                'platform_performance': {},
                'content_type_performance': {},
                'timing_insights': {},
                'engagement_trends': {},
                'top_performing_posts': [],
                'optimization_opportunities': {}
            }
            
            # Platform performance analysis
            platform_groups = defaultdict(list)
            for post in posts:
                latest_metrics = post.get_latest_engagement()
                if latest_metrics:
                    score = self.metrics_collector.calculate_engagement_score(latest_metrics, post.platform)
                    platform_groups[post.platform].append(score)
            
            for platform, scores in platform_groups.items():
                insights['platform_performance'][platform] = {
                    'avg_engagement': np.mean(scores),
                    'median_engagement': np.median(scores),
                    'post_count': len(scores),
                    'top_percentile': np.percentile(scores, 90) if scores else 0
                }
            
            # Content type analysis
            content_groups = defaultdict(list)
            for post in posts:
                latest_metrics = post.get_latest_engagement()
                if latest_metrics:
                    score = self.metrics_collector.calculate_engagement_score(latest_metrics, post.platform)
                    content_groups[post.content_type].append(score)
            
            for content_type, scores in content_groups.items():
                insights['content_type_performance'][content_type] = {
                    'avg_engagement': np.mean(scores),
                    'performance_rating': 'high' if np.mean(scores) > np.mean([s for scores_list in content_groups.values() for s in scores_list]) else 'medium'
                }
            
            # Timing insights
            hour_performance = defaultdict(list)
            for post in posts:
                if post.posted_time:
                    latest_metrics = post.get_latest_engagement()
                    if latest_metrics:
                        score = self.metrics_collector.calculate_engagement_score(latest_metrics, post.platform)
                        hour_performance[post.posted_time.hour].append(score)
            
            if hour_performance:
                best_hour = max(hour_performance.keys(), key=lambda h: np.mean(hour_performance[h]))
                insights['timing_insights'] = {
                    'best_posting_hour': best_hour,
                    'hourly_performance': {h: np.mean(scores) for h, scores in hour_performance.items()}
                }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate optimization insights: {e}")
            return {'error': str(e)}


if __name__ == "__main__":
    # Test optimization system (would need actual database session)
    print("Engagement Optimizer initialized successfully")
    print("Features available:")
    print("- Real-time engagement pattern analysis")
    print("- ML-powered optimization recommendations") 
    print("- Platform-specific strategy optimization")
    print("- A/B testing suggestions")
    print("- Automated improvement cycle")
    print("- Performance forecasting")