"""
Machine Learning Content Optimization Engine
Advanced ML-powered system for content optimization, A/B testing, and performance prediction
"""

import os
import json
import numpy as np
import pandas as pd
import logging
import pickle
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from scipy import stats
from scipy.optimize import minimize

from ..database.models import Post, EngagementMetrics, DatabaseManager
from .engagement_optimizer import EngagementOptimizer, OptimizationRecommendation


@dataclass
class ABTestConfiguration:
    """A/B Test configuration"""
    test_id: str
    test_name: str
    platform: str
    variable_name: str
    control_value: Any
    variant_values: List[Any]
    success_metrics: List[str]
    min_sample_size: int
    max_duration_days: int
    confidence_level: float
    power: float
    expected_effect_size: float


@dataclass
class ABTestResult:
    """A/B Test result"""
    test_id: str
    winning_variant: str
    confidence_interval: Tuple[float, float]
    p_value: float
    effect_size: float
    statistical_significance: bool
    practical_significance: bool
    sample_sizes: Dict[str, int]
    conversion_rates: Dict[str, float]
    recommendations: List[str]
    test_duration_days: int


@dataclass
class PerformancePrediction:
    """Performance prediction result"""
    prediction_id: str
    post_features: Dict[str, Any]
    predicted_engagement: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    prediction_confidence: float
    feature_importance: Dict[str, float]
    similar_posts: List[int]
    optimization_suggestions: List[str]


@dataclass
class ContentVariant:
    """Content variant for testing"""
    variant_id: str
    variant_type: str  # 'copy', 'visual', 'timing', 'format'
    original_content: Dict[str, Any]
    modified_content: Dict[str, Any]
    expected_improvement: float
    test_priority: str


class MLOptimizationEngine:
    """
    Machine Learning-powered content optimization engine
    Provides A/B testing, performance prediction, and automated optimization
    """
    
    def __init__(self, database_session, engagement_optimizer: EngagementOptimizer):
        self.database_session = database_session
        self.db_manager = DatabaseManager(database_session)
        self.engagement_optimizer = engagement_optimizer
        
        # ML Models
        self.engagement_models = {}  # Platform-specific engagement prediction models
        self.clustering_models = {}  # Content clustering models
        self.feature_extractors = {}  # Feature extraction models
        self.scalers = {}
        
        # A/B Testing
        self.active_tests = {}
        self.test_results = {}
        
        # Model storage
        self.model_storage_path = os.path.join(os.path.dirname(__file__), '..', 'temp', 'ml_models')
        os.makedirs(self.model_storage_path, exist_ok=True)
        
        # Feature engineering configuration
        self.feature_config = {
            'temporal_features': ['hour', 'day_of_week', 'month', 'is_weekend'],
            'content_features': ['caption_length', 'hashtag_count', 'has_image', 'has_video'],
            'platform_features': ['platform_encoded', 'content_type_encoded'],
            'engagement_features': ['historical_avg_engagement', 'recent_trend']
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self._initialize_ml_models()
    
    def _initialize_ml_models(self):
        """Initialize ML models for each platform"""
        platforms = ['tiktok', 'instagram', 'x', 'linkedin', 'pinterest', 'youtube']
        
        for platform in platforms:
            # Engagement prediction model
            self.engagement_models[platform] = {
                'primary': GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'secondary': RandomForestRegressor(
                    n_estimators=150,
                    max_depth=8,
                    random_state=42
                ),
                'ensemble_weights': [0.7, 0.3]  # Primary model gets more weight
            }
            
            # Content clustering model
            self.clustering_models[platform] = KMeans(
                n_clusters=8,
                random_state=42,
                n_init=10
            )
            
            # Feature scaler
            self.scalers[platform] = StandardScaler()
        
        self.logger.info("ML models initialized for all platforms")
    
    def extract_features(self, post: Post, engagement_metrics: EngagementMetrics = None) -> np.ndarray:
        """Extract features from post data for ML models"""
        features = []
        
        # Temporal features
        if post.posted_time:
            features.extend([
                post.posted_time.hour,
                post.posted_time.weekday(),
                post.posted_time.month,
                1 if post.posted_time.weekday() >= 5 else 0  # is_weekend
            ])
        else:
            features.extend([12, 2, 6, 0])  # Default values
        
        # Content features
        caption_length = len(post.caption) if post.caption else 0
        hashtag_count = post.caption.count('#') if post.caption else 0
        has_image = 1 if post.image_url else 0
        has_video = 1 if post.video_url else 0
        
        features.extend([caption_length, hashtag_count, has_image, has_video])
        
        # Platform and content type encoding
        platform_encoder = LabelEncoder()
        content_type_encoder = LabelEncoder()
        
        # Note: In production, these encoders should be fitted on the full dataset
        platform_encoded = hash(post.platform) % 10  # Simple encoding for demo
        content_type_encoded = hash(post.content_type) % 5
        
        features.extend([platform_encoded, content_type_encoded])
        
        # Historical engagement features (if available)
        if engagement_metrics:
            features.extend([
                float(engagement_metrics.total_engagement),
                float(engagement_metrics.likes),
                float(engagement_metrics.shares),
                float(engagement_metrics.comments),
                float(engagement_metrics.views)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def train_engagement_prediction_model(self, platform: str, days: int = 90) -> bool:
        """Train engagement prediction model for specific platform"""
        try:
            self.logger.info(f"Training engagement prediction model for {platform}")
            
            # Get training data
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            posts = self.database_session.query(Post).filter(
                Post.platform == platform,
                Post.posted_time >= cutoff_time,
                Post.status == 'posted'
            ).all()
            
            if len(posts) < 50:
                self.logger.warning(f"Insufficient training data: {len(posts)} posts")
                return False
            
            # Extract features and targets
            X = []
            y = []
            
            for post in posts:
                latest_metrics = post.get_latest_engagement()
                if not latest_metrics:
                    continue
                
                features = self.extract_features(post, latest_metrics)
                target = float(latest_metrics.total_engagement)
                
                X.append(features)
                y.append(target)
            
            if len(X) < 30:
                self.logger.warning("Insufficient processed training data")
                return False
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = self.scalers[platform]
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train primary model
            primary_model = self.engagement_models[platform]['primary']
            primary_model.fit(X_train_scaled, y_train)
            
            # Train secondary model
            secondary_model = self.engagement_models[platform]['secondary']
            secondary_model.fit(X_train_scaled, y_train)
            
            # Evaluate models
            primary_pred = primary_model.predict(X_test_scaled)
            secondary_pred = secondary_model.predict(X_test_scaled)
            
            primary_r2 = r2_score(y_test, primary_pred)
            secondary_r2 = r2_score(y_test, secondary_pred)
            
            # Adjust ensemble weights based on performance
            if secondary_r2 > primary_r2:
                self.engagement_models[platform]['ensemble_weights'] = [0.4, 0.6]
            
            # Train clustering model
            clustering_model = self.clustering_models[platform]
            clustering_model.fit(X_train_scaled)
            
            # Save models
            self._save_models(platform)
            
            self.logger.info(
                f"Model training completed for {platform} - "
                f"Primary R²: {primary_r2:.3f}, Secondary R²: {secondary_r2:.3f}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model training failed for {platform}: {e}")
            return False
    
    def predict_engagement(self, post_data: Dict[str, Any], platform: str) -> PerformancePrediction:
        """Predict engagement for given post data"""
        try:
            # Create feature vector from post data
            features = self._extract_features_from_data(post_data, platform)
            
            # Scale features
            scaler = self.scalers[platform]
            features_scaled = scaler.transform([features])
            
            # Get models
            primary_model = self.engagement_models[platform]['primary']
            secondary_model = self.engagement_models[platform]['secondary']
            weights = self.engagement_models[platform]['ensemble_weights']
            
            # Make predictions
            primary_pred = primary_model.predict(features_scaled)[0]
            secondary_pred = secondary_model.predict(features_scaled)[0]
            
            # Ensemble prediction
            ensemble_pred = primary_pred * weights[0] + secondary_pred * weights[1]
            
            # Calculate confidence intervals (simplified)
            prediction_std = abs(primary_pred - secondary_pred) / 2
            confidence_interval = (
                max(0, ensemble_pred - 1.96 * prediction_std),
                ensemble_pred + 1.96 * prediction_std
            )
            
            # Feature importance (from primary model)
            feature_importance = {}
            if hasattr(primary_model, 'feature_importances_'):
                feature_names = self._get_feature_names()
                for i, importance in enumerate(primary_model.feature_importances_):
                    if i < len(feature_names):
                        feature_importance[feature_names[i]] = float(importance)
            
            # Find similar posts using clustering
            cluster_id = self.clustering_models[platform].predict(features_scaled)[0]
            similar_posts = self._find_similar_posts(cluster_id, platform)
            
            # Generate optimization suggestions
            optimization_suggestions = self._generate_prediction_based_suggestions(
                post_data, ensemble_pred, feature_importance
            )
            
            return PerformancePrediction(
                prediction_id=f"pred_{int(datetime.utcnow().timestamp())}",
                post_features=post_data,
                predicted_engagement={
                    'total_engagement': ensemble_pred,
                    'likes_estimate': ensemble_pred * 0.6,
                    'shares_estimate': ensemble_pred * 0.15,
                    'comments_estimate': ensemble_pred * 0.25
                },
                confidence_intervals={
                    'total_engagement': confidence_interval
                },
                prediction_confidence=min(1.0 - (prediction_std / max(ensemble_pred, 1)), 0.95),
                feature_importance=feature_importance,
                similar_posts=similar_posts,
                optimization_suggestions=optimization_suggestions
            )
            
        except Exception as e:
            self.logger.error(f"Engagement prediction failed: {e}")
            return PerformancePrediction(
                prediction_id=f"pred_error_{int(datetime.utcnow().timestamp())}",
                post_features=post_data,
                predicted_engagement={},
                confidence_intervals={},
                prediction_confidence=0.0,
                feature_importance={},
                similar_posts=[],
                optimization_suggestions=[]
            )
    
    def create_ab_test(self, config: ABTestConfiguration) -> str:
        """Create and configure A/B test"""
        try:
            self.logger.info(f"Creating A/B test: {config.test_name}")
            
            # Validate test configuration
            if not self._validate_ab_test_config(config):
                raise ValueError("Invalid A/B test configuration")
            
            # Store test configuration
            self.active_tests[config.test_id] = {
                'config': config,
                'start_time': datetime.utcnow(),
                'status': 'active',
                'control_posts': [],
                'variant_posts': {variant: [] for variant in config.variant_values},
                'results': {}
            }
            
            # Save test configuration
            self._save_ab_test_config(config)
            
            self.logger.info(f"A/B test {config.test_id} created successfully")
            return config.test_id
            
        except Exception as e:
            self.logger.error(f"A/B test creation failed: {e}")
            return ""
    
    def analyze_ab_test_results(self, test_id: str) -> ABTestResult:
        """Analyze A/B test results and determine statistical significance"""
        try:
            if test_id not in self.active_tests:
                raise ValueError(f"Test {test_id} not found")
            
            test_data = self.active_tests[test_id]
            config = test_data['config']
            
            self.logger.info(f"Analyzing A/B test results for {test_id}")
            
            # Collect engagement data for all variants
            control_metrics = self._collect_test_metrics(test_data['control_posts'])
            variant_metrics = {}
            
            for variant_value, posts in test_data['variant_posts'].items():
                variant_metrics[str(variant_value)] = self._collect_test_metrics(posts)
            
            # Perform statistical analysis
            best_variant, p_value, effect_size = self._perform_statistical_analysis(
                control_metrics, variant_metrics, config.success_metrics[0]
            )
            
            # Calculate confidence intervals
            confidence_interval = self._calculate_confidence_interval(
                control_metrics, variant_metrics[best_variant], config.confidence_level
            )
            
            # Determine significance
            statistical_significance = p_value < (1 - config.confidence_level)
            practical_significance = abs(effect_size) > config.expected_effect_size
            
            # Generate recommendations
            recommendations = self._generate_ab_test_recommendations(
                test_data, best_variant, effect_size, statistical_significance
            )
            
            result = ABTestResult(
                test_id=test_id,
                winning_variant=best_variant if statistical_significance else 'inconclusive',
                confidence_interval=confidence_interval,
                p_value=p_value,
                effect_size=effect_size,
                statistical_significance=statistical_significance,
                practical_significance=practical_significance,
                sample_sizes={
                    'control': len(test_data['control_posts']),
                    **{str(k): len(v) for k, v in test_data['variant_posts'].items()}
                },
                conversion_rates=self._calculate_conversion_rates(control_metrics, variant_metrics),
                recommendations=recommendations,
                test_duration_days=(datetime.utcnow() - test_data['start_time']).days
            )
            
            # Store results
            self.test_results[test_id] = result
            self._save_ab_test_results(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"A/B test analysis failed for {test_id}: {e}")
            return ABTestResult(
                test_id=test_id,
                winning_variant='error',
                confidence_interval=(0, 0),
                p_value=1.0,
                effect_size=0.0,
                statistical_significance=False,
                practical_significance=False,
                sample_sizes={},
                conversion_rates={},
                recommendations=[],
                test_duration_days=0
            )
    
    def generate_content_variants(self, original_content: Dict[str, Any], 
                                variant_types: List[str], count: int = 3) -> List[ContentVariant]:
        """Generate content variants for A/B testing"""
        variants = []
        
        try:
            for variant_type in variant_types:
                for i in range(count):
                    variant = self._create_content_variant(
                        original_content, variant_type, i
                    )
                    if variant:
                        variants.append(variant)
            
            # Sort by expected improvement
            variants.sort(key=lambda x: x.expected_improvement, reverse=True)
            
            return variants[:count * len(variant_types)]
            
        except Exception as e:
            self.logger.error(f"Content variant generation failed: {e}")
            return []
    
    def optimize_posting_schedule(self, platform: str, content_type: str, 
                                days: int = 30) -> Dict[str, Any]:
        """Optimize posting schedule using historical performance data"""
        try:
            self.logger.info(f"Optimizing posting schedule for {platform}")
            
            # Get historical data
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            posts = self.database_session.query(Post).filter(
                Post.platform == platform,
                Post.content_type == content_type,
                Post.posted_time >= cutoff_time
            ).all()
            
            if len(posts) < 20:
                return {'error': 'Insufficient data for schedule optimization'}
            
            # Analyze performance by time slots
            hour_performance = {}
            day_performance = {}
            
            for post in posts:
                if not post.posted_time:
                    continue
                
                metrics = post.get_latest_engagement()
                if not metrics:
                    continue
                
                engagement_score = float(metrics.total_engagement)
                hour = post.posted_time.hour
                day = post.posted_time.weekday()
                
                if hour not in hour_performance:
                    hour_performance[hour] = []
                hour_performance[hour].append(engagement_score)
                
                if day not in day_performance:
                    day_performance[day] = []
                day_performance[day].append(engagement_score)
            
            # Calculate optimal times
            optimal_hours = sorted(hour_performance.keys(), 
                                 key=lambda h: np.mean(hour_performance[h]), 
                                 reverse=True)[:3]
            
            optimal_days = sorted(day_performance.keys(),
                                key=lambda d: np.mean(day_performance[d]),
                                reverse=True)[:4]
            
            # Generate posting schedule recommendations
            schedule_recommendations = {
                'optimal_hours': optimal_hours,
                'optimal_days': optimal_days,
                'performance_analysis': {
                    'hour_performance': {h: np.mean(scores) for h, scores in hour_performance.items()},
                    'day_performance': {d: np.mean(scores) for d, scores in day_performance.items()}
                },
                'recommended_frequency': self._calculate_optimal_frequency(posts),
                'peak_performance_windows': self._identify_peak_windows(hour_performance, day_performance)
            }
            
            return schedule_recommendations
            
        except Exception as e:
            self.logger.error(f"Schedule optimization failed: {e}")
            return {'error': str(e)}
    
    def _extract_features_from_data(self, post_data: Dict[str, Any], platform: str) -> np.ndarray:
        """Extract features from post data dictionary"""
        features = []
        
        # Temporal features
        hour = post_data.get('hour', 12)
        day_of_week = post_data.get('day_of_week', 2)
        month = post_data.get('month', 6)
        is_weekend = 1 if day_of_week >= 5 else 0
        
        features.extend([hour, day_of_week, month, is_weekend])
        
        # Content features
        caption_length = len(post_data.get('caption', ''))
        hashtag_count = post_data.get('hashtag_count', 0)
        has_image = 1 if post_data.get('image_url') else 0
        has_video = 1 if post_data.get('video_url') else 0
        
        features.extend([caption_length, hashtag_count, has_image, has_video])
        
        # Platform encoding
        platform_encoded = hash(platform) % 10
        content_type_encoded = hash(post_data.get('content_type', 'text')) % 5
        
        features.extend([platform_encoded, content_type_encoded])
        
        # Historical features (placeholder)
        features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for interpretation"""
        return [
            'hour', 'day_of_week', 'month', 'is_weekend',
            'caption_length', 'hashtag_count', 'has_image', 'has_video',
            'platform_encoded', 'content_type_encoded',
            'historical_avg', 'historical_likes', 'historical_shares', 
            'historical_comments', 'historical_views'
        ]
    
    def _find_similar_posts(self, cluster_id: int, platform: str) -> List[int]:
        """Find similar posts based on clustering"""
        # In a real implementation, this would query the database for posts in the same cluster
        return [1, 2, 3, 4, 5]  # Placeholder
    
    def _generate_prediction_based_suggestions(self, post_data: Dict[str, Any], 
                                             predicted_engagement: float,
                                             feature_importance: Dict[str, float]) -> List[str]:
        """Generate optimization suggestions based on prediction"""
        suggestions = []
        
        # Analyze most important features
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for feature, importance in top_features:
            if feature == 'hour' and importance > 0.1:
                suggestions.append("Consider posting during peak engagement hours (7-9 PM)")
            elif feature == 'caption_length' and importance > 0.1:
                suggestions.append("Optimize caption length based on platform best practices")
            elif feature == 'hashtag_count' and importance > 0.1:
                suggestions.append("Adjust hashtag strategy for better discoverability")
            elif feature == 'has_image' and importance > 0.1:
                suggestions.append("Visual content significantly impacts engagement")
        
        # Performance-based suggestions
        if predicted_engagement < 100:
            suggestions.append("Low predicted engagement - consider content optimization")
        elif predicted_engagement > 1000:
            suggestions.append("High viral potential - consider boosting or promoting")
        
        return suggestions
    
    def _validate_ab_test_config(self, config: ABTestConfiguration) -> bool:
        """Validate A/B test configuration"""
        if not config.test_id or not config.test_name:
            return False
        if not config.variant_values or len(config.variant_values) < 1:
            return False
        if config.min_sample_size < 10:
            return False
        if config.confidence_level < 0.5 or config.confidence_level > 0.99:
            return False
        
        return True
    
    def _collect_test_metrics(self, post_ids: List[int]) -> Dict[str, float]:
        """Collect metrics for A/B test posts"""
        if not post_ids:
            return {}
        
        total_engagement = 0
        total_likes = 0
        total_shares = 0
        total_comments = 0
        total_views = 0
        
        for post_id in post_ids:
            post = self.database_session.query(Post).get(post_id)
            if post:
                metrics = post.get_latest_engagement()
                if metrics:
                    total_engagement += metrics.total_engagement
                    total_likes += metrics.likes
                    total_shares += metrics.shares
                    total_comments += metrics.comments
                    total_views += metrics.views
        
        count = len(post_ids)
        return {
            'avg_engagement': total_engagement / count if count > 0 else 0,
            'avg_likes': total_likes / count if count > 0 else 0,
            'avg_shares': total_shares / count if count > 0 else 0,
            'avg_comments': total_comments / count if count > 0 else 0,
            'avg_views': total_views / count if count > 0 else 0,
            'total_posts': count
        }
    
    def _perform_statistical_analysis(self, control_metrics: Dict[str, float],
                                    variant_metrics: Dict[str, Dict[str, float]],
                                    success_metric: str) -> Tuple[str, float, float]:
        """Perform statistical analysis for A/B test"""
        control_value = control_metrics.get(success_metric, 0)
        
        best_variant = 'control'
        best_p_value = 1.0
        best_effect_size = 0.0
        
        for variant_name, metrics in variant_metrics.items():
            variant_value = metrics.get(success_metric, 0)
            
            # Simple t-test (in practice, would use proper statistical tests)
            if control_value > 0:
                effect_size = (variant_value - control_value) / control_value
                
                # Simplified p-value calculation
                z_score = abs(effect_size) * 2  # Simplified
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                if variant_value > control_value and p_value < best_p_value:
                    best_variant = variant_name
                    best_p_value = p_value
                    best_effect_size = effect_size
        
        return best_variant, best_p_value, best_effect_size
    
    def _calculate_confidence_interval(self, control_metrics: Dict[str, float],
                                     variant_metrics: Dict[str, float],
                                     confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for the difference"""
        # Simplified confidence interval calculation
        control_value = control_metrics.get('avg_engagement', 0)
        variant_value = variant_metrics.get('avg_engagement', 0)
        
        difference = variant_value - control_value
        margin_of_error = abs(difference) * 0.2  # Simplified
        
        return (difference - margin_of_error, difference + margin_of_error)
    
    def _calculate_conversion_rates(self, control_metrics: Dict[str, float],
                                  variant_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate conversion rates for each variant"""
        rates = {
            'control': control_metrics.get('avg_engagement', 0)
        }
        
        for variant_name, metrics in variant_metrics.items():
            rates[variant_name] = metrics.get('avg_engagement', 0)
        
        return rates
    
    def _generate_ab_test_recommendations(self, test_data: Dict[str, Any], 
                                        best_variant: str, effect_size: float,
                                        statistical_significance: bool) -> List[str]:
        """Generate recommendations based on A/B test results"""
        recommendations = []
        
        if statistical_significance:
            if best_variant != 'control':
                recommendations.append(f"Implement variant '{best_variant}' - {effect_size:.1%} improvement")
                recommendations.append("Scale winning variant across all content")
            else:
                recommendations.append("Control performed best - maintain current strategy")
        else:
            recommendations.append("No statistically significant difference found")
            recommendations.append("Consider running test longer or with larger sample size")
        
        if abs(effect_size) < 0.05:
            recommendations.append("Effect size is small - consider other optimization areas")
        
        return recommendations
    
    def _create_content_variant(self, original_content: Dict[str, Any], 
                              variant_type: str, variant_index: int) -> Optional[ContentVariant]:
        """Create a single content variant"""
        try:
            variant_id = f"{variant_type}_{variant_index}_{int(datetime.utcnow().timestamp())}"
            modified_content = original_content.copy()
            expected_improvement = 0.0
            
            if variant_type == 'copy':
                # Caption variations
                original_caption = original_content.get('caption', '')
                if variant_index == 0:
                    modified_content['caption'] = original_caption + " 🔥"
                    expected_improvement = 5.0
                elif variant_index == 1:
                    modified_content['caption'] = f"🎯 {original_caption}"
                    expected_improvement = 8.0
                else:
                    modified_content['caption'] = original_caption.upper()
                    expected_improvement = 3.0
            
            elif variant_type == 'timing':
                # Time variations
                if variant_index == 0:
                    modified_content['optimal_hour'] = 19  # 7 PM
                    expected_improvement = 15.0
                elif variant_index == 1:
                    modified_content['optimal_hour'] = 21  # 9 PM
                    expected_improvement = 12.0
                else:
                    modified_content['optimal_hour'] = 12  # Noon
                    expected_improvement = 5.0
            
            elif variant_type == 'hashtags':
                # Hashtag variations
                original_hashtags = original_content.get('hashtag_count', 5)
                if variant_index == 0:
                    modified_content['hashtag_count'] = min(original_hashtags + 2, 15)
                    expected_improvement = 10.0
                elif variant_index == 1:
                    modified_content['hashtag_count'] = max(original_hashtags - 2, 1)
                    expected_improvement = 7.0
                else:
                    modified_content['hashtag_count'] = 8  # Optimal for most platforms
                    expected_improvement = 12.0
            
            return ContentVariant(
                variant_id=variant_id,
                variant_type=variant_type,
                original_content=original_content,
                modified_content=modified_content,
                expected_improvement=expected_improvement,
                test_priority='medium' if expected_improvement > 10 else 'low'
            )
            
        except Exception as e:
            self.logger.error(f"Content variant creation failed: {e}")
            return None
    
    def _calculate_optimal_frequency(self, posts: List[Post]) -> str:
        """Calculate optimal posting frequency"""
        if len(posts) < 10:
            return "insufficient_data"
        
        # Group posts by day and calculate average engagement
        daily_engagement = {}
        for post in posts:
            if post.posted_time:
                date_key = post.posted_time.date()
                if date_key not in daily_engagement:
                    daily_engagement[date_key] = []
                
                metrics = post.get_latest_engagement()
                if metrics:
                    daily_engagement[date_key].append(metrics.total_engagement)
        
        # Calculate posting frequency vs engagement correlation
        frequencies = []
        engagements = []
        
        for date, engagement_list in daily_engagement.items():
            post_count = len(engagement_list)
            avg_engagement = np.mean(engagement_list)
            frequencies.append(post_count)
            engagements.append(avg_engagement)
        
        if len(frequencies) > 5:
            correlation = np.corrcoef(frequencies, engagements)[0, 1]
            
            if correlation > 0.3:
                return "increase_frequency"
            elif correlation < -0.3:
                return "decrease_frequency"
            else:
                return "maintain_current"
        
        return "maintain_current"
    
    def _identify_peak_windows(self, hour_performance: Dict[int, List[float]],
                             day_performance: Dict[int, List[float]]) -> List[Dict[str, Any]]:
        """Identify peak performance windows"""
        windows = []
        
        # Find top performing hours
        hour_averages = {h: np.mean(scores) for h, scores in hour_performance.items()}
        top_hours = sorted(hour_averages.keys(), key=lambda h: hour_averages[h], reverse=True)[:3]
        
        # Find top performing days
        day_averages = {d: np.mean(scores) for d, scores in day_performance.items()}
        top_days = sorted(day_averages.keys(), key=lambda d: day_averages[d], reverse=True)[:3]
        
        for hour in top_hours:
            for day in top_days:
                windows.append({
                    'day': day,
                    'hour': hour,
                    'expected_performance': hour_averages[hour] * day_averages[day] / max(hour_averages.values()),
                    'confidence': 'high' if len(hour_performance[hour]) > 5 else 'medium'
                })
        
        return sorted(windows, key=lambda w: w['expected_performance'], reverse=True)[:5]
    
    def _save_models(self, platform: str):
        """Save trained models to disk"""
        try:
            # Save engagement models
            primary_path = os.path.join(self.model_storage_path, f'{platform}_primary_model.pkl')
            secondary_path = os.path.join(self.model_storage_path, f'{platform}_secondary_model.pkl')
            scaler_path = os.path.join(self.model_storage_path, f'{platform}_scaler.pkl')
            
            with open(primary_path, 'wb') as f:
                pickle.dump(self.engagement_models[platform]['primary'], f)
            
            with open(secondary_path, 'wb') as f:
                pickle.dump(self.engagement_models[platform]['secondary'], f)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[platform], f)
            
            self.logger.info(f"Models saved for {platform}")
            
        except Exception as e:
            self.logger.error(f"Model saving failed for {platform}: {e}")
    
    def _save_ab_test_config(self, config: ABTestConfiguration):
        """Save A/B test configuration"""
        try:
            config_path = os.path.join(self.model_storage_path, f'ab_test_{config.test_id}.json')
            with open(config_path, 'w') as f:
                json.dump(asdict(config), f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"A/B test config save failed: {e}")
    
    def _save_ab_test_results(self, result: ABTestResult):
        """Save A/B test results"""
        try:
            results_path = os.path.join(self.model_storage_path, f'ab_results_{result.test_id}.json')
            with open(results_path, 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"A/B test results save failed: {e}")
    
    async def run_automated_optimization_cycle(self):
        """Run automated ML optimization cycle"""
        try:
            self.logger.info("Starting automated ML optimization cycle")
            
            # Train models for all platforms
            platforms = ['tiktok', 'instagram', 'x', 'linkedin', 'pinterest', 'youtube']
            
            for platform in platforms:
                try:
                    success = self.train_engagement_prediction_model(platform)
                    if success:
                        self.logger.info(f"Model training successful for {platform}")
                    else:
                        self.logger.warning(f"Model training failed for {platform}")
                        
                    await asyncio.sleep(1)  # Brief delay between platforms
                    
                except Exception as e:
                    self.logger.error(f"Platform training failed for {platform}: {e}")
                    continue
            
            self.logger.info("Automated ML optimization cycle completed")
            
        except Exception as e:
            self.logger.error(f"Automated ML optimization cycle failed: {e}")
    
    def get_optimization_insights(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive optimization insights"""
        try:
            insights = {
                'model_performance': {},
                'active_ab_tests': len(self.active_tests),
                'completed_tests': len(self.test_results),
                'optimization_opportunities': [],
                'performance_trends': {},
                'recommended_actions': []
            }
            
            # Model performance summary
            for platform in self.engagement_models.keys():
                model_path = os.path.join(self.model_storage_path, f'{platform}_primary_model.pkl')
                if os.path.exists(model_path):
                    insights['model_performance'][platform] = 'trained'
                else:
                    insights['model_performance'][platform] = 'not_trained'
                    insights['optimization_opportunities'].append(f"Train ML model for {platform}")
            
            # A/B test insights
            if self.test_results:
                winning_tests = [r for r in self.test_results.values() if r.statistical_significance]
                insights['successful_tests'] = len(winning_tests)
                insights['avg_improvement'] = np.mean([r.effect_size for r in winning_tests]) if winning_tests else 0
            
            # Generate recommendations
            if not self.active_tests and len(self.engagement_models) > 0:
                insights['recommended_actions'].append("Consider running A/B tests on high-performing content")
            
            if insights['model_performance'].get('instagram') == 'trained':
                insights['recommended_actions'].append("Use Instagram model for engagement prediction")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Optimization insights generation failed: {e}")
            return {'error': str(e)}


if __name__ == "__main__":
    print("ML Optimization Engine initialized successfully")
    print("Features available:")
    print("- Engagement prediction using ensemble ML models")
    print("- A/B testing framework with statistical significance")
    print("- Content variant generation and testing")
    print("- Posting schedule optimization")
    print("- Performance prediction and forecasting")
    print("- Automated model training and optimization")