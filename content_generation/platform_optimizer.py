"""
Platform-Specific Content Optimization Strategies
Implements specialized optimization logic for each social media platform
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import numpy as np
from decimal import Decimal

from .engagement_optimizer import EngagementOptimizer, OptimizationRecommendation


@dataclass
class PlatformOptimizationStrategy:
    """Platform-specific optimization strategy configuration"""
    platform: str
    priority_metrics: List[str]
    optimal_posting_times: List[int]
    content_format_preferences: Dict[str, float]
    engagement_multipliers: Dict[str, float]
    algorithmic_factors: Dict[str, float]
    audience_behavior: Dict[str, Any]


class BasePlatformOptimizer(ABC):
    """Base class for platform-specific optimizers"""
    
    def __init__(self, platform: str, base_optimizer: EngagementOptimizer):
        self.platform = platform
        self.base_optimizer = base_optimizer
        self.logger = logging.getLogger(f"{__name__}.{platform}")
    
    @abstractmethod
    def generate_platform_recommendations(self, post_data: Dict[str, Any], 
                                        engagement_data: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate platform-specific optimization recommendations"""
        pass
    
    @abstractmethod
    def calculate_viral_potential(self, post_data: Dict[str, Any], 
                                engagement_data: Dict[str, Any]) -> float:
        """Calculate the viral potential score for content on this platform"""
        pass
    
    @abstractmethod
    def optimize_content_timing(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize posting timing for maximum engagement"""
        pass


class TikTokOptimizer(BasePlatformOptimizer):
    """TikTok-specific content optimization"""
    
    def __init__(self, base_optimizer: EngagementOptimizer):
        super().__init__('tiktok', base_optimizer)
        
        self.viral_indicators = {
            'completion_rate': 0.3,
            'share_velocity': 0.25,
            'comment_engagement': 0.2,
            'trending_audio': 0.15,
            'hashtag_performance': 0.1
        }
        
        self.content_factors = {
            'video_length': {'15-30s': 1.0, '30-60s': 0.8, '60s+': 0.6},
            'hook_timing': {'0-3s': 1.0, '3-5s': 0.7, '5s+': 0.4},
            'text_overlay': {'present': 1.2, 'absent': 1.0},
            'trending_sound': {'yes': 1.5, 'no': 1.0, 'original': 0.8}
        }
    
    def generate_platform_recommendations(self, post_data: Dict[str, Any], 
                                        engagement_data: Dict[str, Any]) -> List[OptimizationRecommendation]:
        recommendations = []
        
        # Video length optimization
        current_length = post_data.get('video_duration', 30)
        if current_length > 45:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"tiktok_length_{post_data.get('post_id')}",
                priority="high",
                category="format",
                current_value=f"{current_length}s",
                recommended_value="15-30s",
                expected_improvement=25.0,
                confidence=0.8,
                reasoning="Shorter videos have 60% higher completion rates on TikTok",
                action_required="Create shorter, more focused video content",
                estimated_impact={
                    'completion_rate': 35.0,
                    'shares': 28.0,
                    'viral_potential': 40.0
                }
            ))
        
        # Hook optimization
        has_strong_hook = post_data.get('has_strong_hook', False)
        if not has_strong_hook:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"tiktok_hook_{post_data.get('post_id')}",
                priority="critical",
                category="content",
                current_value="weak or no hook",
                recommended_value="strong 3-second hook",
                expected_improvement=45.0,
                confidence=0.9,
                reasoning="First 3 seconds determine 70% of TikTok engagement",
                action_required="Add compelling hook in first 3 seconds",
                estimated_impact={
                    'completion_rate': 50.0,
                    'likes': 35.0,
                    'shares': 42.0,
                    'comments': 38.0
                }
            ))
        
        # Trending audio optimization
        uses_trending_audio = post_data.get('uses_trending_audio', False)
        if not uses_trending_audio:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"tiktok_audio_{post_data.get('post_id')}",
                priority="high",
                category="content",
                current_value="original or non-trending audio",
                recommended_value="trending audio track",
                expected_improvement=35.0,
                confidence=0.75,
                reasoning="Trending audio increases discovery by 300%",
                action_required="Use current trending audio that fits content",
                estimated_impact={
                    'views': 80.0,
                    'viral_potential': 45.0,
                    'algorithm_boost': 60.0
                }
            ))
        
        # Text overlay optimization
        has_text_overlay = post_data.get('has_text_overlay', False)
        if not has_text_overlay:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"tiktok_text_{post_data.get('post_id')}",
                priority="medium",
                category="visual",
                current_value="no text overlay",
                recommended_value="clear text overlay",
                expected_improvement=20.0,
                confidence=0.7,
                reasoning="Text overlays increase accessibility and retention",
                action_required="Add readable text overlay summarizing key points",
                estimated_impact={
                    'completion_rate': 25.0,
                    'accessibility_score': 40.0,
                    'engagement_rate': 18.0
                }
            ))
        
        # Hashtag strategy
        hashtag_count = post_data.get('hashtag_count', 0)
        if hashtag_count < 3 or hashtag_count > 8:
            optimal_count = 5
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"tiktok_hashtags_{post_data.get('post_id')}",
                priority="medium",
                category="content",
                current_value=f"{hashtag_count} hashtags",
                recommended_value=f"{optimal_count} hashtags",
                expected_improvement=15.0,
                confidence=0.6,
                reasoning="3-8 hashtags with mix of trending and niche tags perform best",
                action_required="Optimize hashtag strategy with trending + niche tags",
                estimated_impact={
                    'discoverability': 30.0,
                    'views': 20.0,
                    'reach': 25.0
                }
            ))
        
        return recommendations
    
    def calculate_viral_potential(self, post_data: Dict[str, Any], 
                                engagement_data: Dict[str, Any]) -> float:
        """Calculate TikTok viral potential based on key indicators"""
        viral_score = 0.0
        
        # Completion rate (most important for TikTok algorithm)
        completion_rate = engagement_data.get('completion_rate', 0.3)
        viral_score += completion_rate * self.viral_indicators['completion_rate'] * 100
        
        # Share velocity (shares per hour in first 6 hours)
        shares_per_hour = engagement_data.get('shares_per_hour', 0)
        share_velocity_score = min(shares_per_hour / 10, 1.0)  # Normalize to 0-1
        viral_score += share_velocity_score * self.viral_indicators['share_velocity'] * 100
        
        # Comment engagement rate
        comments = engagement_data.get('comments', 0)
        views = engagement_data.get('views', 1)
        comment_rate = comments / views
        comment_score = min(comment_rate * 100, 1.0)  # Cap at 1.0
        viral_score += comment_score * self.viral_indicators['comment_engagement'] * 100
        
        # Trending audio bonus
        uses_trending_audio = post_data.get('uses_trending_audio', False)
        if uses_trending_audio:
            viral_score += self.viral_indicators['trending_audio'] * 100
        
        # Hashtag performance
        trending_hashtags = post_data.get('trending_hashtags_used', 0)
        hashtag_score = min(trending_hashtags / 5, 1.0)
        viral_score += hashtag_score * self.viral_indicators['hashtag_performance'] * 100
        
        return min(viral_score, 100.0)  # Cap at 100
    
    def optimize_content_timing(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize TikTok posting timing"""
        # TikTok peak hours (based on global audience)
        peak_hours = [18, 19, 20, 21, 22]  # 6PM-10PM
        peak_days = [1, 2, 3, 4]  # Tuesday-Friday
        
        current_hour = post_data.get('posted_hour', 12)
        current_day = post_data.get('posted_day', 0)
        
        recommendations = {
            'optimal_hours': peak_hours,
            'optimal_days': peak_days,
            'current_timing_score': 0,
            'recommended_improvements': []
        }
        
        # Score current timing
        hour_score = 1.0 if current_hour in peak_hours else 0.5
        day_score = 1.0 if current_day in peak_days else 0.7
        recommendations['current_timing_score'] = (hour_score + day_score) / 2
        
        # Generate improvement recommendations
        if current_hour not in peak_hours:
            best_hour = 20  # 8PM typically best
            recommendations['recommended_improvements'].append({
                'type': 'hour_optimization',
                'current': f"{current_hour}:00",
                'recommended': f"{best_hour}:00",
                'expected_improvement': 25.0
            })
        
        if current_day not in peak_days:
            recommendations['recommended_improvements'].append({
                'type': 'day_optimization',
                'current': f"Day {current_day}",
                'recommended': "Tuesday-Friday",
                'expected_improvement': 15.0
            })
        
        return recommendations


class InstagramOptimizer(BasePlatformOptimizer):
    """Instagram-specific content optimization"""
    
    def __init__(self, base_optimizer: EngagementOptimizer):
        super().__init__('instagram', base_optimizer)
        
        self.format_multipliers = {
            'reel': 2.1,
            'carousel': 1.3,
            'single_image': 1.0,
            'story': 0.8,
            'igtv': 0.7
        }
        
        self.engagement_factors = {
            'save_rate': 0.4,  # Most important for algorithm
            'share_rate': 0.3,
            'comment_rate': 0.2,
            'like_rate': 0.1
        }
    
    def generate_platform_recommendations(self, post_data: Dict[str, Any], 
                                        engagement_data: Dict[str, Any]) -> List[OptimizationRecommendation]:
        recommendations = []
        
        # Format optimization
        current_format = post_data.get('format', 'single_image')
        if current_format != 'reel':
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"ig_format_{post_data.get('post_id')}",
                priority="high",
                category="format",
                current_value=current_format,
                recommended_value="reel",
                expected_improvement=110.0,  # Reels get 2.1x more engagement
                confidence=0.85,
                reasoning="Reels receive 110% more engagement than other formats",
                action_required="Convert content to Reel format with trending audio",
                estimated_impact={
                    'reach': 120.0,
                    'likes': 95.0,
                    'shares': 130.0,
                    'saves': 140.0
                }
            ))
        
        # Aspect ratio optimization
        aspect_ratio = post_data.get('aspect_ratio', '1:1')
        if current_format == 'reel' and aspect_ratio != '9:16':
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"ig_ratio_{post_data.get('post_id')}",
                priority="high",
                category="visual",
                current_value=aspect_ratio,
                recommended_value="9:16",
                expected_improvement=30.0,
                confidence=0.8,
                reasoning="Vertical 9:16 format performs best for Reels",
                action_required="Use vertical video format for maximum screen coverage",
                estimated_impact={
                    'completion_rate': 35.0,
                    'algorithm_preference': 40.0
                }
            ))
        
        # Save rate optimization
        save_rate = engagement_data.get('saves', 0) / max(engagement_data.get('views', 1), 1)
        if save_rate < 0.05:  # Less than 5% save rate
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"ig_saves_{post_data.get('post_id')}",
                priority="critical",
                category="content",
                current_value=f"{save_rate:.2%}",
                recommended_value="5%+ save rate",
                expected_improvement=60.0,
                confidence=0.9,
                reasoning="Save rate is primary Instagram algorithm ranking factor",
                action_required="Create saveable content: tips, tutorials, inspiration",
                estimated_impact={
                    'algorithm_boost': 80.0,
                    'reach': 70.0,
                    'long_term_performance': 90.0
                }
            ))
        
        # Hashtag optimization
        hashtag_count = post_data.get('hashtag_count', 0)
        if hashtag_count < 5 or hashtag_count > 15:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"ig_hashtags_{post_data.get('post_id')}",
                priority="medium",
                category="content",
                current_value=f"{hashtag_count} hashtags",
                recommended_value="8-12 hashtags",
                expected_improvement=20.0,
                confidence=0.7,
                reasoning="8-12 hashtags with mix of sizes perform optimally",
                action_required="Use strategic hashtag mix: 3 large, 5 medium, 4 small",
                estimated_impact={
                    'discoverability': 35.0,
                    'reach': 25.0
                }
            ))
        
        # Story completion optimization for story content
        if current_format == 'story':
            completion_rate = engagement_data.get('story_completion_rate', 0.5)
            if completion_rate < 0.7:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"ig_story_completion_{post_data.get('post_id')}",
                    priority="high",
                    category="content",
                    current_value=f"{completion_rate:.1%}",
                    recommended_value="70%+ completion",
                    expected_improvement=40.0,
                    confidence=0.75,
                    reasoning="Higher story completion increases profile visits",
                    action_required="Add interactive elements and compelling progression",
                    estimated_impact={
                        'profile_visits': 50.0,
                        'follower_conversion': 35.0
                    }
                ))
        
        return recommendations
    
    def calculate_viral_potential(self, post_data: Dict[str, Any], 
                                engagement_data: Dict[str, Any]) -> float:
        """Calculate Instagram viral potential"""
        viral_score = 0.0
        
        # Save rate (most important)
        saves = engagement_data.get('saves', 0)
        views = engagement_data.get('views', 1)
        save_rate = saves / views
        viral_score += min(save_rate * 20, 1.0) * self.engagement_factors['save_rate'] * 100
        
        # Share rate
        shares = engagement_data.get('shares', 0)
        share_rate = shares / views
        viral_score += min(share_rate * 10, 1.0) * self.engagement_factors['share_rate'] * 100
        
        # Comment engagement
        comments = engagement_data.get('comments', 0)
        comment_rate = comments / views
        viral_score += min(comment_rate * 5, 1.0) * self.engagement_factors['comment_rate'] * 100
        
        # Like rate
        likes = engagement_data.get('likes', 0)
        like_rate = likes / views
        viral_score += min(like_rate * 2, 1.0) * self.engagement_factors['like_rate'] * 100
        
        # Format bonus
        format_type = post_data.get('format', 'single_image')
        format_multiplier = self.format_multipliers.get(format_type, 1.0)
        viral_score *= format_multiplier
        
        return min(viral_score, 100.0)
    
    def optimize_content_timing(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize Instagram posting timing"""
        # Different optimal times for different content types
        timing_strategy = {
            'reel': {'hours': [11, 19, 20], 'days': [1, 2, 3, 4]},
            'single_image': {'hours': [11, 12, 19], 'days': [1, 2, 3]},
            'carousel': {'hours': [12, 19, 20], 'days': [1, 2, 3, 4]},
            'story': {'hours': [9, 12, 18, 21], 'days': [0, 1, 2, 3, 4]}
        }
        
        content_format = post_data.get('format', 'single_image')
        strategy = timing_strategy.get(content_format, timing_strategy['single_image'])
        
        return {
            'optimal_hours': strategy['hours'],
            'optimal_days': strategy['days'],
            'content_specific': True,
            'format': content_format
        }


class TwitterOptimizer(BasePlatformOptimizer):
    """X/Twitter-specific content optimization"""
    
    def __init__(self, base_optimizer: EngagementOptimizer):
        super().__init__('x', base_optimizer)
        
        self.tweet_factors = {
            'retweet_potential': 0.4,
            'reply_engagement': 0.3,
            'quote_tweet_rate': 0.2,
            'click_through_rate': 0.1
        }
    
    def generate_platform_recommendations(self, post_data: Dict[str, Any], 
                                        engagement_data: Dict[str, Any]) -> List[OptimizationRecommendation]:
        recommendations = []
        
        # Character count optimization
        char_count = post_data.get('character_count', 0)
        if char_count > 200:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"x_length_{post_data.get('post_id')}",
                priority="medium",
                category="copy",
                current_value=f"{char_count} characters",
                recommended_value="100-200 characters",
                expected_improvement=15.0,
                confidence=0.7,
                reasoning="Shorter tweets get 18% more engagement",
                action_required="Condense message while maintaining impact",
                estimated_impact={
                    'retweets': 18.0,
                    'likes': 12.0,
                    'replies': 15.0
                }
            ))
        
        # Media attachment optimization
        has_media = post_data.get('has_media', False)
        if not has_media:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"x_media_{post_data.get('post_id')}",
                priority="high",
                category="format",
                current_value="text only",
                recommended_value="include image/GIF",
                expected_improvement=80.0,
                confidence=0.85,
                reasoning="Tweets with images receive 80% more engagement",
                action_required="Add relevant image, GIF, or video",
                estimated_impact={
                    'likes': 75.0,
                    'retweets': 85.0,
                    'replies': 65.0,
                    'impressions': 90.0
                }
            ))
        
        # Thread optimization
        is_thread = post_data.get('is_thread', False)
        complex_topic = post_data.get('complex_topic', False)
        if complex_topic and not is_thread:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"x_thread_{post_data.get('post_id')}",
                priority="medium",
                category="format",
                current_value="single tweet",
                recommended_value="thread format",
                expected_improvement=35.0,
                confidence=0.6,
                reasoning="Threads increase engagement for complex topics",
                action_required="Break content into engaging thread",
                estimated_impact={
                    'engagement_time': 45.0,
                    'retweets': 30.0,
                    'bookmarks': 50.0
                }
            ))
        
        # Hashtag optimization
        hashtag_count = post_data.get('hashtag_count', 0)
        if hashtag_count > 2:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"x_hashtags_{post_data.get('post_id')}",
                priority="low",
                category="content",
                current_value=f"{hashtag_count} hashtags",
                recommended_value="1-2 hashtags max",
                expected_improvement=8.0,
                confidence=0.5,
                reasoning="Fewer hashtags look less spammy on Twitter",
                action_required="Reduce hashtag usage, focus on quality",
                estimated_impact={
                    'perceived_quality': 15.0,
                    'engagement_rate': 8.0
                }
            ))
        
        return recommendations
    
    def calculate_viral_potential(self, post_data: Dict[str, Any], 
                                engagement_data: Dict[str, Any]) -> float:
        """Calculate Twitter viral potential"""
        viral_score = 0.0
        
        # Retweet potential (most important for virality)
        retweets = engagement_data.get('retweets', 0)
        impressions = engagement_data.get('impressions', 1)
        retweet_rate = retweets / impressions
        viral_score += min(retweet_rate * 100, 1.0) * self.tweet_factors['retweet_potential'] * 100
        
        # Reply engagement
        replies = engagement_data.get('replies', 0)
        reply_rate = replies / impressions
        viral_score += min(reply_rate * 50, 1.0) * self.tweet_factors['reply_engagement'] * 100
        
        # Quote tweet rate
        quote_tweets = engagement_data.get('quote_tweets', 0)
        quote_rate = quote_tweets / impressions
        viral_score += min(quote_rate * 200, 1.0) * self.tweet_factors['quote_tweet_rate'] * 100
        
        # Click-through rate
        clicks = engagement_data.get('url_clicks', 0)
        ctr = clicks / impressions if impressions > 0 else 0
        viral_score += min(ctr * 20, 1.0) * self.tweet_factors['click_through_rate'] * 100
        
        return min(viral_score, 100.0)
    
    def optimize_content_timing(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize Twitter posting timing"""
        # Twitter has multiple peak periods throughout the day
        peak_hours = [9, 12, 15, 18, 21]  # Business hours + evening
        peak_days = [1, 2, 3, 4]  # Tuesday-Friday
        
        return {
            'optimal_hours': peak_hours,
            'optimal_days': peak_days,
            'multiple_peaks': True,
            'real_time_bonus': True  # Twitter rewards real-time engagement
        }


class LinkedInOptimizer(BasePlatformOptimizer):
    """LinkedIn-specific content optimization"""
    
    def __init__(self, base_optimizer: EngagementOptimizer):
        super().__init__('linkedin', base_optimizer)
        
        self.professional_factors = {
            'comment_quality': 0.4,
            'share_rate': 0.3,
            'click_through_rate': 0.2,
            'professional_relevance': 0.1
        }
    
    def generate_platform_recommendations(self, post_data: Dict[str, Any], 
                                        engagement_data: Dict[str, Any]) -> List[OptimizationRecommendation]:
        recommendations = []
        
        # Professional tone optimization
        professional_score = post_data.get('professional_score', 0.5)
        if professional_score < 0.7:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"linkedin_tone_{post_data.get('post_id')}",
                priority="high",
                category="copy",
                current_value=f"{professional_score:.1f} professional score",
                recommended_value="0.8+ professional score",
                expected_improvement=30.0,
                confidence=0.8,
                reasoning="Professional tone increases LinkedIn engagement by 40%",
                action_required="Adjust language to be more professional and industry-focused",
                estimated_impact={
                    'shares': 35.0,
                    'comments': 25.0,
                    'connection_requests': 40.0
                }
            ))
        
        # Document/carousel optimization
        format_type = post_data.get('format', 'text')
        if format_type not in ['document', 'carousel']:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"linkedin_format_{post_data.get('post_id')}",
                priority="medium",
                category="format",
                current_value=format_type,
                recommended_value="document or carousel",
                expected_improvement=25.0,
                confidence=0.7,
                reasoning="Documents and carousels get 300% more reach on LinkedIn",
                action_required="Convert content to document or carousel format",
                estimated_impact={
                    'reach': 200.0,
                    'shares': 150.0,
                    'saves': 180.0
                }
            ))
        
        # Industry relevance
        industry_relevance = post_data.get('industry_relevance', 0.5)
        if industry_relevance < 0.8:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"linkedin_relevance_{post_data.get('post_id')}",
                priority="medium",
                category="content",
                current_value=f"{industry_relevance:.1f} relevance",
                recommended_value="0.8+ industry relevance",
                expected_improvement=20.0,
                confidence=0.6,
                reasoning="Industry-specific content performs better on LinkedIn",
                action_required="Add industry insights and professional context",
                estimated_impact={
                    'engagement_quality': 30.0,
                    'shares': 20.0,
                    'profile_views': 25.0
                }
            ))
        
        return recommendations
    
    def calculate_viral_potential(self, post_data: Dict[str, Any], 
                                engagement_data: Dict[str, Any]) -> float:
        """Calculate LinkedIn viral potential"""
        # LinkedIn virality is different - more about professional sharing
        viral_score = 0.0
        
        # Comment quality and quantity
        comments = engagement_data.get('comments', 0)
        comment_quality_score = post_data.get('comment_quality_score', 0.5)
        viral_score += (comments * comment_quality_score) * self.professional_factors['comment_quality']
        
        # Share rate (very important for LinkedIn reach)
        shares = engagement_data.get('shares', 0)
        views = engagement_data.get('views', 1)
        share_rate = shares / views
        viral_score += min(share_rate * 20, 1.0) * self.professional_factors['share_rate'] * 100
        
        return min(viral_score, 100.0)
    
    def optimize_content_timing(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize LinkedIn posting timing"""
        # LinkedIn follows business hours more strictly
        peak_hours = [8, 9, 10, 17, 18]  # Morning and end-of-day
        peak_days = [1, 2, 3, 4]  # Tuesday-Friday
        
        return {
            'optimal_hours': peak_hours,
            'optimal_days': peak_days,
            'business_focused': True,
            'timezone_important': True
        }


class PinterestOptimizer(BasePlatformOptimizer):
    """Pinterest-specific content optimization"""
    
    def __init__(self, base_optimizer: EngagementOptimizer):
        super().__init__('pinterest', base_optimizer)
        
        self.pinterest_factors = {
            'save_rate': 0.5,  # Most important on Pinterest
            'click_through_rate': 0.3,
            'seasonal_relevance': 0.2
        }
    
    def generate_platform_recommendations(self, post_data: Dict[str, Any], 
                                        engagement_data: Dict[str, Any]) -> List[OptimizationRecommendation]:
        recommendations = []
        
        # Aspect ratio optimization
        aspect_ratio = post_data.get('aspect_ratio', '1:1')
        if aspect_ratio != '2:3':
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"pinterest_ratio_{post_data.get('post_id')}",
                priority="high",
                category="visual",
                current_value=aspect_ratio,
                recommended_value="2:3 vertical",
                expected_improvement=40.0,
                confidence=0.9,
                reasoning="2:3 vertical pins get 60% more impressions",
                action_required="Use vertical 2:3 aspect ratio for pins",
                estimated_impact={
                    'impressions': 60.0,
                    'saves': 45.0,
                    'clicks': 35.0
                }
            ))
        
        # Text overlay optimization
        has_text_overlay = post_data.get('has_text_overlay', False)
        if not has_text_overlay:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"pinterest_text_{post_data.get('post_id')}",
                priority="high",
                category="visual",
                current_value="no text overlay",
                recommended_value="clear text overlay",
                expected_improvement=35.0,
                confidence=0.8,
                reasoning="Text overlays increase Pinterest saves by 50%",
                action_required="Add clear, readable text describing the pin value",
                estimated_impact={
                    'saves': 50.0,
                    'clicks': 30.0,
                    'search_ranking': 40.0
                }
            ))
        
        return recommendations
    
    def calculate_viral_potential(self, post_data: Dict[str, Any], 
                                engagement_data: Dict[str, Any]) -> float:
        """Calculate Pinterest viral potential"""
        viral_score = 0.0
        
        # Save rate is everything on Pinterest
        saves = engagement_data.get('saves', 0)
        impressions = engagement_data.get('impressions', 1)
        save_rate = saves / impressions
        viral_score += min(save_rate * 10, 1.0) * self.pinterest_factors['save_rate'] * 100
        
        # Click-through rate
        clicks = engagement_data.get('clicks', 0)
        ctr = clicks / impressions
        viral_score += min(ctr * 20, 1.0) * self.pinterest_factors['click_through_rate'] * 100
        
        return min(viral_score, 100.0)
    
    def optimize_content_timing(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize Pinterest posting timing"""
        # Pinterest has evening peak usage
        peak_hours = [14, 15, 20, 21, 22]  # Afternoon and evening
        peak_days = [0, 5, 6]  # Weekend focused
        
        return {
            'optimal_hours': peak_hours,
            'optimal_days': peak_days,
            'seasonal_important': True,
            'evergreen_content': True  # Pinterest content has longer lifespan
        }


class YouTubeOptimizer(BasePlatformOptimizer):
    """YouTube-specific content optimization"""
    
    def __init__(self, base_optimizer: EngagementOptimizer):
        super().__init__('youtube', base_optimizer)
        
        self.youtube_factors = {
            'watch_time': 0.4,
            'click_through_rate': 0.3,
            'subscriber_conversion': 0.2,
            'comment_engagement': 0.1
        }
    
    def generate_platform_recommendations(self, post_data: Dict[str, Any], 
                                        engagement_data: Dict[str, Any]) -> List[OptimizationRecommendation]:
        recommendations = []
        
        # Thumbnail optimization
        thumbnail_ctr = engagement_data.get('thumbnail_ctr', 0.05)
        if thumbnail_ctr < 0.1:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"youtube_thumbnail_{post_data.get('post_id')}",
                priority="critical",
                category="visual",
                current_value=f"{thumbnail_ctr:.1%} CTR",
                recommended_value="10%+ CTR",
                expected_improvement=100.0,
                confidence=0.9,
                reasoning="Good thumbnails can double video performance",
                action_required="Create high-contrast thumbnail with faces and clear text",
                estimated_impact={
                    'views': 150.0,
                    'impressions': 120.0,
                    'overall_performance': 100.0
                }
            ))
        
        # Watch time optimization
        watch_time_ratio = engagement_data.get('watch_time_ratio', 0.3)
        if watch_time_ratio < 0.5:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"youtube_retention_{post_data.get('post_id')}",
                priority="high",
                category="content",
                current_value=f"{watch_time_ratio:.1%} retention",
                recommended_value="50%+ retention",
                expected_improvement=80.0,
                confidence=0.8,
                reasoning="Watch time is the primary YouTube ranking factor",
                action_required="Improve content pacing and hook strength",
                estimated_impact={
                    'algorithm_boost': 100.0,
                    'suggested_video_placement': 90.0,
                    'subscriber_growth': 70.0
                }
            ))
        
        return recommendations
    
    def calculate_viral_potential(self, post_data: Dict[str, Any], 
                                engagement_data: Dict[str, Any]) -> float:
        """Calculate YouTube viral potential"""
        viral_score = 0.0
        
        # Watch time percentage
        watch_time_ratio = engagement_data.get('watch_time_ratio', 0.3)
        viral_score += watch_time_ratio * self.youtube_factors['watch_time'] * 100
        
        # Thumbnail CTR
        thumbnail_ctr = engagement_data.get('thumbnail_ctr', 0.05)
        ctr_score = min(thumbnail_ctr * 10, 1.0)  # Normalize 10% CTR = 1.0
        viral_score += ctr_score * self.youtube_factors['click_through_rate'] * 100
        
        return min(viral_score, 100.0)
    
    def optimize_content_timing(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize YouTube posting timing"""
        # YouTube prime time
        peak_hours = [19, 20, 21, 22]  # Evening
        peak_days = [0, 1, 2, 3, 4, 5, 6]  # All days work for YouTube
        
        return {
            'optimal_hours': peak_hours,
            'optimal_days': peak_days,
            'consistency_important': True,
            'subscriber_timezone': True  # Match subscriber timezone
        }


class PlatformOptimizerManager:
    """Manages all platform-specific optimizers"""
    
    def __init__(self, base_optimizer: EngagementOptimizer):
        self.base_optimizer = base_optimizer
        
        # Initialize platform optimizers
        self.optimizers = {
            'tiktok': TikTokOptimizer(base_optimizer),
            'instagram': InstagramOptimizer(base_optimizer),
            'x': TwitterOptimizer(base_optimizer),
            'linkedin': LinkedInOptimizer(base_optimizer),
            'pinterest': PinterestOptimizer(base_optimizer),
            'youtube': YouTubeOptimizer(base_optimizer)
        }
        
        self.logger = logging.getLogger(__name__)
    
    def get_platform_optimizer(self, platform: str) -> Optional[BasePlatformOptimizer]:
        """Get optimizer for specific platform"""
        return self.optimizers.get(platform.lower())
    
    def generate_comprehensive_recommendations(self, post_data: Dict[str, Any], 
                                             engagement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for all relevant platforms"""
        platform = post_data.get('platform', '').lower()
        optimizer = self.get_platform_optimizer(platform)
        
        if not optimizer:
            self.logger.warning(f"No optimizer available for platform: {platform}")
            return {'error': f'Unsupported platform: {platform}'}
        
        try:
            recommendations = optimizer.generate_platform_recommendations(post_data, engagement_data)
            viral_potential = optimizer.calculate_viral_potential(post_data, engagement_data)
            timing_optimization = optimizer.optimize_content_timing(post_data)
            
            return {
                'platform': platform,
                'recommendations': [asdict(rec) for rec in recommendations],
                'viral_potential_score': viral_potential,
                'timing_optimization': timing_optimization,
                'optimization_count': len(recommendations),
                'high_priority_count': len([r for r in recommendations if r.priority in ['critical', 'high']])
            }
            
        except Exception as e:
            self.logger.error(f"Platform optimization failed for {platform}: {e}")
            return {'error': str(e)}
    
    def get_cross_platform_insights(self, posts_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate cross-platform optimization insights"""
        platform_performance = {}
        
        for post_data in posts_data:
            platform = post_data.get('platform', '').lower()
            optimizer = self.get_platform_optimizer(platform)
            
            if optimizer and post_data.get('engagement_data'):
                viral_score = optimizer.calculate_viral_potential(
                    post_data, post_data['engagement_data']
                )
                
                if platform not in platform_performance:
                    platform_performance[platform] = []
                platform_performance[platform].append(viral_score)
        
        # Calculate averages and insights
        insights = {
            'platform_rankings': {},
            'best_performing_platform': None,
            'optimization_priorities': {},
            'cross_platform_recommendations': []
        }
        
        for platform, scores in platform_performance.items():
            avg_score = np.mean(scores) if scores else 0
            insights['platform_rankings'][platform] = {
                'avg_viral_score': avg_score,
                'post_count': len(scores),
                'performance_tier': 'high' if avg_score > 70 else 'medium' if avg_score > 40 else 'low'
            }
        
        # Find best performing platform
        if platform_performance:
            best_platform = max(platform_performance.keys(), 
                               key=lambda p: np.mean(platform_performance[p]))
            insights['best_performing_platform'] = best_platform
        
        return insights


if __name__ == "__main__":
    print("Platform-Specific Optimization System initialized")
    print("Available platforms:")
    print("- TikTok: Completion rate, trending audio, hook optimization")
    print("- Instagram: Save rate, Reel format, aspect ratio optimization")
    print("- X/Twitter: Retweet potential, media attachment, character optimization")
    print("- LinkedIn: Professional tone, document format, industry relevance")
    print("- Pinterest: Save rate, vertical format, text overlay optimization")
    print("- YouTube: Watch time, thumbnail CTR, subscriber conversion")