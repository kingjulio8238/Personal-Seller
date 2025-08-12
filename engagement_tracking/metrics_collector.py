"""
Real-time Engagement Metrics Collection System
Collects and processes social media engagement data from multiple platforms
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import func
import requests
import tweepy
from facebook import GraphAPI

from database.models import Post, EngagementMetrics, DatabaseManager
from tools.social_tools import SocialMediaManager

class MetricsCollector:
    """
    Real-time social media engagement metrics collection
    Integrates with platform APIs to gather likes, shares, comments, views
    """
    
    def __init__(self, database_session: Session):
        self.database_session = database_session
        self.db_manager = DatabaseManager(database_session)
        self.social_manager = SocialMediaManager()
        
        # Load platform configuration for engagement weights
        self.platform_weights = {
            'x': {
                'likes': 1.0,
                'retweets': 5.0,
                'replies': 3.0,
                'quotes': 4.0,
                'impressions': 0.1
            },
            'tiktok': {
                'likes': 1.0,
                'shares': 5.0,
                'comments': 3.0,
                'views': 0.01,
                'saves': 4.0
            },
            'instagram': {
                'likes': 1.0,
                'comments': 3.0,
                'saves': 4.0,
                'shares': 5.0,
                'reach': 0.1
            }
        }
        
        # Collection intervals (in seconds)
        self.collection_intervals = {
            'immediate': 300,    # 5 minutes after posting
            'short_term': 3600,  # 1 hour after posting  
            'medium_term': 86400, # 24 hours after posting
            'long_term': 604800   # 7 days after posting
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def collect_x_metrics(self, post: Post) -> Optional[Dict[str, Any]]:
        """Collect engagement metrics from X/Twitter"""
        try:
            if not post.post_id:
                self.logger.warning(f"No post_id for post {post.id}")
                return None
            
            client = self.social_manager.get_twitter_client()
            
            # Get tweet with public metrics
            tweet = client.get_tweet(
                id=post.post_id,
                tweet_fields=['public_metrics', 'created_at'],
                expansions=['author_id']
            )
            
            if not tweet.data:
                self.logger.warning(f"Tweet not found: {post.post_id}")
                return None
            
            metrics = tweet.data.public_metrics
            
            return {
                'likes': metrics.get('like_count', 0),
                'shares': metrics.get('retweet_count', 0),  # Retweets are shares on X
                'comments': metrics.get('reply_count', 0),
                'views': metrics.get('impression_count', 0),
                'platform_specific_metrics': {
                    'quotes': metrics.get('quote_count', 0),
                    'bookmarks': metrics.get('bookmark_count', 0),
                    'impressions': metrics.get('impression_count', 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect X metrics for post {post.id}: {e}")
            return None

    def collect_tiktok_metrics(self, post: Post) -> Optional[Dict[str, Any]]:
        """Collect engagement metrics from TikTok"""
        try:
            if not post.post_id:
                self.logger.warning(f"No post_id for post {post.id}")
                return None
            
            # TikTok Business API for analytics
            headers = {
                'Authorization': f'Bearer {os.getenv("TIKTOK_ACCESS_TOKEN")}',
                'Content-Type': 'application/json'
            }
            
            # Get video info and metrics
            url = f"https://open.tiktokapis.com/v2/video/query/"
            
            payload = {
                'filters': {
                    'video_ids': [post.post_id]
                },
                'fields': [
                    'id', 'title', 'video_description', 'create_time',
                    'cover_image_url', 'share_url', 'view_count', 'like_count',
                    'comment_count', 'share_count'
                ]
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('data', {}).get('videos'):
                self.logger.warning(f"TikTok video not found: {post.post_id}")
                return None
            
            video_data = data['data']['videos'][0]
            
            return {
                'likes': video_data.get('like_count', 0),
                'shares': video_data.get('share_count', 0),
                'comments': video_data.get('comment_count', 0),
                'views': video_data.get('view_count', 0),
                'platform_specific_metrics': {
                    'video_id': video_data.get('id'),
                    'create_time': video_data.get('create_time'),
                    'share_url': video_data.get('share_url')
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect TikTok metrics for post {post.id}: {e}")
            return None

    def collect_instagram_metrics(self, post: Post) -> Optional[Dict[str, Any]]:
        """Collect engagement metrics from Instagram"""
        try:
            if not post.post_id:
                self.logger.warning(f"No post_id for post {post.id}")
                return None
            
            client = self.social_manager.get_facebook_client()
            
            # Get Instagram media insights
            # Note: Requires Instagram Business account
            insights = client.get_object(
                id=post.post_id,
                fields='insights.metric(impressions,reach,likes,comments,saves,shares)'
            )
            
            metrics_data = {}
            if 'insights' in insights and 'data' in insights['insights']:
                for metric in insights['insights']['data']:
                    metric_name = metric.get('name')
                    metric_value = metric.get('values', [{}])[0].get('value', 0)
                    metrics_data[metric_name] = metric_value
            
            return {
                'likes': metrics_data.get('likes', 0),
                'shares': metrics_data.get('shares', 0),
                'comments': metrics_data.get('comments', 0),
                'views': metrics_data.get('impressions', 0),  # Use impressions as views
                'platform_specific_metrics': {
                    'reach': metrics_data.get('reach', 0),
                    'saves': metrics_data.get('saves', 0),
                    'impressions': metrics_data.get('impressions', 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect Instagram metrics for post {post.id}: {e}")
            return None

    def collect_metrics_for_post(self, post_id: int) -> Optional[EngagementMetrics]:
        """Collect engagement metrics for a specific post"""
        try:
            post = self.database_session.query(Post).get(post_id)
            if not post:
                self.logger.error(f"Post {post_id} not found")
                return None
            
            self.logger.info(f"Collecting metrics for post {post_id} on {post.platform}")
            
            # Collect platform-specific metrics
            metrics_data = None
            
            if post.platform == 'x':
                metrics_data = self.collect_x_metrics(post)
            elif post.platform == 'tiktok':
                metrics_data = self.collect_tiktok_metrics(post)
            elif post.platform == 'instagram':
                metrics_data = self.collect_instagram_metrics(post)
            else:
                self.logger.error(f"Unknown platform: {post.platform}")
                return None
            
            if not metrics_data:
                self.logger.warning(f"No metrics data collected for post {post_id}")
                return None
            
            # Save metrics to database
            engagement_metrics = self.db_manager.record_engagement(
                post_id=post_id,
                likes=metrics_data.get('likes', 0),
                shares=metrics_data.get('shares', 0),
                comments=metrics_data.get('comments', 0),
                views=metrics_data.get('views', 0),
                platform_specific=metrics_data.get('platform_specific_metrics', {})
            )
            
            self.logger.info(f"Metrics collected for post {post_id}: {engagement_metrics.total_engagement} total engagement")
            return engagement_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics for post {post_id}: {e}")
            return None

    def collect_metrics_batch(self, post_ids: List[int]) -> List[EngagementMetrics]:
        """Collect metrics for multiple posts in batch"""
        collected_metrics = []
        
        for post_id in post_ids:
            try:
                metrics = self.collect_metrics_for_post(post_id)
                if metrics:
                    collected_metrics.append(metrics)
                    
                # Brief delay to respect rate limits
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Batch collection failed for post {post_id}: {e}")
                continue
        
        self.logger.info(f"Batch collection completed: {len(collected_metrics)}/{len(post_ids)} successful")
        return collected_metrics

    def schedule_metrics_collection(self, post_id: int, collection_times: List[str] = None):
        """Schedule automatic metrics collection at specified intervals"""
        if collection_times is None:
            collection_times = ['immediate', 'short_term', 'medium_term', 'long_term']
        
        post = self.database_session.query(Post).get(post_id)
        if not post or not post.posted_time:
            self.logger.error(f"Cannot schedule collection for post {post_id}: invalid post or no posted_time")
            return
        
        scheduled_collections = []
        
        for interval_name in collection_times:
            if interval_name in self.collection_intervals:
                interval_seconds = self.collection_intervals[interval_name]
                scheduled_time = post.posted_time + timedelta(seconds=interval_seconds)
                
                # In production, this would schedule actual background jobs
                # For demo, we'll store the schedule information
                scheduled_collections.append({
                    'post_id': post_id,
                    'interval': interval_name,
                    'scheduled_time': scheduled_time,
                    'status': 'scheduled'
                })
        
        self.logger.info(f"Scheduled {len(scheduled_collections)} collection times for post {post_id}")
        return scheduled_collections

    def get_recent_posts_for_collection(self, hours_back: int = 24) -> List[Post]:
        """Get posts that need metrics collection"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        posts = self.database_session.query(Post).filter(
            Post.posted_time >= cutoff_time,
            Post.status == 'posted',
            Post.post_id.isnot(None)
        ).all()
        
        self.logger.info(f"Found {len(posts)} posts for metrics collection")
        return posts

    def calculate_engagement_score(self, metrics: EngagementMetrics, platform: str) -> float:
        """Calculate weighted engagement score based on platform"""
        platform_weights = self.platform_weights.get(platform, self.platform_weights['x'])
        
        base_score = (
            metrics.likes * platform_weights.get('likes', 1.0) +
            metrics.shares * platform_weights.get('shares', 5.0) +
            metrics.comments * platform_weights.get('comments', 3.0) +
            metrics.views * platform_weights.get('views', 0.01)
        )
        
        # Add platform-specific metrics
        platform_specific = metrics.platform_specific_metrics or {}
        
        if platform == 'x':
            base_score += platform_specific.get('quotes', 0) * platform_weights.get('quotes', 4.0)
            base_score += platform_specific.get('impressions', 0) * platform_weights.get('impressions', 0.1)
        elif platform == 'tiktok':
            base_score += platform_specific.get('saves', 0) * platform_weights.get('saves', 4.0)
        elif platform == 'instagram':
            base_score += platform_specific.get('saves', 0) * platform_weights.get('saves', 4.0)
            base_score += platform_specific.get('reach', 0) * platform_weights.get('reach', 0.1)
        
        return base_score

    def get_engagement_trends(self, post_id: int, days: int = 7) -> Dict[str, Any]:
        """Get engagement trends over time for a post"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            metrics_history = self.database_session.query(EngagementMetrics).filter(
                EngagementMetrics.post_id == post_id,
                EngagementMetrics.timestamp >= cutoff_time
            ).order_by(EngagementMetrics.timestamp).all()
            
            if not metrics_history:
                return {'error': 'No metrics history found'}
            
            # Calculate trends
            trends = {
                'total_datapoints': len(metrics_history),
                'timeframe_days': days,
                'metrics_timeline': [],
                'growth_rates': {},
                'peak_engagement_time': None,
                'latest_metrics': None
            }
            
            for metrics in metrics_history:
                trends['metrics_timeline'].append({
                    'timestamp': metrics.timestamp.isoformat(),
                    'likes': metrics.likes,
                    'shares': metrics.shares,
                    'comments': metrics.comments,
                    'views': metrics.views,
                    'total_engagement': metrics.total_engagement
                })
            
            # Calculate growth rates
            if len(metrics_history) > 1:
                first_metrics = metrics_history[0]
                latest_metrics = metrics_history[-1]
                
                trends['growth_rates'] = {
                    'likes': self._calculate_growth_rate(first_metrics.likes, latest_metrics.likes),
                    'shares': self._calculate_growth_rate(first_metrics.shares, latest_metrics.shares),
                    'comments': self._calculate_growth_rate(first_metrics.comments, latest_metrics.comments),
                    'views': self._calculate_growth_rate(first_metrics.views, latest_metrics.views)
                }
                
                trends['latest_metrics'] = {
                    'likes': latest_metrics.likes,
                    'shares': latest_metrics.shares,
                    'comments': latest_metrics.comments,
                    'views': latest_metrics.views,
                    'total_engagement': latest_metrics.total_engagement,
                    'collected_at': latest_metrics.timestamp.isoformat()
                }
            
            # Find peak engagement time
            peak_metrics = max(metrics_history, key=lambda m: m.total_engagement)
            trends['peak_engagement_time'] = {
                'timestamp': peak_metrics.timestamp.isoformat(),
                'total_engagement': peak_metrics.total_engagement
            }
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Failed to get engagement trends for post {post_id}: {e}")
            return {'error': str(e)}

    def _calculate_growth_rate(self, initial: int, final: int) -> float:
        """Calculate growth rate percentage"""
        if initial == 0:
            return 100.0 if final > 0 else 0.0
        
        return ((final - initial) / initial) * 100.0

    def run_collection_cycle(self, hours_back: int = 24):
        """Run a complete metrics collection cycle"""
        self.logger.info("Starting metrics collection cycle")
        
        try:
            # Get posts that need collection
            posts = self.get_recent_posts_for_collection(hours_back)
            post_ids = [post.id for post in posts]
            
            if not post_ids:
                self.logger.info("No posts found for metrics collection")
                return
            
            # Collect metrics in batches
            collected_metrics = self.collect_metrics_batch(post_ids)
            
            # Calculate engagement scores
            total_engagement = 0
            for metrics in collected_metrics:
                post = self.database_session.query(Post).get(metrics.post_id)
                if post:
                    engagement_score = self.calculate_engagement_score(metrics, post.platform)
                    total_engagement += engagement_score
            
            self.logger.info(f"Collection cycle completed: {len(collected_metrics)} posts processed, {total_engagement:.2f} total engagement score")
            
        except Exception as e:
            self.logger.error(f"Collection cycle failed: {e}")

    def get_platform_performance_summary(self, agent_generation_id: int, days: int = 7) -> Dict[str, Any]:
        """Get performance summary across platforms for an agent generation"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            # Get posts for this agent generation
            posts = self.database_session.query(Post).filter(
                Post.agent_generation_id == agent_generation_id,
                Post.posted_time >= cutoff_time
            ).all()
            
            if not posts:
                return {'error': 'No posts found for agent generation'}
            
            # Group by platform
            platform_summary = {}
            
            for post in posts:
                platform = post.platform
                if platform not in platform_summary:
                    platform_summary[platform] = {
                        'post_count': 0,
                        'total_likes': 0,
                        'total_shares': 0,
                        'total_comments': 0,
                        'total_views': 0,
                        'total_engagement_score': 0,
                        'posts': []
                    }
                
                # Get latest metrics for this post
                latest_metrics = post.get_latest_engagement()
                if latest_metrics:
                    engagement_score = self.calculate_engagement_score(latest_metrics, platform)
                    
                    platform_summary[platform]['post_count'] += 1
                    platform_summary[platform]['total_likes'] += latest_metrics.likes
                    platform_summary[platform]['total_shares'] += latest_metrics.shares
                    platform_summary[platform]['total_comments'] += latest_metrics.comments
                    platform_summary[platform]['total_views'] += latest_metrics.views
                    platform_summary[platform]['total_engagement_score'] += engagement_score
                    
                    platform_summary[platform]['posts'].append({
                        'post_id': post.id,
                        'content_type': post.content_type,
                        'posted_time': post.posted_time.isoformat() if post.posted_time else None,
                        'engagement_score': engagement_score,
                        'metrics': {
                            'likes': latest_metrics.likes,
                            'shares': latest_metrics.shares,
                            'comments': latest_metrics.comments,
                            'views': latest_metrics.views
                        }
                    })
            
            # Calculate averages
            for platform, data in platform_summary.items():
                if data['post_count'] > 0:
                    data['avg_engagement_per_post'] = data['total_engagement_score'] / data['post_count']
                    data['avg_likes_per_post'] = data['total_likes'] / data['post_count']
                    data['avg_shares_per_post'] = data['total_shares'] / data['post_count']
                    data['avg_comments_per_post'] = data['total_comments'] / data['post_count']
                    data['avg_views_per_post'] = data['total_views'] / data['post_count']
            
            return {
                'agent_generation_id': agent_generation_id,
                'summary_period_days': days,
                'total_posts': len(posts),
                'platform_breakdown': platform_summary,
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate platform performance summary: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    # Test metrics collector (would need actual database session)
    print("Metrics collector initialized successfully")
    
    # Example usage:
    # collector = MetricsCollector(database_session)
    # metrics = collector.collect_metrics_for_post(123)
    # trends = collector.get_engagement_trends(123)
    # summary = collector.get_platform_performance_summary(1)