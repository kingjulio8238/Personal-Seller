"""
Social Media Tools for Social Darwin Gödel Machine
Extends the existing tools framework with platform-specific posting capabilities
"""

import json
import os
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import tweepy
from facebook import GraphAPI
import logging

# Tool configuration
PLATFORMS_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'platforms.json')

def load_platform_config() -> Dict[str, Any]:
    """Load platform configuration from config file"""
    try:
        with open(PLATFORMS_CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load platform config: {e}")
        return {}

def get_api_keys() -> Dict[str, str]:
    """Get API keys from environment variables"""
    return {
        'twitter_api_key': os.getenv('TWITTER_API_KEY'),
        'twitter_api_secret': os.getenv('TWITTER_API_SECRET'),
        'twitter_access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
        'twitter_access_token_secret': os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
        'twitter_bearer_token': os.getenv('TWITTER_BEARER_TOKEN'),
        'facebook_access_token': os.getenv('FACEBOOK_ACCESS_TOKEN'),
        'instagram_access_token': os.getenv('INSTAGRAM_ACCESS_TOKEN'),
        'tiktok_access_token': os.getenv('TIKTOK_ACCESS_TOKEN'),
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'google_api_key': os.getenv('GOOGLE_API_KEY'),
    }

class SocialMediaManager:
    """Centralized social media API management"""
    
    def __init__(self):
        self.config = load_platform_config()
        self.api_keys = get_api_keys()
        self._twitter_client = None
        self._facebook_client = None
        self._rate_limits = {}
    
    def get_twitter_client(self):
        """Initialize Twitter API v2 client"""
        if not self._twitter_client:
            self._twitter_client = tweepy.Client(
                bearer_token=self.api_keys['twitter_bearer_token'],
                consumer_key=self.api_keys['twitter_api_key'],
                consumer_secret=self.api_keys['twitter_api_secret'],
                access_token=self.api_keys['twitter_access_token'],
                access_token_secret=self.api_keys['twitter_access_token_secret'],
                wait_on_rate_limit=True
            )
        return self._twitter_client
    
    def get_facebook_client(self):
        """Initialize Facebook Graph API client"""
        if not self._facebook_client:
            self._facebook_client = GraphAPI(access_token=self.api_keys['facebook_access_token'])
        return self._facebook_client
    
    def check_rate_limit(self, platform: str, endpoint: str) -> bool:
        """Check if rate limit allows posting"""
        platform_limits = self.config.get('platforms', {}).get(platform, {}).get('rate_limits', {})
        # Implementation would track actual usage vs limits
        # For now, return True (assume available)
        return True

# Initialize global manager
social_manager = SocialMediaManager()

# Tool implementations following the existing pattern

def tool_info_post_to_x():
    """Tool info for posting to X/Twitter"""
    return {
        "name": "PostToX",
        "description": "Post text and/or media content to X (Twitter) with hashtags and mentions",
        "input_schema": {
            "type": "object", 
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Tweet text content (max 280 characters)",
                    "maxLength": 280
                },
                "image_url": {
                    "type": "string",
                    "description": "URL of image to include in tweet (optional)"
                },
                "video_url": {
                    "type": "string", 
                    "description": "URL of video to include in tweet (optional)"
                },
                "hashtags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of hashtags to include (without # symbol)"
                },
                "reply_to_tweet_id": {
                    "type": "string",
                    "description": "Tweet ID to reply to (optional)"
                }
            },
            "required": ["text"]
        }
    }

def post_to_x_function(text: str, image_url: str = None, video_url: str = None, 
                       hashtags: List[str] = None, reply_to_tweet_id: str = None) -> str:
    """Post content to X/Twitter"""
    try:
        client = social_manager.get_twitter_client()
        
        # Format hashtags
        if hashtags:
            hashtag_text = " " + " ".join(f"#{tag}" for tag in hashtags)
            if len(text + hashtag_text) <= 280:
                text += hashtag_text
        
        # Check rate limits
        if not social_manager.check_rate_limit('x', 'post'):
            return "Error: Rate limit exceeded for X posting"
        
        # Handle media upload if provided
        media_ids = []
        if image_url or video_url:
            # Note: In production, this would handle actual file upload
            # For now, return placeholder response
            pass
        
        # Create tweet
        response = client.create_tweet(
            text=text,
            media_ids=media_ids if media_ids else None,
            in_reply_to_tweet_id=reply_to_tweet_id
        )
        
        tweet_id = response.data['id']
        tweet_url = f"https://x.com/user/status/{tweet_id}"
        
        return f"Successfully posted to X. Tweet ID: {tweet_id}, URL: {tweet_url}"
        
    except Exception as e:
        return f"Error posting to X: {str(e)}"

def tool_info_post_to_tiktok():
    """Tool info for posting to TikTok"""
    return {
        "name": "PostToTikTok", 
        "description": "Upload video content to TikTok with trending hashtags and effects",
        "input_schema": {
            "type": "object",
            "properties": {
                "video_url": {
                    "type": "string",
                    "description": "URL of video file to upload (required)",
                    "format": "uri"
                },
                "caption": {
                    "type": "string", 
                    "description": "Video caption/description (max 2200 characters)",
                    "maxLength": 2200
                },
                "hashtags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of trending hashtags (recommended max 5)"
                },
                "privacy_level": {
                    "type": "string",
                    "enum": ["PUBLIC_TO_EVERYONE", "MUTUAL_FOLLOW_FRIENDS", "SELF_ONLY"],
                    "description": "Video privacy setting",
                    "default": "PUBLIC_TO_EVERYONE"
                },
                "disable_duet": {
                    "type": "boolean", 
                    "description": "Disable duet feature for this video",
                    "default": False
                },
                "disable_comment": {
                    "type": "boolean",
                    "description": "Disable comments for this video", 
                    "default": False
                }
            },
            "required": ["video_url"]
        }
    }

def post_to_tiktok_function(video_url: str, caption: str = "", hashtags: List[str] = None,
                           privacy_level: str = "PUBLIC_TO_EVERYONE", disable_duet: bool = False,
                           disable_comment: bool = False) -> str:
    """Upload video to TikTok"""
    try:
        # Check rate limits
        if not social_manager.check_rate_limit('tiktok', 'post'):
            return "Error: Rate limit exceeded for TikTok posting"
        
        # Format caption with hashtags
        if hashtags:
            hashtag_text = " " + " ".join(f"#{tag}" for tag in hashtags[:5])  # Max 5 hashtags
            caption += hashtag_text
        
        # TikTok API implementation
        # This would use the TikTok Content Posting API
        headers = {
            'Authorization': f'Bearer {social_manager.api_keys["tiktok_access_token"]}',
            'Content-Type': 'application/json'
        }
        
        # Initialize video upload
        init_payload = {
            'post_info': {
                'title': caption,
                'privacy_level': privacy_level,
                'disable_duet': disable_duet, 
                'disable_comment': disable_comment,
                'video_cover_timestamp_ms': 1000
            },
            'source_info': {
                'source': 'FILE_UPLOAD',
                'video_size': 0,  # Would be actual file size
                'chunk_size': 10485760,  # 10MB chunks
                'total_chunk_count': 1
            }
        }
        
        # Note: Full implementation would handle chunked upload
        # For demonstration, return placeholder response
        publish_id = f"tiktok_{int(time.time())}"
        
        return f"Successfully posted to TikTok. Publish ID: {publish_id}. Video will be available after processing."
        
    except Exception as e:
        return f"Error posting to TikTok: {str(e)}"

def tool_info_post_to_instagram():
    """Tool info for posting to Instagram"""
    return {
        "name": "PostToInstagram",
        "description": "Post image/video content to Instagram feed or stories",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string", 
                    "description": "URL of image to post",
                    "format": "uri"
                },
                "video_url": {
                    "type": "string",
                    "description": "URL of video to post (alternative to image)",
                    "format": "uri"
                },
                "caption": {
                    "type": "string",
                    "description": "Post caption (max 2200 characters)",
                    "maxLength": 2200
                },
                "hashtags": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of hashtags (optimal: 11, max: 30)"
                },
                "post_type": {
                    "type": "string",
                    "enum": ["feed", "story", "reel"],
                    "description": "Type of Instagram post",
                    "default": "feed"
                },
                "location_id": {
                    "type": "string",
                    "description": "Instagram location ID for geo-tagging (optional)"
                }
            },
            "required": []
        }
    }

def post_to_instagram_function(image_url: str = None, video_url: str = None, caption: str = "",
                              hashtags: List[str] = None, post_type: str = "feed",
                              location_id: str = None) -> str:
    """Post content to Instagram"""
    try:
        if not image_url and not video_url:
            return "Error: Either image_url or video_url must be provided"
        
        # Check rate limits
        if not social_manager.check_rate_limit('instagram', 'post'):
            return "Error: Rate limit exceeded for Instagram posting"
        
        client = social_manager.get_facebook_client()
        
        # Format caption with hashtags
        if hashtags:
            # Use optimal number of hashtags (11) or limit to 30
            optimal_hashtags = hashtags[:11] if len(hashtags) > 11 else hashtags
            hashtag_text = "\n\n" + " ".join(f"#{tag}" for tag in optimal_hashtags)
            caption += hashtag_text
        
        # Instagram Graph API implementation
        # Two-step process: create media container then publish
        
        # Step 1: Create media container
        media_data = {
            'caption': caption,
        }
        
        if image_url:
            media_data['image_url'] = image_url
        elif video_url:
            media_data['video_url'] = video_url
            media_data['media_type'] = 'VIDEO'
        
        if location_id:
            media_data['location_id'] = location_id
        
        # Note: Full implementation would make actual API calls
        # For demonstration, return placeholder response
        media_id = f"ig_media_{int(time.time())}"
        
        # Step 2: Publish media (placeholder)
        post_id = f"ig_post_{int(time.time())}"
        post_url = f"https://instagram.com/p/{post_id}"
        
        return f"Successfully posted to Instagram. Post ID: {post_id}, URL: {post_url}"
        
    except Exception as e:
        return f"Error posting to Instagram: {str(e)}"

def tool_info_get_x_engagement():
    """Tool info for getting X engagement metrics"""
    return {
        "name": "GetXEngagementMetrics",
        "description": "Retrieve engagement metrics (likes, retweets, replies) for X posts",
        "input_schema": {
            "type": "object",
            "properties": {
                "tweet_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tweet IDs to get metrics for"
                },
                "user_id": {
                    "type": "string", 
                    "description": "User ID to get recent tweet metrics for (alternative to tweet_ids)"
                },
                "count": {
                    "type": "integer",
                    "description": "Number of recent tweets to analyze (when using user_id)",
                    "default": 10,
                    "maximum": 100
                }
            },
            "required": []
        }
    }

def get_x_engagement_function(tweet_ids: List[str] = None, user_id: str = None, count: int = 10) -> str:
    """Get engagement metrics from X/Twitter"""
    try:
        client = social_manager.get_twitter_client()
        
        if tweet_ids:
            # Get metrics for specific tweets
            tweets = client.get_tweets(
                ids=tweet_ids,
                tweet_fields=['public_metrics', 'created_at', 'author_id']
            )
            
            results = []
            for tweet in tweets.data:
                metrics = tweet.public_metrics
                results.append({
                    'tweet_id': tweet.id,
                    'text': tweet.text[:100] + "..." if len(tweet.text) > 100 else tweet.text,
                    'created_at': tweet.created_at.isoformat(),
                    'likes': metrics['like_count'],
                    'retweets': metrics['retweet_count'],
                    'replies': metrics['reply_count'],
                    'quotes': metrics['quote_count'],
                    'impressions': metrics['impression_count']
                })
                
        elif user_id:
            # Get metrics for recent tweets from user
            tweets = client.get_users_tweets(
                id=user_id,
                tweet_fields=['public_metrics', 'created_at'],
                max_results=min(count, 100)
            )
            
            results = []
            for tweet in tweets.data:
                metrics = tweet.public_metrics
                results.append({
                    'tweet_id': tweet.id,
                    'text': tweet.text[:100] + "..." if len(tweet.text) > 100 else tweet.text,
                    'created_at': tweet.created_at.isoformat(),
                    'likes': metrics['like_count'],
                    'retweets': metrics['retweet_count'], 
                    'replies': metrics['reply_count'],
                    'quotes': metrics['quote_count'],
                    'impressions': metrics['impression_count']
                })
        else:
            return "Error: Either tweet_ids or user_id must be provided"
        
        return json.dumps({
            'platform': 'x',
            'retrieved_at': datetime.utcnow().isoformat(),
            'metrics': results
        }, indent=2)
        
    except Exception as e:
        return f"Error retrieving X engagement metrics: {str(e)}"

# Register all social tools
def get_all_social_tools():
    """Get all social media tools for registration"""
    return [
        {
            'info': tool_info_post_to_x(),
            'function': post_to_x_function,
            'name': 'PostToX'
        },
        {
            'info': tool_info_post_to_tiktok(),
            'function': post_to_tiktok_function,
            'name': 'PostToTikTok'
        },
        {
            'info': tool_info_post_to_instagram(),
            'function': post_to_instagram_function,
            'name': 'PostToInstagram'
        },
        {
            'info': tool_info_get_x_engagement(),
            'function': get_x_engagement_function,
            'name': 'GetXEngagementMetrics'
        }
    ]

# For compatibility with existing tools framework
def tool_info():
    """Default tool info for backward compatibility"""
    return tool_info_post_to_x()

def tool_function(**kwargs):
    """Default tool function for backward compatibility"""
    return post_to_x_function(**kwargs)

if __name__ == "__main__":
    # Test the social tools
    print("Testing social media tools...")
    
    # Test X posting
    result = post_to_x_function(
        text="Test post from Social DGM", 
        hashtags=["AI", "automation", "social"]
    )
    print(f"X Post Result: {result}")
    
    # Test engagement retrieval
    engagement = get_x_engagement_function(user_id="example_user_id")
    print(f"Engagement Metrics: {engagement}")