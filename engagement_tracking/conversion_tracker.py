"""
Revenue Conversion Tracking with Temporal Attribution
Tracks and validates revenue conversions from social media posts using Stripe webhooks
"""

import os
import json
import logging
import hmac
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
import stripe
import requests
from dataclasses import dataclass

from database.models import Post, ConversionEvent, AgentGeneration, DatabaseManager

@dataclass
class ConversionData:
    """Conversion event data structure"""
    stripe_payment_id: str
    amount: Decimal
    timestamp: datetime
    customer_id: Optional[str] = None
    customer_email: Optional[str] = None
    customer_type: str = 'unknown'
    currency: str = 'usd'
    payment_method: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ConversionTracker:
    """
    Stripe integration for tracking revenue conversions with temporal attribution
    Validates that conversions occur within agent's active period + attribution window
    """
    
    def __init__(self, database_session: Session):
        self.database_session = database_session
        self.db_manager = DatabaseManager(database_session)
        
        # Initialize Stripe
        stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
        self.webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
        
        # Conversion attribution settings
        self.default_attribution_window_hours = 72  # 3 days
        self.max_attribution_window_hours = 720     # 30 days
        
        # Attribution confidence scoring
        self.attribution_confidence_factors = {
            'direct_link_click': 0.9,      # High confidence - direct click from social
            'referrer_match': 0.8,         # High confidence - referrer URL matches
            'time_proximity': 0.7,         # Medium confidence - purchase soon after post
            'customer_behavior': 0.6,       # Medium confidence - behavioral patterns
            'utm_parameter_match': 0.9,     # High confidence - UTM tracking
            'session_tracking': 0.95,      # Very high confidence - session continuity
            'default': 0.5                 # Default confidence level
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def verify_stripe_webhook(self, payload: bytes, signature: str) -> bool:
        """Verify Stripe webhook signature for security"""
        try:
            expected_signature = hmac.new(
                self.webhook_secret.encode('utf-8'),
                payload,
                hashlib.sha256
            ).hexdigest()
            
            # Stripe sends signature as "t=timestamp,v1=signature"
            signature_elements = signature.split(',')
            signature_dict = {}
            
            for element in signature_elements:
                key, value = element.split('=', 1)
                signature_dict[key] = value
            
            if 'v1' not in signature_dict:
                return False
            
            return hmac.compare_digest(expected_signature, signature_dict['v1'])
            
        except Exception as e:
            self.logger.error(f"Webhook signature verification failed: {e}")
            return False

    def process_stripe_webhook(self, payload: Dict[str, Any]) -> Optional[ConversionData]:
        """Process Stripe webhook payload and extract conversion data"""
        try:
            event_type = payload.get('type')
            
            # Only process successful payment events
            if event_type not in ['payment_intent.succeeded', 'charge.succeeded', 'checkout.session.completed']:
                self.logger.info(f"Ignoring webhook event type: {event_type}")
                return None
            
            data_object = payload.get('data', {}).get('object', {})
            
            # Extract payment information
            payment_id = data_object.get('id')
            amount_cents = data_object.get('amount') or data_object.get('amount_total', 0)
            amount = Decimal(amount_cents) / 100  # Convert cents to dollars
            currency = data_object.get('currency', 'usd')
            
            # Get customer information
            customer_id = data_object.get('customer')
            customer_email = data_object.get('receipt_email') or data_object.get('customer_details', {}).get('email')
            
            # Extract metadata for attribution
            metadata = data_object.get('metadata', {})
            
            # Get timestamp (convert from Unix timestamp if needed)
            timestamp = datetime.fromtimestamp(payload.get('created', datetime.utcnow().timestamp()))
            
            conversion_data = ConversionData(
                stripe_payment_id=payment_id,
                amount=amount,
                timestamp=timestamp,
                customer_id=customer_id,
                customer_email=customer_email,
                currency=currency,
                metadata=metadata
            )
            
            self.logger.info(f"Processed Stripe webhook: ${amount} payment {payment_id}")
            return conversion_data
            
        except Exception as e:
            self.logger.error(f"Failed to process Stripe webhook: {e}")
            return None

    def determine_customer_type(self, conversion_data: ConversionData) -> str:
        """Determine if customer is new or returning based on purchase history"""
        try:
            if not conversion_data.customer_id and not conversion_data.customer_email:
                return 'unknown'
            
            # Query existing conversions
            existing_conversions_query = self.database_session.query(ConversionEvent)
            
            if conversion_data.customer_id:
                # Check by Stripe customer ID
                stripe_metadata_filter = func.json_extract(ConversionEvent.metadata, '$.customer_id') == conversion_data.customer_id
                existing_conversions_query = existing_conversions_query.filter(stripe_metadata_filter)
            elif conversion_data.customer_email:
                # Check by email
                email_metadata_filter = func.json_extract(ConversionEvent.metadata, '$.customer_email') == conversion_data.customer_email
                existing_conversions_query = existing_conversions_query.filter(email_metadata_filter)
            
            existing_conversions = existing_conversions_query.first()
            
            return 'returning' if existing_conversions else 'new'
            
        except Exception as e:
            self.logger.error(f"Failed to determine customer type: {e}")
            return 'unknown'

    def find_attributable_posts(self, conversion_data: ConversionData, 
                              attribution_window_hours: int = None) -> List[Tuple[Post, float]]:
        """Find posts that could be attributed to this conversion with confidence scores"""
        if attribution_window_hours is None:
            attribution_window_hours = self.default_attribution_window_hours
        
        try:
            # Calculate attribution window
            window_start = conversion_data.timestamp - timedelta(hours=attribution_window_hours)
            window_end = conversion_data.timestamp
            
            # Find posts within attribution window
            potential_posts = self.database_session.query(Post).filter(
                and_(
                    Post.posted_time >= window_start,
                    Post.posted_time <= window_end,
                    Post.status == 'posted',
                    Post.post_id.isnot(None)
                )
            ).all()
            
            if not potential_posts:
                self.logger.info("No posts found within attribution window")
                return []
            
            # Score each post for attribution confidence
            scored_posts = []
            
            for post in potential_posts:
                confidence_score = self.calculate_attribution_confidence(post, conversion_data)
                
                if confidence_score > 0.3:  # Minimum confidence threshold
                    scored_posts.append((post, confidence_score))
            
            # Sort by confidence score (highest first)
            scored_posts.sort(key=lambda x: x[1], reverse=True)
            
            self.logger.info(f"Found {len(scored_posts)} attributable posts for conversion {conversion_data.stripe_payment_id}")
            return scored_posts
            
        except Exception as e:
            self.logger.error(f"Failed to find attributable posts: {e}")
            return []

    def calculate_attribution_confidence(self, post: Post, conversion_data: ConversionData) -> float:
        """Calculate confidence score for attributing conversion to a specific post"""
        try:
            confidence_factors = []
            
            # Time proximity factor
            time_diff = abs((conversion_data.timestamp - post.posted_time).total_seconds())
            max_time_diff = self.default_attribution_window_hours * 3600  # Convert to seconds
            
            # Higher confidence for purchases closer to post time
            time_proximity_score = max(0, (max_time_diff - time_diff) / max_time_diff)
            confidence_factors.append(time_proximity_score * self.attribution_confidence_factors['time_proximity'])
            
            # UTM parameter matching
            metadata = conversion_data.metadata or {}
            
            if metadata.get('utm_source') == 'social' or metadata.get('utm_campaign'):
                confidence_factors.append(self.attribution_confidence_factors['utm_parameter_match'])
            
            # Referrer matching
            referrer = metadata.get('referrer', '')
            
            if post.platform in referrer.lower():
                confidence_factors.append(self.attribution_confidence_factors['referrer_match'])
            
            # Direct link click tracking
            if metadata.get('click_id') or metadata.get('social_click'):
                confidence_factors.append(self.attribution_confidence_factors['direct_link_click'])
            
            # Session continuity
            if metadata.get('session_id') or metadata.get('social_session'):
                confidence_factors.append(self.attribution_confidence_factors['session_tracking'])
            
            # Platform-specific factors
            if post.platform == 'x' and 'twitter.com' in referrer:
                confidence_factors.append(0.8)
            elif post.platform == 'instagram' and 'instagram.com' in referrer:
                confidence_factors.append(0.8)
            elif post.platform == 'tiktok' and 'tiktok.com' in referrer:
                confidence_factors.append(0.8)
            
            # Calculate final confidence score
            if confidence_factors:
                # Use weighted average with higher weight on stronger signals
                final_confidence = sum(confidence_factors) / len(confidence_factors)
                # Cap at 1.0
                final_confidence = min(1.0, final_confidence)
            else:
                final_confidence = self.attribution_confidence_factors['default']
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"Failed to calculate attribution confidence: {e}")
            return self.attribution_confidence_factors['default']

    def validate_temporal_attribution(self, post: Post, conversion_data: ConversionData) -> bool:
        """Validate that conversion occurred within agent's active period + attribution window"""
        try:
            if not post.agent_generation_id:
                self.logger.warning(f"Post {post.id} has no agent_generation_id")
                return False
            
            agent = self.database_session.query(AgentGeneration).get(post.agent_generation_id)
            if not agent:
                self.logger.warning(f"Agent generation {post.agent_generation_id} not found")
                return False
            
            # Get agent's active period
            agent_start = agent.start_date
            agent_end = agent.end_date or datetime.utcnow()  # Use current time if still active
            
            if not agent_start:
                self.logger.warning(f"Agent generation {agent.id} has no start_date")
                return False
            
            # Check if conversion falls within agent period + attribution window
            attribution_window_hours = self.default_attribution_window_hours
            valid_start = agent_start
            valid_end = agent_end + timedelta(hours=attribution_window_hours)
            
            is_valid = valid_start <= conversion_data.timestamp <= valid_end
            
            if not is_valid:
                self.logger.info(f"Conversion {conversion_data.stripe_payment_id} outside agent {agent.id} attribution window")
                self.logger.info(f"Conversion time: {conversion_data.timestamp}, Valid window: {valid_start} to {valid_end}")
            
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Temporal attribution validation failed: {e}")
            return False

    def record_conversion(self, conversion_data: ConversionData) -> List[ConversionEvent]:
        """Record conversion events with attribution to relevant posts"""
        try:
            # Determine customer type
            customer_type = self.determine_customer_type(conversion_data)
            conversion_data.customer_type = customer_type
            
            # Find attributable posts
            attributable_posts = self.find_attributable_posts(conversion_data)
            
            if not attributable_posts:
                self.logger.info(f"No attributable posts found for conversion {conversion_data.stripe_payment_id}")
                return []
            
            recorded_conversions = []
            
            # Create conversion events for each attributable post
            for post, confidence_score in attributable_posts:
                
                # Validate temporal attribution
                if not self.validate_temporal_attribution(post, conversion_data):
                    continue
                
                # Calculate conversion window hours
                time_diff = conversion_data.timestamp - post.posted_time
                conversion_window_hours = max(1, int(time_diff.total_seconds() / 3600))
                
                # Create conversion event
                conversion_event = self.db_manager.record_conversion(
                    post_id=post.id,
                    stripe_payment_id=conversion_data.stripe_payment_id,
                    amount=conversion_data.amount,
                    customer_type=customer_type,
                    conversion_window_hours=conversion_window_hours
                )
                
                # Update attribution confidence
                conversion_event.attribution_confidence = confidence_score
                
                # Store additional metadata
                conversion_event.metadata = {
                    'customer_id': conversion_data.customer_id,
                    'customer_email': conversion_data.customer_email,
                    'currency': conversion_data.currency,
                    'payment_method': conversion_data.payment_method,
                    'original_metadata': conversion_data.metadata
                }
                
                self.database_session.commit()
                recorded_conversions.append(conversion_event)
                
                self.logger.info(f"Recorded conversion: Post {post.id} -> ${conversion_data.amount} (confidence: {confidence_score:.2f})")
            
            return recorded_conversions
            
        except Exception as e:
            self.logger.error(f"Failed to record conversion: {e}")
            self.database_session.rollback()
            return []

    def get_post_conversions(self, post_id: int, validated_only: bool = True) -> List[ConversionEvent]:
        """Get all conversions attributed to a specific post"""
        try:
            query = self.database_session.query(ConversionEvent).filter(
                ConversionEvent.post_id == post_id
            )
            
            if validated_only:
                query = query.filter(ConversionEvent.validated == True)
            
            conversions = query.order_by(ConversionEvent.timestamp.desc()).all()
            return conversions
            
        except Exception as e:
            self.logger.error(f"Failed to get post conversions: {e}")
            return []

    def get_agent_conversions(self, agent_generation_id: int, validated_only: bool = True) -> List[ConversionEvent]:
        """Get all conversions attributed to an agent generation"""
        try:
            query = self.database_session.query(ConversionEvent).join(Post).filter(
                Post.agent_generation_id == agent_generation_id
            )
            
            if validated_only:
                query = query.filter(ConversionEvent.validated == True)
            
            conversions = query.order_by(ConversionEvent.timestamp.desc()).all()
            return conversions
            
        except Exception as e:
            self.logger.error(f"Failed to get agent conversions: {e}")
            return []

    def calculate_conversion_metrics(self, agent_generation_id: int) -> Dict[str, Any]:
        """Calculate comprehensive conversion metrics for an agent generation"""
        try:
            conversions = self.get_agent_conversions(agent_generation_id, validated_only=True)
            
            if not conversions:
                return {
                    'total_conversions': 0,
                    'total_revenue': 0.0,
                    'average_order_value': 0.0,
                    'conversion_rate': 0.0,
                    'customer_breakdown': {'new': 0, 'returning': 0, 'unknown': 0},
                    'platform_breakdown': {},
                    'conversion_timeline': []
                }
            
            # Calculate totals
            total_revenue = sum(c.amount for c in conversions)
            total_conversions = len(conversions)
            average_order_value = total_revenue / total_conversions if total_conversions > 0 else 0
            
            # Customer type breakdown
            customer_breakdown = {'new': 0, 'returning': 0, 'unknown': 0}
            for conversion in conversions:
                customer_type = conversion.customer_type or 'unknown'
                customer_breakdown[customer_type] = customer_breakdown.get(customer_type, 0) + 1
            
            # Platform breakdown
            platform_breakdown = {}
            for conversion in conversions:
                post = conversion.post
                platform = post.platform if post else 'unknown'
                
                if platform not in platform_breakdown:
                    platform_breakdown[platform] = {
                        'conversions': 0,
                        'revenue': 0.0,
                        'avg_confidence': 0.0,
                        'posts_count': 0
                    }
                
                platform_breakdown[platform]['conversions'] += 1
                platform_breakdown[platform]['revenue'] += float(conversion.amount)
                platform_breakdown[platform]['avg_confidence'] += conversion.attribution_confidence
            
            # Calculate averages for platform breakdown
            for platform_data in platform_breakdown.values():
                if platform_data['conversions'] > 0:
                    platform_data['avg_confidence'] /= platform_data['conversions']
                    platform_data['avg_order_value'] = platform_data['revenue'] / platform_data['conversions']
            
            # Conversion timeline
            conversion_timeline = []
            for conversion in sorted(conversions, key=lambda c: c.timestamp):
                conversion_timeline.append({
                    'timestamp': conversion.timestamp.isoformat(),
                    'amount': float(conversion.amount),
                    'post_id': conversion.post_id,
                    'platform': conversion.post.platform if conversion.post else 'unknown',
                    'attribution_confidence': conversion.attribution_confidence,
                    'customer_type': conversion.customer_type
                })
            
            # Calculate conversion rate (conversions per post)
            posts_count = self.database_session.query(Post).filter(
                Post.agent_generation_id == agent_generation_id,
                Post.status == 'posted'
            ).count()
            
            conversion_rate = (total_conversions / posts_count * 100) if posts_count > 0 else 0
            
            return {
                'agent_generation_id': agent_generation_id,
                'total_conversions': total_conversions,
                'total_revenue': float(total_revenue),
                'average_order_value': float(average_order_value),
                'conversion_rate': conversion_rate,
                'customer_breakdown': customer_breakdown,
                'platform_breakdown': platform_breakdown,
                'conversion_timeline': conversion_timeline,
                'posts_with_conversions': len(set(c.post_id for c in conversions)),
                'total_posts': posts_count,
                'calculated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate conversion metrics: {e}")
            return {'error': str(e)}

    def cleanup_old_conversion_data(self, days_to_keep: int = 90):
        """Clean up old conversion data beyond retention period"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            old_conversions = self.database_session.query(ConversionEvent).filter(
                ConversionEvent.timestamp < cutoff_date
            ).all()
            
            count = len(old_conversions)
            
            for conversion in old_conversions:
                self.database_session.delete(conversion)
            
            self.database_session.commit()
            self.logger.info(f"Cleaned up {count} old conversion records")
            
        except Exception as e:
            self.logger.error(f"Conversion data cleanup failed: {e}")
            self.database_session.rollback()

if __name__ == "__main__":
    # Test conversion tracker (would need actual database session)
    print("Conversion tracker initialized successfully")
    
    # Example usage:
    # tracker = ConversionTracker(database_session)
    # conversion_data = tracker.process_stripe_webhook(webhook_payload)
    # conversions = tracker.record_conversion(conversion_data)
    # metrics = tracker.calculate_conversion_metrics(agent_id)