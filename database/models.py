"""
SQLAlchemy ORM Models for Social Darwin Gödel Machine
Defines database models with relationships for social media content distribution system
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, Boolean, 
    ForeignKey, DECIMAL, ARRAY, JSON, Date, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func

Base = declarative_base()

class Product(Base):
    """Product model for seller-uploaded product information"""
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    features = Column(Text)
    target_audience = Column(Text)
    base_image_url = Column(String(512))
    category = Column(String(100))
    price = Column(DECIMAL(10, 2))
    brand_voice = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    posts = relationship("Post", back_populates="product")

class AgentGeneration(Base):
    """Social agent generation tracking for evolution"""
    __tablename__ = 'agent_generations'
    
    id = Column(Integer, primary_key=True)
    parent_id = Column(Integer, ForeignKey('agent_generations.id'), nullable=True)
    code_diff = Column(Text)
    fitness_score = Column(Float)
    engagement_score = Column(Float)
    conversion_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    total_posts = Column(Integer, default=0)
    total_revenue = Column(DECIMAL(10, 2), default=0.00)
    approval_rate = Column(Float, default=0.0)
    status = Column(String(50), default='active')
    
    # Relationships
    parent = relationship("AgentGeneration", remote_side=[id])
    children = relationship("AgentGeneration", back_populates="parent")
    posts = relationship("Post", back_populates="agent_generation")
    performance_snapshots = relationship("AgentPerformanceSnapshot", back_populates="agent_generation")
    
    @property
    def is_active(self) -> bool:
        """Check if agent is currently active"""
        return self.status == 'active' and (
            self.end_date is None or self.end_date > datetime.utcnow()
        )
    
    @property
    def runtime_duration(self) -> Optional[timedelta]:
        """Calculate agent runtime duration"""
        if not self.start_date:
            return None
        end_time = self.end_date or datetime.utcnow()
        return end_time - self.start_date
    
    def get_conversion_window_range(self, hours: int = 72) -> tuple:
        """Get the valid conversion attribution window for this agent"""
        if not self.start_date:
            return None, None
        
        start = self.start_date
        end = (self.end_date or datetime.utcnow()) + timedelta(hours=hours)
        return start, end

class Post(Base):
    """Social media post with temporal tracking"""
    __tablename__ = 'posts'
    
    id = Column(Integer, primary_key=True)
    platform = Column(String(50), nullable=False)  # 'x', 'tiktok', 'instagram'
    post_id = Column(String(255))  # External platform post ID
    product_id = Column(Integer, ForeignKey('products.id'))
    image_url = Column(String(512))
    caption = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default='pending')
    approval_status = Column(String(50), default='pending')
    agent_generation_id = Column(Integer, ForeignKey('agent_generations.id'))
    content_type = Column(String(50), nullable=False)
    video_url = Column(String(512))
    hashtags = Column(ARRAY(String))
    scheduled_time = Column(DateTime)
    posted_time = Column(DateTime)
    
    # Relationships
    product = relationship("Product", back_populates="posts")
    agent_generation = relationship("AgentGeneration", back_populates="posts")
    engagement_metrics = relationship("EngagementMetrics", back_populates="post")
    conversion_events = relationship("ConversionEvent", back_populates="post")
    
    # Indexes
    __table_args__ = (
        Index('idx_posts_timestamp', 'timestamp'),
        Index('idx_posts_agent_generation', 'agent_generation_id'),
        Index('idx_posts_product', 'product_id'),
        Index('idx_posts_platform', 'platform'),
    )
    
    @property
    def is_approved(self) -> bool:
        """Check if post is approved for publishing"""
        return self.approval_status == 'approved'
    
    @property
    def is_posted(self) -> bool:
        """Check if post has been successfully posted"""
        return self.status == 'posted' and self.posted_time is not None
    
    def get_latest_engagement(self) -> Optional['EngagementMetrics']:
        """Get the most recent engagement metrics for this post"""
        if not self.engagement_metrics:
            return None
        return max(self.engagement_metrics, key=lambda em: em.timestamp)

class EngagementMetrics(Base):
    """Real-time social media engagement tracking"""
    __tablename__ = 'engagement_metrics'
    
    id = Column(Integer, primary_key=True)
    post_id = Column(Integer, ForeignKey('posts.id'))
    likes = Column(Integer, default=0)
    shares = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    views = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow)
    platform_specific_metrics = Column(JSON)  # Store platform-specific data
    
    # Relationships
    post = relationship("Post", back_populates="engagement_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_engagement_post_timestamp', 'post_id', 'timestamp'),
    )
    
    @property
    def total_engagement(self) -> int:
        """Calculate total engagement score"""
        return self.likes + self.shares + self.comments + self.views
    
    @property
    def weighted_engagement(self) -> float:
        """Calculate weighted engagement score (matches plan.md formula)"""
        return (
            self.likes * 1.0 + 
            self.comments * 3.0 + 
            self.shares * 5.0 + 
            (self.platform_specific_metrics or {}).get('saves', 0) * 4.0
        )

class ConversionEvent(Base):
    """Revenue tracking with temporal attribution validation"""
    __tablename__ = 'conversion_events'
    
    id = Column(Integer, primary_key=True)
    post_id = Column(Integer, ForeignKey('posts.id'))
    stripe_payment_id = Column(String(255))
    amount = Column(DECIMAL(10, 2), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    attribution_confidence = Column(Float, default=1.0)
    customer_type = Column(String(50), default='unknown')
    conversion_window_hours = Column(Integer, default=72)
    validated = Column(Boolean, default=False)
    
    # Relationships
    post = relationship("Post", back_populates="conversion_events")
    
    # Indexes
    __table_args__ = (
        Index('idx_conversion_post_timestamp', 'post_id', 'timestamp'),
        Index('idx_conversion_stripe_id', 'stripe_payment_id'),
    )
    
    @property
    def is_within_attribution_window(self) -> bool:
        """Check if conversion is within the valid attribution window"""
        if not self.post or not self.post.agent_generation:
            return False
        
        agent = self.post.agent_generation
        start, end = agent.get_conversion_window_range(self.conversion_window_hours)
        
        if not start or not end:
            return False
            
        return start <= self.timestamp <= end
    
    @property
    def attribution_window_remaining_hours(self) -> Optional[float]:
        """Get remaining hours in attribution window"""
        if not self.post or not self.post.posted_time:
            return None
        
        window_end = self.post.posted_time + timedelta(hours=self.conversion_window_hours)
        if datetime.utcnow() >= window_end:
            return 0.0
        
        remaining = window_end - datetime.utcnow()
        return remaining.total_seconds() / 3600.0

class AgentPerformanceSnapshot(Base):
    """Daily performance aggregation for agent generations"""
    __tablename__ = 'agent_performance_snapshots'
    
    id = Column(Integer, primary_key=True)
    agent_generation_id = Column(Integer, ForeignKey('agent_generations.id'))
    snapshot_date = Column(Date, nullable=False)
    daily_posts = Column(Integer, default=0)
    daily_revenue = Column(DECIMAL(10, 2), default=0.00)
    daily_engagement = Column(Integer, default=0)
    platform_breakdown = Column(JSON)  # Platform-specific metrics
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    agent_generation = relationship("AgentGeneration", back_populates="performance_snapshots")
    
    # Indexes
    __table_args__ = (
        Index('idx_agent_performance_date', 'agent_generation_id', 'snapshot_date'),
    )

# Database utility functions
class DatabaseManager:
    """Database operations manager with business logic"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create_agent_generation(
        self, 
        parent_id: Optional[int] = None,
        code_diff: Optional[str] = None,
        start_immediately: bool = True
    ) -> AgentGeneration:
        """Create new agent generation"""
        agent = AgentGeneration(
            parent_id=parent_id,
            code_diff=code_diff,
            start_date=datetime.utcnow() if start_immediately else None
        )
        self.session.add(agent)
        self.session.commit()
        return agent
    
    def get_active_agent(self) -> Optional[AgentGeneration]:
        """Get currently active agent generation"""
        return self.session.query(AgentGeneration).filter(
            AgentGeneration.status == 'active',
            AgentGeneration.end_date.is_(None) | (AgentGeneration.end_date > datetime.utcnow())
        ).first()
    
    def create_post(
        self,
        platform: str,
        product_id: int,
        content_type: str,
        agent_generation_id: int,
        caption: Optional[str] = None,
        image_url: Optional[str] = None,
        video_url: Optional[str] = None,
        hashtags: Optional[List[str]] = None
    ) -> Post:
        """Create new social media post"""
        post = Post(
            platform=platform,
            product_id=product_id,
            content_type=content_type,
            agent_generation_id=agent_generation_id,
            caption=caption,
            image_url=image_url,
            video_url=video_url,
            hashtags=hashtags or []
        )
        self.session.add(post)
        self.session.commit()
        return post
    
    def record_engagement(
        self,
        post_id: int,
        likes: int = 0,
        shares: int = 0,
        comments: int = 0,
        views: int = 0,
        platform_specific: Optional[Dict[str, Any]] = None
    ) -> EngagementMetrics:
        """Record engagement metrics for a post"""
        metrics = EngagementMetrics(
            post_id=post_id,
            likes=likes,
            shares=shares,
            comments=comments,
            views=views,
            platform_specific_metrics=platform_specific or {}
        )
        self.session.add(metrics)
        self.session.commit()
        return metrics
    
    def record_conversion(
        self,
        post_id: int,
        stripe_payment_id: str,
        amount: Decimal,
        customer_type: str = 'unknown',
        conversion_window_hours: int = 72
    ) -> ConversionEvent:
        """Record revenue conversion with temporal attribution"""
        conversion = ConversionEvent(
            post_id=post_id,
            stripe_payment_id=stripe_payment_id,
            amount=amount,
            customer_type=customer_type,
            conversion_window_hours=conversion_window_hours
        )
        self.session.add(conversion)
        self.session.commit()
        
        # Trigger validation (handled by database trigger in production)
        conversion.validated = conversion.is_within_attribution_window
        self.session.commit()
        
        return conversion
    
    def get_agent_performance(
        self, 
        agent_generation_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get comprehensive performance metrics for an agent generation"""
        agent = self.session.query(AgentGeneration).get(agent_generation_id)
        if not agent:
            return {}
        
        # Base query filters
        post_filter = [Post.agent_generation_id == agent_generation_id]
        if start_date:
            post_filter.append(Post.timestamp >= start_date)
        if end_date:
            post_filter.append(Post.timestamp <= end_date)
        
        # Get posts and metrics
        posts = self.session.query(Post).filter(*post_filter).all()
        total_revenue = self.session.query(func.sum(ConversionEvent.amount)).join(Post).filter(
            *post_filter,
            ConversionEvent.validated == True
        ).scalar() or Decimal('0.00')
        
        total_engagement = self.session.query(
            func.sum(EngagementMetrics.likes + EngagementMetrics.shares + 
                    EngagementMetrics.comments + EngagementMetrics.views)
        ).join(Post).filter(*post_filter).scalar() or 0
        
        return {
            'agent_id': agent_generation_id,
            'total_posts': len(posts),
            'total_revenue': float(total_revenue),
            'total_engagement': total_engagement,
            'approval_rate': agent.approval_rate,
            'fitness_score': agent.fitness_score,
            'runtime_duration': agent.runtime_duration,
            'platform_breakdown': self._get_platform_breakdown(posts),
            'content_type_breakdown': self._get_content_type_breakdown(posts)
        }
    
    def _get_platform_breakdown(self, posts: List[Post]) -> Dict[str, int]:
        """Calculate platform distribution from posts"""
        breakdown = {}
        for post in posts:
            breakdown[post.platform] = breakdown.get(post.platform, 0) + 1
        return breakdown
    
    def _get_content_type_breakdown(self, posts: List[Post]) -> Dict[str, int]:
        """Calculate content type distribution from posts"""
        breakdown = {}
        for post in posts:
            breakdown[post.content_type] = breakdown.get(post.content_type, 0) + 1
        return breakdown