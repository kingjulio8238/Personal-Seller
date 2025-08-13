"""
Database Models for Content Quality Tracking and Analytics
Extends the main database schema with quality validation and improvement tracking
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, Boolean, 
    ForeignKey, DECIMAL, JSON, Index, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func
import enum

# Use existing base or create new one
try:
    from .models import Base, Product, Post
    BASE_EXISTS = True
except ImportError:
    Base = declarative_base()
    BASE_EXISTS = False


class QualityValidationStatus(enum.Enum):
    """Quality validation status"""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"
    IMPROVED = "improved"


class FilterDecision(enum.Enum):
    """Content filter decision"""
    APPROVED = "approved"
    REJECTED = "rejected"
    HUMAN_REVIEW = "human_review"
    NEEDS_REVISION = "needs_revision"
    QUARANTINED = "quarantined"


class ImprovementStatus(enum.Enum):
    """Improvement attempt status"""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


class QualityValidationRecord(Base):
    """Record of content quality validation attempts"""
    __tablename__ = 'quality_validation_records'
    
    id = Column(Integer, primary_key=True)
    content_id = Column(String(255), nullable=False, index=True)
    content_type = Column(String(50), nullable=False)  # 'image', 'video', 'text'
    platform = Column(String(50), nullable=False)
    
    # Foreign key relationships
    product_id = Column(Integer, ForeignKey('products.id'), nullable=True)
    post_id = Column(Integer, ForeignKey('posts.id'), nullable=True)
    
    # Quality scores
    overall_score = Column(Float, nullable=False)
    technical_quality_score = Column(Float)
    aesthetic_appeal_score = Column(Float)
    engagement_potential_score = Column(Float)
    brand_compliance_score = Column(Float)
    platform_optimization_score = Column(Float)
    
    # Validation result
    validation_status = Column(SQLEnum(QualityValidationStatus), nullable=False)
    passed_validation = Column(Boolean, nullable=False)
    requires_human_review = Column(Boolean, default=False)
    
    # Configuration and metadata
    quality_tier = Column(String(50))  # 'premium', 'standard', 'basic'
    validation_config = Column(JSON)
    individual_scores = Column(JSON)  # Detailed score breakdown
    improvement_suggestions = Column(JSON)  # List of suggestions
    
    # Performance tracking
    processing_time = Column(Float)  # seconds
    cost = Column(DECIMAL(10, 4), default=0.0000)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    if BASE_EXISTS:
        product = relationship("Product", back_populates="quality_validations")
        post = relationship("Post", back_populates="quality_validations")
    
    filter_records = relationship("ContentFilterRecord", back_populates="quality_validation")
    improvement_attempts = relationship("ImprovementAttempt", back_populates="quality_validation")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_quality_platform_date', 'platform', 'created_at'),
        Index('idx_quality_score_status', 'overall_score', 'validation_status'),
        Index('idx_quality_content_type', 'content_type', 'created_at'),
    )
    
    def get_score_improvement_potential(self) -> float:
        """Calculate potential for score improvement"""
        return max(0, 90 - self.overall_score)
    
    def get_quality_grade(self) -> str:
        """Get letter grade for quality score"""
        if self.overall_score >= 90:
            return 'A'
        elif self.overall_score >= 80:
            return 'B'
        elif self.overall_score >= 70:
            return 'C'
        elif self.overall_score >= 60:
            return 'D'
        else:
            return 'F'


class ContentFilterRecord(Base):
    """Record of content filtering attempts"""
    __tablename__ = 'content_filter_records'
    
    id = Column(Integer, primary_key=True)
    content_id = Column(String(255), nullable=False, index=True)
    quality_validation_id = Column(Integer, ForeignKey('quality_validation_records.id'))
    
    # Filter decision
    final_decision = Column(SQLEnum(FilterDecision), nullable=False)
    confidence_score = Column(Float, nullable=False)
    
    # Safety and compliance scores
    brand_safety_score = Column(Float)
    toxicity_score = Column(Float)
    spam_score = Column(Float)
    hate_speech_score = Column(Float)
    adult_content_score = Column(Float)
    
    # Rule violations
    triggered_rules = Column(JSON)  # List of triggered rule IDs
    policy_compliance = Column(JSON)  # Platform compliance results
    safety_scores = Column(JSON)  # Detailed safety analysis
    
    # Review and recommendations
    human_review_reasons = Column(JSON)
    improvement_recommendations = Column(JSON)
    
    # Filter configuration
    filter_level = Column(String(50))  # 'strict', 'moderate', 'permissive'
    platform_policies = Column(JSON)
    
    # Performance tracking
    processing_time = Column(Float)
    cost = Column(DECIMAL(10, 4), default=0.0000)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    quality_validation = relationship("QualityValidationRecord", back_populates="filter_records")
    
    # Indexes
    __table_args__ = (
        Index('idx_filter_decision_date', 'final_decision', 'created_at'),
        Index('idx_filter_confidence', 'confidence_score'),
        Index('idx_filter_brand_safety', 'brand_safety_score'),
    )


class ImprovementAttempt(Base):
    """Record of automated content improvement attempts"""
    __tablename__ = 'improvement_attempts'
    
    id = Column(Integer, primary_key=True)
    retry_id = Column(String(255), nullable=False, unique=True, index=True)
    quality_validation_id = Column(Integer, ForeignKey('quality_validation_records.id'))
    
    # Attempt details
    attempt_number = Column(Integer, default=1)
    improvement_status = Column(SQLEnum(ImprovementStatus), nullable=False)
    
    # Improvements applied
    improvements_applied = Column(JSON)  # List of improvement titles
    improvement_types = Column(JSON)  # List of improvement types
    parameter_adjustments = Column(JSON)  # Technical parameters used
    
    # Results
    original_score = Column(Float)
    improved_score = Column(Float)
    score_improvement = Column(Float)  # Can be negative if worse
    success = Column(Boolean, default=False)
    
    # New validation results after improvement
    new_quality_score = Column(Float)
    new_filter_decision = Column(String(50))
    
    # Cost and performance
    processing_time = Column(Float)
    cost = Column(DECIMAL(10, 4), default=0.0000)
    estimated_cost = Column(DECIMAL(10, 4))
    
    # Recommendation for further attempts
    next_retry_recommended = Column(Boolean, default=False)
    next_improvement_plan = Column(JSON)
    
    # Error tracking
    error_message = Column(Text)
    failure_reason = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime)
    
    # Relationships
    quality_validation = relationship("QualityValidationRecord", back_populates="improvement_attempts")
    
    # Indexes
    __table_args__ = (
        Index('idx_improvement_status_date', 'improvement_status', 'created_at'),
        Index('idx_improvement_success', 'success', 'score_improvement'),
        Index('idx_improvement_cost', 'cost'),
    )
    
    def get_improvement_efficiency(self) -> float:
        """Calculate improvement efficiency (score improvement per dollar)"""
        if self.cost and self.cost > 0 and self.score_improvement:
            return float(self.score_improvement) / float(self.cost)
        return 0.0
    
    def get_roi(self) -> float:
        """Calculate return on investment for improvement"""
        if self.score_improvement and self.score_improvement > 0:
            # Simplified ROI calculation - could be enhanced with engagement correlation
            return self.score_improvement * 10  # 10x multiplier for scoring improvement
        return 0.0


class QualityTrendAnalysis(Base):
    """Aggregated quality trend analysis for performance tracking"""
    __tablename__ = 'quality_trend_analysis'
    
    id = Column(Integer, primary_key=True)
    
    # Time period
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False, index=True)
    period_type = Column(String(20))  # 'daily', 'weekly', 'monthly'
    
    # Scope
    platform = Column(String(50), index=True)
    content_type = Column(String(50), index=True)
    quality_tier = Column(String(50))
    
    # Aggregate metrics
    total_validations = Column(Integer, default=0)
    passed_validations = Column(Integer, default=0)
    failed_validations = Column(Integer, default=0)
    human_review_count = Column(Integer, default=0)
    
    # Quality scores
    average_overall_score = Column(Float)
    average_technical_score = Column(Float)
    average_aesthetic_score = Column(Float)
    average_engagement_score = Column(Float)
    average_brand_compliance_score = Column(Float)
    
    # Pass rates
    overall_pass_rate = Column(Float)
    technical_pass_rate = Column(Float)
    aesthetic_pass_rate = Column(Float)
    engagement_pass_rate = Column(Float)
    brand_compliance_pass_rate = Column(Float)
    
    # Filter metrics
    total_filtered = Column(Integer, default=0)
    approved_count = Column(Integer, default=0)
    rejected_count = Column(Integer, default=0)
    filter_pass_rate = Column(Float)
    average_brand_safety_score = Column(Float)
    
    # Improvement metrics
    improvement_attempts = Column(Integer, default=0)
    successful_improvements = Column(Integer, default=0)
    improvement_success_rate = Column(Float)
    average_score_improvement = Column(Float)
    total_improvement_cost = Column(DECIMAL(10, 2), default=0.00)
    
    # Performance metrics
    average_processing_time = Column(Float)
    total_cost = Column(DECIMAL(10, 2), default=0.00)
    cost_per_validation = Column(DECIMAL(10, 4))
    
    # Top issues and improvements
    most_common_failures = Column(JSON)  # List of most common failure reasons
    most_effective_improvements = Column(JSON)  # List of most effective improvement types
    triggered_rules_frequency = Column(JSON)  # Frequency of filter rule violations
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_trend_period_platform', 'period_start', 'period_end', 'platform'),
        Index('idx_trend_content_type', 'content_type', 'period_start'),
        Index('idx_trend_pass_rate', 'overall_pass_rate'),
    )


class QualityBenchmark(Base):
    """Quality benchmarks and targets for different contexts"""
    __tablename__ = 'quality_benchmarks'
    
    id = Column(Integer, primary_key=True)
    
    # Benchmark context
    benchmark_name = Column(String(255), nullable=False, unique=True)
    platform = Column(String(50), index=True)
    content_type = Column(String(50), index=True)
    quality_tier = Column(String(50))
    
    # Target scores
    target_overall_score = Column(Float, nullable=False)
    target_technical_score = Column(Float)
    target_aesthetic_score = Column(Float)
    target_engagement_score = Column(Float)
    target_brand_compliance_score = Column(Float)
    target_platform_optimization_score = Column(Float)
    
    # Pass rate targets
    target_pass_rate = Column(Float, default=0.85)  # 85% default target
    target_filter_pass_rate = Column(Float, default=0.90)  # 90% default target
    
    # Performance targets
    max_processing_time = Column(Float)  # seconds
    max_cost_per_item = Column(DECIMAL(10, 4))
    
    # Improvement targets
    target_improvement_success_rate = Column(Float, default=0.70)
    max_improvement_cost = Column(DECIMAL(10, 2))
    
    # Status and metadata
    active = Column(Boolean, default=True)
    description = Column(Text)
    created_by = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_benchmark_platform_type', 'platform', 'content_type'),
        Index('idx_benchmark_active', 'active'),
    )


# Extend existing models if they exist
if BASE_EXISTS:
    # Add relationships to existing Product model
    def add_quality_relationships():
        # This would be called to add relationships to existing models
        Product.quality_validations = relationship("QualityValidationRecord", back_populates="product")
        Post.quality_validations = relationship("QualityValidationRecord", back_populates="post")


class QualityAnalytics:
    """Quality analytics and reporting service"""
    
    def __init__(self, db_session: Session):
        self.session = db_session
    
    def get_quality_summary(self, days: int = 7, platform: Optional[str] = None,
                          content_type: Optional[str] = None) -> Dict[str, Any]:
        """Get quality summary for specified period"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = self.session.query(QualityValidationRecord).filter(
            QualityValidationRecord.created_at >= cutoff_date
        )
        
        if platform:
            query = query.filter(QualityValidationRecord.platform == platform)
        if content_type:
            query = query.filter(QualityValidationRecord.content_type == content_type)
        
        records = query.all()
        
        if not records:
            return {'message': 'No quality data available for specified period'}
        
        # Calculate metrics
        total_validations = len(records)
        passed_validations = len([r for r in records if r.passed_validation])
        pass_rate = passed_validations / total_validations
        
        avg_overall_score = sum(r.overall_score for r in records) / total_validations
        avg_processing_time = sum(r.processing_time or 0 for r in records) / total_validations
        total_cost = sum(r.cost or 0 for r in records)
        
        # Quality distribution
        grade_distribution = {}
        for record in records:
            grade = record.get_quality_grade()
            grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
        
        return {
            'period_days': days,
            'total_validations': total_validations,
            'pass_rate': pass_rate,
            'average_overall_score': avg_overall_score,
            'average_processing_time': avg_processing_time,
            'total_cost': float(total_cost),
            'cost_per_validation': float(total_cost / total_validations) if total_validations > 0 else 0,
            'grade_distribution': grade_distribution,
            'platform_filter': platform,
            'content_type_filter': content_type
        }
    
    def get_improvement_effectiveness(self, days: int = 30) -> Dict[str, Any]:
        """Get improvement effectiveness analysis"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        attempts = self.session.query(ImprovementAttempt).filter(
            ImprovementAttempt.created_at >= cutoff_date
        ).all()
        
        if not attempts:
            return {'message': 'No improvement data available for specified period'}
        
        total_attempts = len(attempts)
        successful_attempts = len([a for a in attempts if a.success])
        success_rate = successful_attempts / total_attempts
        
        total_cost = sum(a.cost or 0 for a in attempts)
        avg_score_improvement = sum(a.score_improvement or 0 for a in attempts) / total_attempts
        
        # Most effective improvement types
        improvement_types = {}
        for attempt in attempts:
            if attempt.improvements_applied:
                for improvement in attempt.improvements_applied:
                    if improvement not in improvement_types:
                        improvement_types[improvement] = {
                            'count': 0, 'successes': 0, 'total_improvement': 0
                        }
                    
                    improvement_types[improvement]['count'] += 1
                    if attempt.success:
                        improvement_types[improvement]['successes'] += 1
                    improvement_types[improvement]['total_improvement'] += attempt.score_improvement or 0
        
        # Calculate effectiveness for each improvement type
        for improvement, data in improvement_types.items():
            data['success_rate'] = data['successes'] / data['count']
            data['avg_improvement'] = data['total_improvement'] / data['count']
            data['effectiveness'] = data['success_rate'] * data['avg_improvement']
        
        # Sort by effectiveness
        most_effective = sorted(
            improvement_types.items(),
            key=lambda x: x[1]['effectiveness'],
            reverse=True
        )[:10]
        
        return {
            'period_days': days,
            'total_attempts': total_attempts,
            'success_rate': success_rate,
            'average_score_improvement': avg_score_improvement,
            'total_cost': float(total_cost),
            'cost_per_attempt': float(total_cost / total_attempts) if total_attempts > 0 else 0,
            'most_effective_improvements': dict(most_effective)
        }
    
    def get_filter_performance(self, days: int = 7) -> Dict[str, Any]:
        """Get content filter performance analysis"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        filters = self.session.query(ContentFilterRecord).filter(
            ContentFilterRecord.created_at >= cutoff_date
        ).all()
        
        if not filters:
            return {'message': 'No filter data available for specified period'}
        
        total_filtered = len(filters)
        
        # Decision breakdown
        decisions = {}
        for record in filters:
            decision = record.final_decision.value
            decisions[decision] = decisions.get(decision, 0) + 1
        
        approval_rate = decisions.get('approved', 0) / total_filtered
        avg_confidence = sum(r.confidence_score for r in filters) / total_filtered
        avg_brand_safety = sum(r.brand_safety_score or 0 for r in filters) / total_filtered
        
        # Most triggered rules
        rule_counts = {}
        for record in filters:
            if record.triggered_rules:
                for rule_id in record.triggered_rules:
                    rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1
        
        most_triggered = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'period_days': days,
            'total_filtered': total_filtered,
            'decision_breakdown': decisions,
            'approval_rate': approval_rate,
            'average_confidence_score': avg_confidence,
            'average_brand_safety_score': avg_brand_safety,
            'most_triggered_rules': dict(most_triggered)
        }
    
    def create_trend_analysis(self, period_type: str = 'daily') -> None:
        """Create trend analysis records for the specified period"""
        # This would aggregate data and create QualityTrendAnalysis records
        # Implementation would depend on specific business logic
        pass
    
    def get_quality_trends(self, days: int = 30, period_type: str = 'daily') -> Dict[str, Any]:
        """Get quality trends over time"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        trends = self.session.query(QualityTrendAnalysis).filter(
            QualityTrendAnalysis.period_start >= cutoff_date,
            QualityTrendAnalysis.period_type == period_type
        ).order_by(QualityTrendAnalysis.period_start).all()
        
        if not trends:
            return {'message': f'No {period_type} trend data available for specified period'}
        
        # Extract time series data
        dates = [t.period_start.isoformat() for t in trends]
        pass_rates = [t.overall_pass_rate for t in trends]
        avg_scores = [t.average_overall_score for t in trends]
        processing_times = [t.average_processing_time for t in trends]
        costs = [float(t.total_cost) for t in trends]
        
        return {
            'period_days': days,
            'period_type': period_type,
            'data_points': len(trends),
            'time_series': {
                'dates': dates,
                'pass_rates': pass_rates,
                'average_scores': avg_scores,
                'processing_times': processing_times,
                'costs': costs
            }
        }


# Database initialization functions
def create_quality_tables(engine):
    """Create quality tracking tables in database"""
    Base.metadata.create_all(engine)


def drop_quality_tables(engine):
    """Drop quality tracking tables (for testing/cleanup)"""
    Base.metadata.drop_all(engine)


if __name__ == "__main__":
    # Example usage
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create in-memory SQLite database for testing
    engine = create_engine('sqlite:///:memory:', echo=True)
    create_quality_tables(engine)
    
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    # Create sample quality validation record
    validation = QualityValidationRecord(
        content_id="test_content_123",
        content_type="image",
        platform="instagram",
        overall_score=85.5,
        technical_quality_score=88.0,
        aesthetic_appeal_score=83.0,
        validation_status=QualityValidationStatus.PASSED,
        passed_validation=True,
        quality_tier="premium",
        processing_time=45.2,
        cost=Decimal('2.50')
    )
    
    session.add(validation)
    session.commit()
    
    # Test analytics
    analytics = QualityAnalytics(session)
    summary = analytics.get_quality_summary(days=7)
    print(f"Quality Summary: {summary}")
    
    session.close()
    print("Quality tracking database models created and tested successfully!")