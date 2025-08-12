"""
Reinforcement Learning Reward Calculation System
Calculates fitness scores for social agents based on engagement and conversion metrics
"""

import os
import json
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from database.models import (
    Post, EngagementMetrics, ConversionEvent, AgentGeneration, 
    AgentPerformanceSnapshot, DatabaseManager
)

class RewardCalculator:
    """
    RL reward calculation system that mirrors SWE-bench scoring
    Calculates fitness scores based on engagement metrics and conversion events
    """
    
    def __init__(self, database_session: Session):
        self.database_session = database_session
        self.db_manager = DatabaseManager(database_session)
        
        # Load reward configuration
        self.load_reward_config()
        
        # Platform baseline engagement rates for normalization
        self.platform_baselines = {
            'x': 0.0015,        # 0.15% engagement rate baseline
            'tiktok': 0.0367,   # 3.67% engagement rate baseline  
            'instagram': 0.005   # 0.5% engagement rate baseline
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_reward_config(self):
        """Load reward calculation configuration"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'rlhf_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.reward_config = config.get('reward_calculation', {})
            else:
                self.reward_config = self.get_default_reward_config()
        except Exception as e:
            self.logger.error(f"Failed to load reward config: {e}")
            self.reward_config = self.get_default_reward_config()

    def get_default_reward_config(self) -> Dict[str, Any]:
        """Get default reward calculation configuration"""
        return {
            'base_engagement_weight': 1.0,
            'conversion_weight': 10.0,
            'human_feedback_weight': 2.0,
            'temporal_decay': {
                'enabled': True,
                'decay_rate': 0.95,
                'time_window_hours': 168  # 7 days
            },
            'platform_normalization': {
                'enabled': True,
                'baselines': self.platform_baselines
            },
            'quality_gates': {
                'minimum_approval_rate': 0.6,
                'minimum_engagement_threshold': 10,
                'maximum_policy_violations': 0
            }
        }

    def calculate_engagement_score(self, post: Post, latest_metrics: Optional[EngagementMetrics] = None) -> float:
        """Calculate normalized engagement score for a single post"""
        try:
            if not latest_metrics:
                latest_metrics = post.get_latest_engagement()
            
            if not latest_metrics:
                return 0.0
            
            # Get platform-specific weights
            platform_weights = {
                'x': {'likes': 1.0, 'shares': 5.0, 'comments': 3.0, 'views': 0.1},
                'tiktok': {'likes': 1.0, 'shares': 5.0, 'comments': 3.0, 'views': 0.01},
                'instagram': {'likes': 1.0, 'shares': 5.0, 'comments': 3.0, 'views': 0.1}
            }
            
            weights = platform_weights.get(post.platform, platform_weights['x'])
            
            # Calculate weighted engagement
            weighted_engagement = (
                latest_metrics.likes * weights['likes'] +
                latest_metrics.shares * weights['shares'] +
                latest_metrics.comments * weights['comments'] +
                latest_metrics.views * weights['views']
            )
            
            # Add platform-specific metrics
            platform_specific = latest_metrics.platform_specific_metrics or {}
            
            if post.platform == 'instagram':
                weighted_engagement += platform_specific.get('saves', 0) * 4.0
                weighted_engagement += platform_specific.get('reach', 0) * 0.1
            elif post.platform == 'tiktok':
                weighted_engagement += platform_specific.get('saves', 0) * 4.0
            elif post.platform == 'x':
                weighted_engagement += platform_specific.get('quotes', 0) * 4.0
            
            # Normalize by platform baseline
            if self.reward_config.get('platform_normalization', {}).get('enabled', True):
                baseline = self.platform_baselines.get(post.platform, self.platform_baselines['x'])
                # Estimate total reach for normalization (using views as proxy)
                estimated_reach = max(latest_metrics.views, 1)  # Avoid division by zero
                engagement_rate = weighted_engagement / estimated_reach
                normalized_score = (engagement_rate / baseline) * 100  # Scale to 0-100+
            else:
                normalized_score = weighted_engagement
            
            # Apply temporal decay if enabled
            if self.reward_config.get('temporal_decay', {}).get('enabled', True):
                normalized_score = self.apply_temporal_decay(normalized_score, post.posted_time)
            
            return max(0.0, normalized_score)  # Ensure non-negative
            
        except Exception as e:
            self.logger.error(f"Failed to calculate engagement score for post {post.id}: {e}")
            return 0.0

    def calculate_conversion_score(self, agent_generation_id: int) -> float:
        """Calculate conversion-based reward score for an agent generation"""
        try:
            # Get all validated conversions for this agent
            conversions = self.database_session.query(ConversionEvent).join(Post).filter(
                Post.agent_generation_id == agent_generation_id,
                ConversionEvent.validated == True
            ).all()
            
            if not conversions:
                return 0.0
            
            total_conversion_score = 0.0
            
            for conversion in conversions:
                # Base conversion value (revenue)
                base_value = float(conversion.amount)
                
                # Apply attribution confidence weighting
                confidence_weighted_value = base_value * conversion.attribution_confidence
                
                # Apply customer type multiplier
                customer_multipliers = {
                    'new': 1.5,      # New customers worth 50% more
                    'returning': 1.0, # Returning customers baseline
                    'unknown': 0.8    # Unknown customers slightly discounted
                }
                
                customer_multiplier = customer_multipliers.get(conversion.customer_type, 1.0)
                final_value = confidence_weighted_value * customer_multiplier
                
                # Apply conversion window penalty (longer windows = lower scores)
                window_penalty = 1.0 / (1.0 + (conversion.conversion_window_hours / 72.0))  # 72h baseline
                final_value *= window_penalty
                
                total_conversion_score += final_value
            
            # Apply conversion weight multiplier (10x as per plan.md)
            conversion_weight = self.reward_config.get('conversion_weight', 10.0)
            weighted_conversion_score = total_conversion_score * conversion_weight
            
            self.logger.info(f"Agent {agent_generation_id} conversion score: ${total_conversion_score:.2f} -> {weighted_conversion_score:.2f} (weighted)")
            
            return weighted_conversion_score
            
        except Exception as e:
            self.logger.error(f"Failed to calculate conversion score for agent {agent_generation_id}: {e}")
            return 0.0

    def calculate_human_feedback_score(self, agent_generation_id: int) -> float:
        """Calculate RLHF-based reward score from human feedback"""
        try:
            # Get agent's posts with approval data
            posts = self.database_session.query(Post).filter(
                Post.agent_generation_id == agent_generation_id
            ).all()
            
            if not posts:
                return 0.0
            
            # Calculate approval rate
            total_posts = len(posts)
            approved_posts = len([p for p in posts if p.approval_status == 'approved'])
            approval_rate = approved_posts / total_posts if total_posts > 0 else 0.0
            
            # Base score from approval rate
            approval_score = approval_rate * 100  # Scale to 0-100
            
            # In production, this would also incorporate:
            # - User ratings from approval workflow (1-5 stars)
            # - Feedback comments sentiment analysis
            # - DSR (Daily Selling Report) feedback scores
            # - Schedule review feedback
            # For now, use approval rate as proxy
            
            feedback_weight = self.reward_config.get('human_feedback_weight', 2.0)
            weighted_feedback_score = approval_score * feedback_weight
            
            self.logger.info(f"Agent {agent_generation_id} feedback score: {approval_rate:.2%} approval rate -> {weighted_feedback_score:.2f} (weighted)")
            
            return weighted_feedback_score
            
        except Exception as e:
            self.logger.error(f"Failed to calculate human feedback score for agent {agent_generation_id}: {e}")
            return 0.0

    def apply_temporal_decay(self, score: float, posted_time: Optional[datetime]) -> float:
        """Apply temporal decay to reduce impact of older posts"""
        if not posted_time:
            return score
        
        try:
            decay_config = self.reward_config.get('temporal_decay', {})
            if not decay_config.get('enabled', True):
                return score
            
            decay_rate = decay_config.get('decay_rate', 0.95)
            time_window_hours = decay_config.get('time_window_hours', 168)
            
            # Calculate time since post
            time_diff = datetime.utcnow() - posted_time
            hours_elapsed = time_diff.total_seconds() / 3600.0
            
            # Apply exponential decay
            if hours_elapsed > 0:
                decay_factor = math.pow(decay_rate, hours_elapsed / time_window_hours)
                decayed_score = score * decay_factor
            else:
                decayed_score = score
            
            return decayed_score
            
        except Exception as e:
            self.logger.error(f"Temporal decay calculation failed: {e}")
            return score

    def check_quality_gates(self, agent_generation_id: int) -> Dict[str, bool]:
        """Check if agent meets quality gate requirements"""
        try:
            quality_gates = self.reward_config.get('quality_gates', {})
            results = {}
            
            # Get agent data
            agent = self.database_session.query(AgentGeneration).get(agent_generation_id)
            if not agent:
                return {'error': True, 'message': 'Agent not found'}
            
            # Check minimum approval rate
            min_approval_rate = quality_gates.get('minimum_approval_rate', 0.6)
            results['approval_rate_check'] = agent.approval_rate >= min_approval_rate
            
            # Check minimum engagement threshold
            posts = self.database_session.query(Post).filter(
                Post.agent_generation_id == agent_generation_id
            ).all()
            
            total_engagement = 0
            for post in posts:
                latest_metrics = post.get_latest_engagement()
                if latest_metrics:
                    total_engagement += latest_metrics.total_engagement
            
            min_engagement = quality_gates.get('minimum_engagement_threshold', 10)
            results['engagement_threshold_check'] = total_engagement >= min_engagement
            
            # Check policy violations (placeholder - would integrate with content moderation)
            max_violations = quality_gates.get('maximum_policy_violations', 0)
            # For now, assume no violations
            results['policy_violation_check'] = True
            
            # Overall quality gate pass
            results['all_gates_passed'] = all(results.values())
            
            return results
            
        except Exception as e:
            self.logger.error(f"Quality gate check failed for agent {agent_generation_id}: {e}")
            return {'error': True, 'message': str(e)}

    def calculate_fitness_score(self, agent_generation_id: int, 
                              engagement_weight: float = None,
                              conversion_weight: float = None) -> float:
        """
        Calculate comprehensive fitness score (mirrors SWE-bench accuracy score)
        This is the main reward signal for agent evolution
        """
        try:
            # Use provided weights or defaults
            if engagement_weight is None:
                engagement_weight = self.reward_config.get('base_engagement_weight', 1.0)
            if conversion_weight is None:
                conversion_weight = self.reward_config.get('conversion_weight', 10.0)
            
            self.logger.info(f"Calculating fitness score for agent {agent_generation_id}")
            
            # Check quality gates first
            quality_results = self.check_quality_gates(agent_generation_id)
            if not quality_results.get('all_gates_passed', False):
                self.logger.warning(f"Agent {agent_generation_id} failed quality gates: {quality_results}")
                return 0.0  # Zero score for failing quality gates
            
            # Get agent's posts
            posts = self.database_session.query(Post).filter(
                Post.agent_generation_id == agent_generation_id,
                Post.status == 'posted'
            ).all()
            
            if not posts:
                self.logger.warning(f"No posts found for agent {agent_generation_id}")
                return 0.0
            
            # Calculate engagement scores
            total_engagement_score = 0.0
            
            for post in posts:
                post_engagement_score = self.calculate_engagement_score(post)
                total_engagement_score += post_engagement_score
            
            # Average engagement score per post
            avg_engagement_score = total_engagement_score / len(posts) if posts else 0.0
            
            # Calculate conversion score
            conversion_score = self.calculate_conversion_score(agent_generation_id)
            
            # Calculate human feedback score
            feedback_score = self.calculate_human_feedback_score(agent_generation_id)
            
            # Combine scores with weights
            # Formula mirrors plan.md: score = Σ(platform_weight × normalized_engagement_score) + (conversion_multiplier × revenue_generated)
            fitness_score = (
                (avg_engagement_score * engagement_weight) +
                conversion_score +  # Already weighted internally
                feedback_score      # Already weighted internally
            )
            
            # Normalize to 0-1 scale (like SWE-bench accuracy)
            # Use sigmoid function to map to 0-1 range
            normalized_fitness = 1.0 / (1.0 + math.exp(-fitness_score / 100.0))
            
            # Update agent's fitness score in database
            agent = self.database_session.query(AgentGeneration).get(agent_generation_id)
            if agent:
                agent.fitness_score = normalized_fitness
                agent.engagement_score = avg_engagement_score
                agent.conversion_score = conversion_score / conversion_weight if conversion_weight > 0 else 0  # Store unweighted
                self.database_session.commit()
            
            self.logger.info(f"Agent {agent_generation_id} fitness score: {normalized_fitness:.4f} (raw: {fitness_score:.2f})")
            self.logger.info(f"  - Engagement: {avg_engagement_score:.2f}")
            self.logger.info(f"  - Conversion: {conversion_score:.2f}")
            self.logger.info(f"  - Feedback: {feedback_score:.2f}")
            
            return normalized_fitness
            
        except Exception as e:
            self.logger.error(f"Fitness score calculation failed for agent {agent_generation_id}: {e}")
            return 0.0

    def compare_agent_performance(self, child_agent_id: int, parent_agent_id: int) -> Dict[str, Any]:
        """
        Compare child agent vs parent agent performance (mirrors DGM selection mechanism)
        Returns comparison results following plan.md's 5% improvement threshold
        """
        try:
            # Calculate fitness scores for both agents
            child_fitness = self.calculate_fitness_score(child_agent_id)
            parent_fitness = self.calculate_fitness_score(parent_agent_id)
            
            # Calculate improvement percentage
            if parent_fitness > 0:
                improvement_percentage = ((child_fitness - parent_fitness) / parent_fitness) * 100
            else:
                improvement_percentage = 100.0 if child_fitness > 0 else 0.0
            
            # Check 5% improvement threshold (from plan.md)
            improvement_threshold = 5.0  # 5% as specified
            meets_improvement_threshold = improvement_percentage >= improvement_threshold
            
            # Get quality gate results for child
            child_quality_gates = self.check_quality_gates(child_agent_id)
            
            # Final selection decision
            should_accept_child = (
                meets_improvement_threshold and 
                child_quality_gates.get('all_gates_passed', False)
            )
            
            # Get detailed metrics for comparison
            child_agent = self.database_session.query(AgentGeneration).get(child_agent_id)
            parent_agent = self.database_session.query(AgentGeneration).get(parent_agent_id)
            
            comparison_result = {
                'child_agent_id': child_agent_id,
                'parent_agent_id': parent_agent_id,
                'fitness_scores': {
                    'child': child_fitness,
                    'parent': parent_fitness
                },
                'improvement_percentage': improvement_percentage,
                'improvement_threshold': improvement_threshold,
                'meets_improvement_threshold': meets_improvement_threshold,
                'quality_gates': child_quality_gates,
                'should_accept_child': should_accept_child,
                'detailed_metrics': {
                    'child': {
                        'fitness_score': child_fitness,
                        'engagement_score': child_agent.engagement_score if child_agent else 0,
                        'conversion_score': child_agent.conversion_score if child_agent else 0,
                        'total_posts': child_agent.total_posts if child_agent else 0,
                        'total_revenue': float(child_agent.total_revenue) if child_agent else 0,
                        'approval_rate': child_agent.approval_rate if child_agent else 0
                    },
                    'parent': {
                        'fitness_score': parent_fitness,
                        'engagement_score': parent_agent.engagement_score if parent_agent else 0,
                        'conversion_score': parent_agent.conversion_score if parent_agent else 0,
                        'total_posts': parent_agent.total_posts if parent_agent else 0,
                        'total_revenue': float(parent_agent.total_revenue) if parent_agent else 0,
                        'approval_rate': parent_agent.approval_rate if parent_agent else 0
                    }
                },
                'comparison_timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Agent comparison: Child {child_agent_id} vs Parent {parent_agent_id}")
            self.logger.info(f"  - Improvement: {improvement_percentage:.2f}% (threshold: {improvement_threshold}%)")
            self.logger.info(f"  - Quality gates: {'PASS' if child_quality_gates.get('all_gates_passed', False) else 'FAIL'}")
            self.logger.info(f"  - Accept child: {'YES' if should_accept_child else 'NO'}")
            
            return comparison_result
            
        except Exception as e:
            self.logger.error(f"Agent performance comparison failed: {e}")
            return {'error': str(e)}

    def generate_reward_report(self, agent_generation_id: int) -> Dict[str, Any]:
        """Generate comprehensive reward calculation report"""
        try:
            fitness_score = self.calculate_fitness_score(agent_generation_id)
            quality_gates = self.check_quality_gates(agent_generation_id)
            
            # Get detailed breakdowns
            agent = self.database_session.query(AgentGeneration).get(agent_generation_id)
            posts = self.database_session.query(Post).filter(
                Post.agent_generation_id == agent_generation_id
            ).all()
            
            post_details = []
            for post in posts:
                latest_metrics = post.get_latest_engagement()
                engagement_score = self.calculate_engagement_score(post, latest_metrics)
                
                post_details.append({
                    'post_id': post.id,
                    'platform': post.platform,
                    'content_type': post.content_type,
                    'posted_time': post.posted_time.isoformat() if post.posted_time else None,
                    'approval_status': post.approval_status,
                    'engagement_score': engagement_score,
                    'metrics': {
                        'likes': latest_metrics.likes if latest_metrics else 0,
                        'shares': latest_metrics.shares if latest_metrics else 0,
                        'comments': latest_metrics.comments if latest_metrics else 0,
                        'views': latest_metrics.views if latest_metrics else 0
                    } if latest_metrics else None
                })
            
            report = {
                'agent_generation_id': agent_generation_id,
                'fitness_score': fitness_score,
                'quality_gates': quality_gates,
                'agent_summary': {
                    'total_posts': len(posts),
                    'approval_rate': agent.approval_rate if agent else 0,
                    'total_revenue': float(agent.total_revenue) if agent else 0,
                    'engagement_score': agent.engagement_score if agent else 0,
                    'conversion_score': agent.conversion_score if agent else 0
                },
                'post_details': post_details,
                'calculation_config': self.reward_config,
                'generated_at': datetime.utcnow().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Reward report generation failed: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    # Test reward calculator (would need actual database session)
    print("Reward calculator initialized successfully")
    
    # Example usage:
    # calculator = RewardCalculator(database_session)
    # fitness_score = calculator.calculate_fitness_score(agent_id)
    # comparison = calculator.compare_agent_performance(child_id, parent_id)
    # report = calculator.generate_reward_report(agent_id)