# Social Darwin Gödel Machine Implementation Plan

## Overview
Transform the existing Darwin Gödel Machine (DGM) self-improving coding agent framework into a social media distribution system with self-evolving social agents. The system is designed for companies (the "seller") who upload non-professional product photos which are automatically enhanced through AI image editing before being distributed as optimized social media content across X/Twitter, TikTok, and Instagram using reinforcement learning and evolutionary self-improvement.

**Target Use Case**: Companies upload raw product images + product descriptions → AI enhances photos, generates text content & creates videos → Social agent creates diverse content types (text-only, text+image, image-only, text+video, video-only) → **Mandatory human approval** → Distribution across platforms → Self-evolution based on engagement + conversion metrics.

## Core System Architecture

### 1. Content Creation Pipeline (Core Value Proposition)
**Input**: Non-professional product photos + product descriptions uploaded by seller companies

**Image Enhancement Process**: 
- **OpenAI Image Edit API**: Transform raw product photos with professional backgrounds, lighting, styling, and contextual environments
- **Quality Enhancement**: Remove backgrounds, improve lighting, add professional staging
- **Platform Optimization**: Auto-resize/crop for platform requirements (Instagram square, TikTok vertical, X 16:9)

**Video Generation Process**:
- **Google Veo 3 API**: Transform enhanced product images into dynamic videos with motion, transitions, and effects
- **Product Showcasing**: Create videos highlighting product features, usage scenarios, and benefits
- **Platform-Specific Videos**: Generate vertical videos for TikTok/Instagram Reels, horizontal for YouTube/X

**Content Type Generation**:
- **Text-Only**: Product promotional copy using deep product understanding
- **Text + Image**: Enhanced photos with optimized captions
- **Image-Only**: Professional product photos without text overlay
- **Text + Video**: Dynamic product videos with promotional captions
- **Video-Only**: Standalone product showcase videos

**Output**: Complete content portfolio ready for multi-platform social media distribution

### 2. Social Media Integration Layer

**X/Twitter API v2 Integration**:
- **Authentication**: OAuth 1.0a with read+write access (media upload requires OAuth 1.0a, not OAuth 2.0)
- **Required Scopes**: `tweet.read`, `users.read`, `tweet.write`, `media.write`
- **Media Upload Endpoint**: `POST https://api.x.com/2/media/upload` (new v2 endpoint, v1.1 deprecated March 31, 2025)
- **Post Creation**: `POST https://api.x.com/2/tweets` with media attachment IDs
- **Engagement Metrics**: `GET https://api.x.com/2/tweets/{id}?tweet.fields=public_metrics`
- **Rate Limits**: Varies by access tier (Free tier has limited media posting)
- **Implementation**: Tweepy v4 library with v2 endpoint migration

**TikTok Content Posting API Integration**:
- **Authentication**: User access token with `video.publish` scope authorization
- **Video Upload Endpoints**: 
  - Initialize: `POST https://open.tiktokapis.com/v2/post/publish/video/init/`
  - Photo Content: `POST https://open.tiktokapis.com/v2/post/publish/content/init/`
- **Upload Methods**: `FILE_UPLOAD` (local files) or `PULL_FROM_URL` (hosted videos)
- **Chunked Upload**: 5MB-64MB chunks (final chunk up to 128MB), videos <5MB uploaded whole
- **Status Monitoring**: `GET https://open.tiktokapis.com/v2/post/publish/status/fetch/`
- **Rate Limits**: 6 requests per minute per user access token
- **Content Restrictions**: Unaudited clients limited to private viewing until TikTok audit approval
- **Required Scopes**: `user.info.basic`, `video.publish`, `video.insights`, `comment.list`

**Instagram Graph API Integration**:
- **Authentication**: Facebook User access token (Instagram Professional accounts accessed via Facebook)
- **Account Requirements**: Instagram Business/Creator accounts only (not personal accounts)
- **Content Publishing Endpoints**:
  - Create Media Container: `POST /{ig-user-id}/media`
  - Publish Content: `POST /{ig-user-id}/media_publish`
  - Status Check: `GET /{ig-container-id}?fields=status_code`
  - Rate Limit Monitor: `GET /{ig-user-id}/content_publishing_limit`
- **Publishing Process**: Two-stage (create container → publish media)
- **Rate Limits**: 25 posts per 24 hours per account
- **Content Requirements**: Images ≥600px width (recommended 640px+)
- **API Migration**: Legacy endpoints deprecated by April 21, 2025 (migrate to IG User/Media objects)
- **New 2025 Features**: Views metrics across all media types (Reels, Photos, Carousels, Stories)

**Cross-Platform Post Management**:
- **Queue System**: Priority-based scheduling with platform-specific optimal timing
- **Rate Limiting**: Exponential backoff with platform-specific limits and quotas
- **Retry Logic**: Failed post recovery with error classification and reprocessing
- **Security**: OAuth token refresh, secure credential storage, webhook verification
- **Monitoring**: Real-time API health checks, error logging, performance metrics

### 3. Database Schema (PostgreSQL)
```sql
posts (id, platform, post_id, product_id, image_url, caption, timestamp, status, approval_status, agent_generation_id, content_type)
engagement_metrics (post_id, likes, shares, comments, views, timestamp, platform_specific_metrics)  
conversion_events (id, post_id, stripe_payment_id, amount, timestamp, attribution_confidence, customer_type, conversion_window_hours)
products (id, name, description, features, target_audience, base_image_url, category, price, brand_voice)
agent_generations (id, parent_id, code_diff, fitness_score, engagement_score, conversion_score, timestamp, start_date, end_date, total_posts, total_revenue, approval_rate, status)
agent_performance_snapshots (id, agent_generation_id, snapshot_date, daily_posts, daily_revenue, daily_engagement, platform_breakdown)
```

### 4. RL Reward System (Mirrors SWE-bench)
**Fitness Function**: `score = Σ(platform_weight × normalized_engagement_score) + (conversion_multiplier × revenue_generated)`
**Reward Components**:
- Engagement metrics (likes, shares, comments): 1x weight
- Conversion events (Stripe/payment processing): 10x weight ONLY if temporally attributed to agent's active period
- Revenue generated: Direct RL reward proportional to sales amount occurring during agent runtime + 72-hour attribution window
- **Temporal Attribution Rules**: 
  - Conversions must occur between agent start_date and (end_date + 72 hours) to be attributed
  - Stripe webhook timestamps must align with agent's active posting period
  - Cross-reference conversion events with specific posts published by the agent
**Evaluation Episodes**: 10 posts per generation, 24-hour engagement window
**Baseline Comparison**: Current generation vs parent generation average
**Selection Mechanism**: Accept child agent only if ALL criteria are met:
- **Performance Requirement**: `child_score > parent_score * 1.05` (5% improvement threshold)
- **Code Quality Gates**: Child agent code must compile and pass all unit tests
- **Functional Validation**: Child agent must successfully complete at least 3 test posts in sandbox
- **Safety Compliance**: Child agent passes content policy filters with 100% success rate
- **Resource Efficiency**: Child agent memory/CPU usage ≤ parent agent resource consumption
- **API Reliability**: Child agent demonstrates stable API connections (no timeout failures)
- **Backward Compatibility**: Child agent maintains compatibility with existing database schema and tools

### 5. Self-Improvement Architecture
**Self-Evolving Social Agent** (`social_agent.py`): NEW FILE - Agent that generates content AND rewrites its own source code based on engagement + conversion rewards (adapts existing `coding_agent.py`)
**Evolution Loop** (`social_dgm_outer.py`): NEW FILE - Manages self-improvement cycles where agent modifies itself (adapts existing `DGM_outer.py`)
**Docker Isolation** (`self_improve_step.py`): EXISTING FILE - Reuse Docker containerization for safe agent evolution
**Self-Modification Targets**: Content generation algorithms, caption prompts, image editing strategies, posting timing logic, platform selection rules, engagement optimization techniques

### 6. Agent Tool Access (Required Capabilities)
**Coding Tools**: 
- `Read`: Access own source code and configuration files
- `Write`: Create new code modules and configuration files  
- `Edit`: Modify existing source code for self-improvement
- `Bash`: Execute tests and system commands during evolution

**Content Tools**:
- `PostToX`: Publish text/image tweets with hashtags, mentions, and threading capabilities
- `PostToTikTok`: Upload vertical videos with trending sounds, effects, and hashtag optimization
- `PostToInstagram`: Publish square/story posts with Instagram-specific hashtags and location tags
- `GenerateTextContent`: Create product promotional text using deep product understanding (features, benefits, target audience)
- `GenerateProductVideo`: Create dynamic product videos using Google Veo 3 from enhanced product images
- `GetXEngagementMetrics`: Retrieve likes, retweets, replies, impressions from Twitter API v2
- `GetTikTokEngagementMetrics`: Retrieve views, likes, shares, comments from TikTok Business API
- `GetInstagramEngagementMetrics`: Retrieve likes, comments, saves, reach from Meta Graph API
- `GetConversionMetrics`: Monitor Stripe/payment processing webhooks for revenue attribution with temporal alignment
- `ValidateConversionAttribution`: Cross-reference conversion timestamps with agent active periods and post publication times
- `EnhanceProductImageForX`: Optimize photos for Twitter's 16:9 aspect ratio and visual standards
- `EnhanceProductImageForTikTok`: Create vertical 9:16 product videos/images with trending aesthetics
- `EnhanceProductImageForInstagram`: Generate square 1:1 posts and 9:16 stories with Instagram styling
- `CreateVideoForTikTok`: Generate vertical 9:16 product showcase videos using Veo 3
- `CreateVideoForInstagram`: Generate square/story format product videos using Veo 3
- `CreateVideoForX`: Generate horizontal 16:9 product demonstration videos using Veo 3
- `ScheduleXPost`: Queue tweets for optimal engagement times based on audience analytics
- `ScheduleTikTokPost`: Queue videos for peak TikTok engagement windows and trending periods
- `ScheduleInstagramPost`: Queue posts/stories for optimal Instagram engagement times
- `AnalyzeXTrends`: Monitor trending hashtags, topics, and engagement patterns on Twitter
- `AnalyzeTikTokTrends`: Track viral sounds, effects, and hashtag trends on TikTok
- `AnalyzeInstagramTrends`: Monitor trending hashtags, styles, and content formats on Instagram
- `SendApprovalNotification`: Send generated content to user for **mandatory approval** before posting via polished UI with rating/feedback options
- `SendDailyScheduleReview`: Send daily posting plan to user for approval each morning via interactive UI with modification capabilities
- `SendDailySellingReport`: Send daily selling report (DSR) with engagement and conversion TL;DR to user each evening via feedback-enabled UI
- `CollectUserFeedback`: Gather user ratings, comments, and preferences from approval workflows and daily reports for RLHF training
- `UpdateRLHFModel`: Integrate user feedback signals into agent learning to improve content alignment with user preferences
- `QueryDatabase`: Access post history, engagement data, performance analytics, and product knowledge base

### 7. Management Dashboard with Polished Notification UI
**Web Interface**: React frontend with post preview, approval workflow, analytics dashboard, and comprehensive notification response system
**Mobile App**: React Native for on-the-go management and approvals with push notification support
**Features**: Post review queue, engagement analytics, social agent performance metrics, manual override controls

**Polished Notification Response UI** (Mobile + Desktop Support):

**Approval Notification Interface**:
- **Content Preview**: Full-screen preview of generated content with platform-specific formatting (Twitter card, Instagram square, TikTok vertical)
- **Quick Actions**: Large "Approve" and "Reject" buttons with one-tap responses
- **Rating System**: 5-star rating scale for content quality with optional comment field
- **Modification Tools**: Inline text editing, image cropping, hashtag suggestions with real-time preview
- **Batch Approval**: Select multiple posts for bulk approval with consistent rating
- **Smart Scheduling**: Drag-and-drop calendar interface for optimal posting time adjustments

**Daily Schedule Review Interface**:
- **Timeline View**: Visual timeline showing all planned posts for the day with platform icons
- **Content Carousel**: Swipeable card interface for quick review of each scheduled post
- **Approval Actions**: 
  - "Approve All" (green button) - one-tap approval for entire day's schedule
  - "Modify Schedule" (yellow button) - opens drag-and-drop editor
  - "Reject & Regenerate" (red button) - triggers new content generation
- **Feedback Collection**: Quick feedback buttons (👍👎) with optional strategy preference selection
- **Performance Preview**: Estimated engagement scores for each post based on timing and content analysis

**Daily Selling Report (DSR) Interface**:
- **Performance Summary Cards**: Clean, visual cards showing key metrics (engagement, revenue, top performing posts)
- **Interactive Charts**: Touch-friendly charts with zoom/pan for detailed metric exploration
- **Feedback Integration**: 
  - "Love this strategy" / "Need improvement" buttons with explanation fields
  - Strategy preference sliders (more text vs more visuals, posting frequency, platform focus)
  - Content type preference ranking (drag-and-drop ranking of text-only, text+image, etc.)
- **Action Items**: Agent-suggested improvements with user approval/rejection options
- **Revenue Attribution**: Detailed breakdown of sales attributed to specific posts with confidence indicators

**Cross-Platform UI Features**:
- **Push Notifications**: Instant mobile alerts for approval requests with preview thumbnails
- **Offline Mode**: Cache notifications for review when internet connectivity returns
- **Voice Input**: Voice-to-text for feedback comments and content modifications
- **Gesture Navigation**: Swipe gestures for quick approval/rejection on mobile
- **Dark/Light Mode**: Automatic theme switching based on time of day
- **Accessibility**: Full screen reader support, high contrast mode, large text options
- **Multi-language**: Support for interface localization based on user preferences

**Agent Archive Visualization**: Multi-view interface for comprehensive social agent evolution tracking:

**Primary Views**:
- **Timeline View**: Chronological evolution showing agent generations with branching for failed attempts
- **Tree View**: Hierarchical parent-child relationships with visual branches for successful/failed mutations
- **List View**: Tabular data with sortable columns (generation, performance, revenue, runtime)

**Agent Version Detail Panel** (activated on click):
- **Performance Metrics Dashboard**:
  - Total engagement metrics (likes, shares, comments, views) across all platforms
  - Platform-specific engagement breakdown (X vs TikTok vs Instagram performance)
  - Conversion revenue generated ($USD with attribution confidence scores)
  - ROI calculation (revenue per dollar spent on content creation/posting)
  - Engagement rate trends over the agent's active period

- **Operational Analytics**:
  - Agent runtime duration (start date → end date or current if active)
  - Total posts published across all platforms
  - Content type distribution (text-only: X%, text+image: X%, image-only: X%, text+video: X%, video-only: X%)
  - Platform posting distribution (X: X posts, TikTok: X posts, Instagram: X posts)
  - Average posts per day during active period
  - Content approval rate (approved/total generated content)

- **Comparative Performance Analysis**:
  - **Parent Comparison**: Percentage improvement/decline vs parent agent
  - **Fitness Score Evolution**: Current agent fitness vs parent fitness (with 5% improvement threshold indicator)
  - **Revenue Impact**: Total revenue difference from parent agent
  - **Engagement Efficiency**: Engagement per post compared to parent
  - **Content Quality Metrics**: Approval rate comparison, policy violation comparison

- **Technical Evolution Details**:
  - **Code Diff Viewer**: Side-by-side comparison of agent source code vs parent
  - **Mutation Summary**: List of specific code changes made during evolution
  - **Self-Modification Log**: Record of autonomous code improvements with timestamps
  - **Performance Triggers**: What engagement/conversion patterns triggered this evolution

- **Business Intelligence**:
  - **Revenue Attribution**: Breakdown of conversions by post, platform, content type with temporal validation
  - **Agent-Specific Revenue Validation**: Only conversions occurring during agent's active runtime period are attributed
  - **Conversion Window Analysis**: Time between post publication and purchase (tracking 24hr, 48hr, 7-day, 30-day windows)
  - **Revenue Timeline Alignment**: Visual timeline showing agent runtime overlaid with Stripe deposit timestamps
  - **Customer Acquisition**: New customers vs returning customers driven by this agent (temporally validated)
  - **Content ROI**: Most profitable content types and posting strategies with time-bound attribution
  - **Optimal Timing**: Best performing posting schedules discovered by this agent version

**Interactive Features**:
- **Rollback Capability**: One-click reversion to any previous agent version
- **A/B Comparison**: Side-by-side comparison of any two agent versions
- **Performance Forecasting**: Projected performance if this agent continues running
- **Export Reports**: PDF/CSV export of agent performance for business reporting

### 8. Social Agent Sandbox Environment
**Purpose**: Secure isolated environment for social agent self-evolution and RL training (following DGM Docker practices)

**Sandbox Architecture**:
- **Docker Containerization**: Each social agent generation runs in isolated container (reuses existing `self_improve_step.py` and `utils/docker_utils.py`)
- **Resource Limits**: CPU/memory constraints to prevent runaway processes during evolution
- **Network Isolation**: Controlled internet access - only to specified social media APIs and LLM services
- **File System**: Read-only base system, writable workspace for agent code modifications
- **Time Limits**: Maximum execution time per generation (e.g., 30 minutes for training, 24 hours for evaluation)

**RL Training Environment**:
- **Mock Social Platforms**: Simulated X/Twitter, TikTok, Instagram APIs for rapid RL training without rate limits
- **Historical Data Replay**: Pre-loaded engagement datasets for offline RL pre-training 
- **Engagement Simulation**: Realistic engagement patterns based on content features, timing, platform
- **A/B Testing Framework**: Parallel social agent testing with statistical significance validation

**Self-Evolution Process** (following DGM pattern with RLHF integration):
1. **Content Generation**: Social agent creates posts using product knowledge and trend analysis
2. **Mandatory Approval**: Send generated content to user for verification before posting via polished UI
3. **RLHF Learning from Approval**: Agent learns from user approval/rejection patterns, comments, and feedback ratings to improve content generation algorithms
4. **Content Distribution**: Post approved content across platforms with optimal timing
5. **Performance Collection**: Gather engagement metrics and conversion events (Stripe webhooks)
6. **Daily Reports**: Send scheduling preview (morning) and daily selling report/DSR (evening) to user via interactive UI
7. **RLHF Learning from DSR Feedback**: Agent learns from user responses to scheduling notifications and DSR feedback (thumbs up/down, comments, strategy preferences) to optimize posting strategies
8. **Performance Analysis**: Agent diagnoses areas for improvement using engagement + conversion data + human feedback signals
9. **Code Mutation**: LLM proposes modifications to social agent's source code incorporating RLHF learnings
10. **Sandbox Testing**: Modified agent tested in isolated environment with mock APIs
11. **Quality Gates Validation**: Child agent must pass ALL acceptance criteria:
    - **Compilation Test**: Code compiles without errors
    - **Unit Test Suite**: All existing tests pass + new functionality tested
    - **Functional Test**: 3 successful test posts in sandbox environment
    - **Safety Validation**: 100% content policy compliance rate
    - **Resource Test**: Memory/CPU usage within parent agent limits
    - **API Stability**: Stable connections to all social media APIs
    - **Compatibility Check**: Maintains database/tool compatibility
    - **RLHF Validation**: Child agent demonstrates improved alignment with user preferences from feedback history
12. **Performance Evaluation**: Real-world posting with small sample size (5 posts minimum)
13. **Selection Decision**: Accept child agent ONLY if performance improvement ≥5% AND all quality gates passed AND RLHF alignment score improved
14. **Deployment**: If accepted, child becomes new active agent; if rejected, revert to parent and log failure reasons

**Safety Mechanisms**:
- **Content Filtering**: All generated content passes through safety filters before posting
- **Rate Limit Enforcement**: Hard limits prevent API abuse during rapid evolution cycles
- **Rollback Capability**: Instant reversion to previous social agent version if issues detected
- **Human Override**: Manual intervention capability for problematic social agent behaviors

### 9. Testing Strategy  
**Unit Tests**: All API integrations, database operations, image processing
**Integration Tests**: End-to-end post creation and tracking workflows
**Mock APIs**: For rapid testing without hitting rate limits
**Performance Tests**: Load testing for concurrent post processing
**Sandbox Tests**: Isolated social agent evolution validation

## File Structure and Implementation Mapping

### Existing DGM Files to Reuse/Adapt:
- `DGM_outer.py` → `social_dgm_outer.py` (main evolution loop)
- `coding_agent.py` → `social_agent.py` (self-improving agent)
- `self_improve_step.py` → REUSE (Docker isolation for agent evolution)
- `llm_withtools.py` → REUSE (LLM interactions with tool access)
- `utils/docker_utils.py` → REUSE (Docker containerization utilities)
- `utils/evo_utils.py` → REUSE (evolutionary algorithm utilities)
- `utils/eval_utils.py` → ADAPT (evaluation utilities for social metrics)
- `swe_bench/harness.py` → `social_media/harness.py` (evaluation harness for social platforms)
- `prompts/self_improvement_prompt.py` → ADAPT (self-improvement prompts for content optimization)
- `analysis/visualize_archive.py` → ADAPT (agent evolution visualization)

### New Files to Create:
- `social_agent.py` - Main self-improving social agent (adapts `coding_agent.py`)
- `social_dgm_outer.py` - Social media evolution loop (adapts `DGM_outer.py`)
- `social_media/harness.py` - Social platform evaluation system (replaces `swe_bench/harness.py`)
- `social_media/platforms/` - Platform-specific API integrations
  - `social_media/platforms/x_api.py` - X/Twitter API integration
  - `social_media/platforms/tiktok_api.py` - TikTok API integration  
  - `social_media/platforms/instagram_api.py` - Instagram API integration
- `content_generation/` - Content creation pipeline
  - `content_generation/image_enhancer.py` - OpenAI Image Edit integration
  - `content_generation/video_generator.py` - Google Veo 3 integration
  - `content_generation/text_generator.py` - LLM-based text content
- `database/` - Data persistence layer
  - `database/models.py` - SQLAlchemy ORM models
  - `database/schema.sql` - PostgreSQL database schema
  - `database/migrations/` - Database migration scripts
- `engagement_tracking/` - Analytics and metrics
  - `engagement_tracking/metrics_collector.py` - Real-time engagement data
  - `engagement_tracking/conversion_tracker.py` - Stripe integration
  - `engagement_tracking/reward_calculator.py` - RL reward computation
- `ui/` - User interfaces
  - `ui/web/` - React web dashboard
  - `ui/mobile/` - React Native mobile app
  - `ui/notifications/` - Push notification system
- `tools/social_tools.py` - Social media posting tools (extends existing `tools/`)
- `prompts/social_prompts.py` - Social content generation prompts
- `tests/social/` - Social system tests (extends existing `tests/`)
- `requirements_social.txt` - Additional dependencies (extends `requirements.txt`)
- `docker/social_agent.dockerfile` - Social agent containerization
- `config/` - Configuration management
  - `config/platforms.json` - Platform-specific settings
  - `config/rlhf_config.json` - RLHF training parameters

## Sub-Agent Task Delegation

### Sub-Agent 1: Database Architecture Specialist
**OBJECTIVE**: Build the complete data storage layer that tracks every post, engagement metric, conversion event, and agent generation.
**MISSION**: Create PostgreSQL database with 6 tables defined in `database/schema.sql`, implement SQLAlchemy ORM models in `database/models.py`, and build database utilities. Create migration scripts in `database/migrations/` for schema updates. Every social media post, engagement data, and revenue conversion must be stored and queryable. Implement connection pooling and query optimization.
**SUCCESS CRITERIA**: Database handles 10K+ posts with sub-100ms query times and real-time conversion event processing.

### Sub-Agent 2: Content Creation Specialist (Images & Videos)
**OBJECTIVE**: Transform seller-uploaded non-professional product photos into high-quality images AND videos for social media.
**MISSION**: Implement `content_generation/image_enhancer.py` (OpenAI Image Edit API) and `content_generation/video_generator.py` (Google Veo 3 API). Create `content_generation/text_generator.py` for LLM-based content. Transform amateur photos into professional-quality images, then convert enhanced images into dynamic product showcase videos. Create comprehensive content pipeline supporting all 5 content types with platform-optimized formats. Handle all API errors and implement content caching.
**SUCCESS CRITERIA**: Transform non-professional product photos into 10 enhanced image variants + 5 video variants per upload in under 60 seconds with 99% API success rate.

### Sub-Agent 3: Social Media API Integration Specialist
**OBJECTIVE**: Enable automated posting to X/Twitter, TikTok, and Instagram with engagement tracking.
**MISSION**: Build unified posting system in `social_media/platforms/` with `x_api.py` (Tweepy), `tiktok_api.py` (TikTok Business API), and `instagram_api.py` (Meta Graph API). Create `tools/social_tools.py` extending existing tools framework. Implement rate limiting, post queuing, and engagement data collection. Every post must return a trackable ID and retrieve engagement metrics.
**SUCCESS CRITERIA**: Post to all 3 platforms simultaneously with zero rate limit violations and real-time engagement tracking.

### Sub-Agent 4: Self-Evolving Social Agent Specialist
**OBJECTIVE**: Build social agent that rewrites its own source code based on engagement + conversion rewards.
**MISSION**: Create `social_agent.py` (adapts existing `coding_agent.py`) that generates content AND uses LLM to propose modifications to its own code. Agent must understand individual products deeply, generate both text-only and image-enhanced posts, implement mandatory human approval workflow, and self-diagnose poor performance. Implement reward-driven self-modification where engagement + conversion metrics trigger source code evolution. Reuse existing `llm_withtools.py` for LLM interactions.
**SUCCESS CRITERIA**: Social agent autonomously rewrites its own content generation logic 10+ times with measurable engagement + conversion improvements after each self-modification.

### Sub-Agent 5: Evolution Loop Specialist
**OBJECTIVE**: Orchestrate the self-evolution cycle where social agent rewrites itself based on engagement rewards with strict quality gates.
**MISSION**: Adapt existing `DGM_outer.py` to create `social_dgm_outer.py` that manages the self-improvement loop with comprehensive validation. Reuse existing evolution utilities from `utils/evo_utils.py` and `utils/eval_utils.py`. Agent proposes code changes to itself, evaluates via engagement metrics, and accepts/rejects modifications only after passing ALL quality gates: compilation, unit tests, functional tests, safety compliance, resource efficiency, API stability, and backward compatibility. Replace SWE-bench harness (`swe_bench/harness.py`) with social media engagement evaluation system. Implement fitness evaluation using engagement rewards with 5% improvement threshold PLUS quality gate requirements.
**SUCCESS CRITERIA**: Complete self-evolution pipeline where social agent autonomously improves its own source code over 10+ generations with 100% quality gate compliance and zero production failures from child agents.

### Sub-Agent 6: Frontend Dashboard Specialist
**OBJECTIVE**: Build web and mobile interfaces for human oversight and analytics.
**MISSION**: Build `ui/web/` (React dashboard) and `ui/mobile/` (React Native app) with post review queue, engagement analytics visualization, and agent performance monitoring. Create `ui/notifications/` for push notification system. Adapt existing `analysis/visualize_archive.py` for agent evolution display. Include manual post approval workflow and real-time metrics display.
**SUCCESS CRITERIA**: Non-technical users can review posts and understand agent performance within 30 seconds of opening the interface.

### Sub-Agent 7: Engagement Analytics Specialist
**OBJECTIVE**: Track social media performance AND revenue conversions with precise temporal attribution to calculate comprehensive RL rewards.
**MISSION**: Build `engagement_tracking/metrics_collector.py` for real-time engagement collection, `engagement_tracking/conversion_tracker.py` for Stripe integration, and `engagement_tracking/reward_calculator.py` for RL computation. Create temporal attribution system ensuring conversions are only attributed to agents active during the conversion window. Implement conversion window tracking (24hr, 48hr, 7-day, 30-day) and visual timeline alignment. Create automated daily reporting system with RLHF feedback collection, and feed temporally-validated rewards + human feedback signals into RL calculation.
**SUCCESS CRITERIA**: Real-time engagement data, temporally-validated conversion attribution with 99.9% timestamp accuracy, platform-normalized scoring, daily automated user reports, and predictive analytics for post performance.

### Sub-Agent 8: Testing Infrastructure Specialist
**OBJECTIVE**: Ensure system reliability through comprehensive automated testing.
**MISSION**: Extend existing `tests/` directory with `tests/social/` for social system tests. Build unit tests for all functions, integration tests for API workflows, mock implementations for development, and performance tests for concurrent operations. Reuse existing `tests/conftest.py` and testing utilities. Set up CI/CD pipeline with 90%+ code coverage requirement.
**SUCCESS CRITERIA**: All critical paths tested with automated test suite completing in under 5 minutes.

### Sub-Agent 9: DevOps and Deployment Specialist
**OBJECTIVE**: Deploy and operate the system in production with 99.9% uptime.
**MISSION**: Create `docker/social_agent.dockerfile` and deployment scripts. Reuse existing Docker utilities from `utils/docker_utils.py` and `self_improve_step.py` for agent isolation. Implement monitoring/logging, set up backups, and handle API key security. Build secure sandbox environment for social agent evolution following DGM's Docker isolation practices. Ensure system can scale and recover from failures automatically.
**SUCCESS CRITERIA**: Zero-downtime deployments with automated scaling, complete disaster recovery within 15 minutes, and fully isolated sandbox environment for safe social agent self-evolution.

## Implementation Timeline

### Phase 1: Core Infrastructure  
**Primary Sub-Agents**: Database Architecture (#1), DevOps (#10)
**Supporting Sub-Agents**: Testing Infrastructure (#8)
- Database setup and basic infrastructure
- Development environment configuration
- Basic testing framework establishment

### Phase 2: Content Pipeline  
**Primary Sub-Agents**: OpenAI Integration (#2), Content Generation (#7)
**Supporting Sub-Agents**: Testing Infrastructure (#8)
- Image generation and editing capabilities
- Content creation and optimization systems
- Initial testing of content pipeline

### Phase 3: Social Media Integration  
**Primary Sub-Agents**: Social Media API (#3), Engagement Analytics (#9)
**Supporting Sub-Agents**: Testing Infrastructure (#8), DevOps (#10)
- Platform API integrations
- Posting and engagement tracking systems
- Performance monitoring and optimization

### Phase 4: RL and Evolution  
**Primary Sub-Agents**: Self-Evolving Social Agent (#4), Evolution Loop (#5)
**Supporting Sub-Agents**: Testing Infrastructure (#8), DevOps (#10)
- Sandbox environment setup for social agent evolution
- Self-improvement mechanisms and RL training environment
- Reward system implementation with mock social platforms
- Evolution loop integration and isolated testing

### Phase 5: Dashboard and Deployment  
**Primary Sub-Agents**: Frontend Dashboard (#6), DevOps (#10)
**Supporting Sub-Agents**: Testing Infrastructure (#8)
- Web and mobile interface development
- Production deployment and monitoring
- Final integration testing and optimization

## Success Metrics
- **Technical**: All tests passing, sub-1s response times, 99.9% uptime
- **Business**: 15% improvement in engagement metrics within first month
- **Operational**: Zero human intervention required for 80% of posts
- **Evolution**: Measurable self-improvement in agent performance over generations

## Risk Mitigation
- **API Rate Limits**: Comprehensive queuing and backoff strategies
- **Platform Policy Violations**: Content moderation and human review workflows  
- **Performance Degradation**: Continuous monitoring and automatic scaling
- **Data Loss**: Regular backups and redundant storage systems