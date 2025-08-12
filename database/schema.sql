-- Social Darwin Gödel Machine Database Schema
-- PostgreSQL Database for Social Media Content Distribution System

-- Products table: Store seller product information
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    features TEXT,
    target_audience TEXT,
    base_image_url VARCHAR(512),
    category VARCHAR(100),
    price DECIMAL(10, 2),
    brand_voice TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agent generations table: Track social agent evolution over time
CREATE TABLE agent_generations (
    id SERIAL PRIMARY KEY,
    parent_id INTEGER REFERENCES agent_generations(id),
    code_diff TEXT,
    fitness_score FLOAT,
    engagement_score FLOAT,
    conversion_score FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    total_posts INTEGER DEFAULT 0,
    total_revenue DECIMAL(10, 2) DEFAULT 0.00,
    approval_rate FLOAT DEFAULT 0.0,
    status VARCHAR(50) DEFAULT 'active' -- active, inactive, archived
);

-- Posts table: Store all social media posts with temporal tracking
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    platform VARCHAR(50) NOT NULL, -- 'x', 'tiktok', 'instagram'
    post_id VARCHAR(255), -- External platform post ID
    product_id INTEGER REFERENCES products(id),
    image_url VARCHAR(512),
    caption TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'pending', -- pending, approved, rejected, posted, failed
    approval_status VARCHAR(50) DEFAULT 'pending', -- pending, approved, rejected
    agent_generation_id INTEGER REFERENCES agent_generations(id),
    content_type VARCHAR(50) NOT NULL, -- text-only, text+image, image-only, text+video, video-only
    video_url VARCHAR(512),
    hashtags TEXT[],
    scheduled_time TIMESTAMP,
    posted_time TIMESTAMP
);

-- Engagement metrics table: Real-time social media performance tracking
CREATE TABLE engagement_metrics (
    id SERIAL PRIMARY KEY,
    post_id INTEGER REFERENCES posts(id),
    likes INTEGER DEFAULT 0,
    shares INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    views INTEGER DEFAULT 0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    platform_specific_metrics JSONB -- Store platform-specific data like saves, retweets, etc.
);

-- Conversion events table: Revenue tracking with temporal attribution
CREATE TABLE conversion_events (
    id SERIAL PRIMARY KEY,
    post_id INTEGER REFERENCES posts(id),
    stripe_payment_id VARCHAR(255),
    amount DECIMAL(10, 2) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    attribution_confidence FLOAT DEFAULT 1.0, -- 0.0 to 1.0 confidence score
    customer_type VARCHAR(50) DEFAULT 'unknown', -- new, returning, unknown
    conversion_window_hours INTEGER DEFAULT 72, -- Hours between post and conversion
    validated BOOLEAN DEFAULT false -- Whether temporal attribution is validated
);

-- Agent performance snapshots table: Daily performance aggregation
CREATE TABLE agent_performance_snapshots (
    id SERIAL PRIMARY KEY,
    agent_generation_id INTEGER REFERENCES agent_generations(id),
    snapshot_date DATE NOT NULL,
    daily_posts INTEGER DEFAULT 0,
    daily_revenue DECIMAL(10, 2) DEFAULT 0.00,
    daily_engagement INTEGER DEFAULT 0,
    platform_breakdown JSONB, -- {"x": {"posts": 5, "engagement": 1200}, "tiktok": {...}}
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance optimization
CREATE INDEX idx_posts_timestamp ON posts(timestamp);
CREATE INDEX idx_posts_agent_generation ON posts(agent_generation_id);
CREATE INDEX idx_posts_product ON posts(product_id);
CREATE INDEX idx_posts_platform ON posts(platform);
CREATE INDEX idx_engagement_post_timestamp ON engagement_metrics(post_id, timestamp);
CREATE INDEX idx_conversion_post_timestamp ON conversion_events(post_id, timestamp);
CREATE INDEX idx_conversion_stripe_id ON conversion_events(stripe_payment_id);
CREATE INDEX idx_agent_generations_parent ON agent_generations(parent_id);
CREATE INDEX idx_agent_performance_date ON agent_performance_snapshots(agent_generation_id, snapshot_date);

-- Functions for temporal attribution validation
CREATE OR REPLACE FUNCTION validate_conversion_attribution()
RETURNS TRIGGER AS $$
BEGIN
    -- Validate that conversion timestamp is within agent's active period + attribution window
    UPDATE conversion_events 
    SET validated = EXISTS (
        SELECT 1 FROM agent_generations ag 
        JOIN posts p ON p.agent_generation_id = ag.id 
        WHERE p.id = NEW.post_id 
        AND NEW.timestamp >= ag.start_date 
        AND NEW.timestamp <= (ag.end_date + INTERVAL '1 hour' * NEW.conversion_window_hours)
    )
    WHERE id = NEW.id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically validate temporal attribution on new conversions
CREATE TRIGGER trigger_validate_conversion_attribution
    AFTER INSERT ON conversion_events
    FOR EACH ROW
    EXECUTE FUNCTION validate_conversion_attribution();

-- Function to update agent performance metrics
CREATE OR REPLACE FUNCTION update_agent_performance()
RETURNS TRIGGER AS $$
BEGIN
    -- Update total posts, revenue, and approval rate for the agent generation
    UPDATE agent_generations 
    SET 
        total_posts = (
            SELECT COUNT(*) FROM posts WHERE agent_generation_id = NEW.agent_generation_id
        ),
        total_revenue = (
            SELECT COALESCE(SUM(ce.amount), 0) 
            FROM conversion_events ce 
            JOIN posts p ON p.id = ce.post_id 
            WHERE p.agent_generation_id = NEW.agent_generation_id AND ce.validated = true
        ),
        approval_rate = (
            SELECT CASE 
                WHEN COUNT(*) = 0 THEN 0.0
                ELSE COUNT(*) FILTER (WHERE approval_status = 'approved') * 1.0 / COUNT(*)
            END
            FROM posts WHERE agent_generation_id = NEW.agent_generation_id
        )
    WHERE id = NEW.agent_generation_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update agent performance on post changes
CREATE TRIGGER trigger_update_agent_performance
    AFTER INSERT OR UPDATE ON posts
    FOR EACH ROW
    EXECUTE FUNCTION update_agent_performance();