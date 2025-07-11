-- Database schema for Telegram bot with LLM summarization and categorization

-- Categories table for storing unique category names
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Messages table for storing forwarded messages with summaries and categories
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    source_url TEXT,
    telegram_link TEXT NOT NULL,
    summary TEXT NOT NULL,
    category TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (category) REFERENCES categories(name)
);

-- Index for faster category lookups
CREATE INDEX idx_messages_category ON messages(category);

-- Index for faster date-based queries
CREATE INDEX idx_messages_created_at ON messages(created_at);

-- Insert some initial common categories
INSERT INTO categories (name) VALUES 
    ('general'),
    ('news'),
    ('technology'),
    ('review');