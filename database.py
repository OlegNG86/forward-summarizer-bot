import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                database=os.getenv('DB_NAME', 'telegram_bot'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', ''),
                port=os.getenv('DB_PORT', '5432')
            )
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def get_cursor(self):
        """Get database cursor"""
        if not self.connection or self.connection.closed:
            self.connect()
        return self.connection.cursor(cursor_factory=RealDictCursor)
    
    def get_existing_categories(self):
        """Get all existing category names"""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT name FROM categories ORDER BY name")
            return [row['name'] for row in cursor.fetchall()]
    
    def category_exists(self, category_name):
        """Check if category exists (case-insensitive)"""
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM categories WHERE LOWER(name) = LOWER(%s)",
                (category_name,)
            )
            return cursor.fetchone() is not None
    
    def add_category(self, category_name):
        """Add new category if it doesn't exist (with deduplication check)"""
        # First check if category already exists (case-insensitive)
        if self.category_exists(category_name):
            logger.info(f"Category '{category_name}' already exists, skipping insert")
            return False
        
        # Double-check for similar categories before inserting
        existing_categories = self.get_existing_categories()
        category_lower = category_name.lower()
        
        # Check for exact matches or very similar names
        for existing in existing_categories:
            if existing.lower() == category_lower:
                logger.info(f"Category '{category_name}' matches existing '{existing}', skipping insert")
                return False
            # Check for substring matches that might indicate duplicates
            if (category_lower in existing.lower() or 
                existing.lower() in category_lower) and len(category_lower) > 3:
                logger.warning(f"Similar category found: '{existing}' vs '{category_name}'")
        
        # Insert new category
        with self.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO categories (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
                (category_name,)
            )
            self.connection.commit()
            logger.info(f"Added new category: {category_name}")
            return True
    
    def save_message(self, source_url, telegram_link, summary, category):
        """Save message to database"""
        # Ensure category exists (with deduplication check)
        self.add_category(category)
        
        # Check if this exact message already exists (deduplication)
        if self._message_exists(telegram_link, summary):
            logger.warning(f"Message already exists: {telegram_link}")
            return None
        
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO messages (source_url, telegram_link, summary, category)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (source_url, telegram_link, summary, category)
            )
            message_id = cursor.fetchone()['id']
            self.connection.commit()
            logger.info(f"Saved message with ID: {message_id}")
            return message_id
    
    def _message_exists(self, telegram_link, summary):
        """Check if message with same telegram_link or summary already exists"""
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT 1 FROM messages 
                WHERE telegram_link = %s OR summary = %s
                LIMIT 1
                """,
                (telegram_link, summary)
            )
            return cursor.fetchone() is not None
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")