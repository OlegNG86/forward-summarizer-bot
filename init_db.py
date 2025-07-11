#!/usr/bin/env python3
"""
Database initialization script for Telegram bot
Note: In Docker setup, database is created automatically via docker-compose
"""
import os
import sys
import logging
import time
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def wait_for_postgres():
    """Wait for PostgreSQL to be ready (for Docker setup)"""
    max_retries = 30
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                database=os.getenv('DB_NAME', 'telegram_bot'),
                user=os.getenv('DB_USER', 'bot_user'),
                password=os.getenv('DB_PASSWORD', 'bot_password'),
                port=os.getenv('DB_PORT', '5432')
            )
            conn.close()
            logger.info("PostgreSQL is ready!")
            return True
        except psycopg2.OperationalError:
            logger.info(f"Waiting for PostgreSQL... (attempt {attempt + 1}/{max_retries})")
            time.sleep(2)
    
    logger.error("PostgreSQL is not ready after 60 seconds")
    return False

def create_database():
    """Create database if it doesn't exist (for non-Docker setup)"""
    try:
        # Connect to PostgreSQL server (not to specific database)
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            user=os.getenv('DB_USER', 'bot_user'),
            password=os.getenv('DB_PASSWORD', 'bot_password'),
            port=os.getenv('DB_PORT', '5432')
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        db_name = os.getenv('DB_NAME', 'telegram_bot')
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f"CREATE DATABASE {db_name}")
            logger.info(f"Database '{db_name}' created successfully")
        else:
            logger.info(f"Database '{db_name}' already exists")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        sys.exit(1)

def init_tables():
    """Initialize database tables"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'telegram_bot'),
            user=os.getenv('DB_USER', 'bot_user'),
            password=os.getenv('DB_PASSWORD', 'bot_password'),
            port=os.getenv('DB_PORT', '5432')
        )
        cursor = conn.cursor()
        
        # Read and execute schema.sql
        with open('schema.sql', 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        cursor.execute(schema_sql)
        conn.commit()
        
        logger.info("Database tables initialized successfully")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error initializing tables: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logger.info("Starting database initialization...")
    
    # Check if we're in Docker (schema.sql is auto-applied in Docker)
    is_docker = os.getenv('DB_HOST') == 'postgres'
    
    if is_docker:
        logger.info("Docker setup detected - waiting for PostgreSQL...")
        if not wait_for_postgres():
            sys.exit(1)
        logger.info("Database initialization completed (schema applied via Docker)!")
    else:
        # Non-Docker setup
        if not os.path.exists('.env'):
            logger.error(".env file not found. Please create it from .env.example")
            sys.exit(1)
        
        if not os.path.exists('schema.sql'):
            logger.error("schema.sql file not found")
            sys.exit(1)
        
        create_database()
        init_tables()
        logger.info("Database initialization completed successfully!")