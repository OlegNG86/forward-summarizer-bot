import os
import logging
import click
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from database import Database
from classifier import MessageClassifier
from summarizer import TextSummarizer

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def get_telegram_token():
    """Read TELEGRAM_BOT_TOKEN from .env file"""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")
    return token

async def handle_forwarded_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle forwarded messages with full processing pipeline"""
    message = update.message
    
    if not (message.forward_from or message.forward_from_chat):
        return
    
    if not message.text:
        logger.warning("Forwarded message has no text, skipping")
        return
    
    logger.info(f"Processing forwarded message: {message.text[:100]}...")
    
    try:
        # Get database and AI services from context
        db = context.bot_data['database']
        classifier = context.bot_data['classifier']
        summarizer = context.bot_data['summarizer']
        
        # Extract information from message
        text = message.text
        source_url = extract_url_from_message(message)
        telegram_link = generate_telegram_link(message)
        
        # Process with AI
        logger.info("Generating summary...")
        summary = summarizer.summarize_text(text)
        
        logger.info("Classifying message...")
        category, confidence = classifier.classify_message(text)
        
        # Log results
        logger.info(f"Summary: {summary}")
        logger.info(f"Category: {category} (confidence: {confidence})")
        logger.info(f"Source URL: {source_url}")
        logger.info(f"Telegram link: {telegram_link}")
        
        # Save to database
        message_id = db.save_message(source_url, telegram_link, summary, category)
        
        if message_id:
            logger.info(f"Message saved successfully with ID: {message_id}")
            # Send confirmation to user
            await message.reply_text(
                f"✅ Сообщение обработано\n"
                f"📝 Резюме: {summary}\n"
                f"🏷️ Категория: {category}\n"
                f"🔗 Источник: {source_url or 'не найден'}"
            )
        else:
            logger.warning("Message not saved (likely duplicate)")
            await message.reply_text("⚠️ Сообщение уже было обработано ранее")
            
    except Exception as e:
        logger.error(f"Error processing forwarded message: {e}")
        await message.reply_text("❌ Ошибка при обработке сообщения")

def extract_url_from_message(message):
    """Extract URL from message text or entities"""
    if message.entities:
        for entity in message.entities:
            if entity.type == 'url':
                return message.text[entity.offset:entity.offset + entity.length]
    
    # Try to find URL in text with regex
    import re
    url_pattern = r'https?://[^\s]+'
    matches = re.findall(url_pattern, message.text)
    return matches[0] if matches else None

def generate_telegram_link(message):
    """Generate Telegram link for the message"""
    chat_id = str(message.chat.id)
    if chat_id.startswith('-100'):
        # Channel/supergroup
        chat_id = chat_id[4:]  # Remove -100 prefix
    return f"https://t.me/c/{chat_id}/{message.message_id}"

@click.command()
def start():
    """Start the Telegram bot"""
    try:
        token = get_telegram_token()
        logger.info("Starting Telegram bot...")
        
        # Initialize services
        db = Database()
        classifier = MessageClassifier(db)
        summarizer = TextSummarizer()
        logger.info("All services initialized")
        
        application = Application.builder().token(token).build()
        
        # Store services in bot_data for handlers
        application.bot_data['database'] = db
        application.bot_data['classifier'] = classifier
        application.bot_data['summarizer'] = summarizer
        
        # Add handlers
        application.add_handler(MessageHandler(filters.FORWARDED, handle_forwarded_message))
        
        logger.info("Bot is running. Press Ctrl+C to stop.")
        application.run_polling()
        
    except ValueError as e:
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    start()