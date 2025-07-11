import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from telegram import Update, Message, User, Chat, MessageEntity
from telegram.ext import ContextTypes
from bot import handle_forwarded_message, extract_url_from_message, generate_telegram_link
from database import Database
from classifier import MessageClassifier
from summarizer import TextSummarizer


class TestMessagePipelineIntegration:
    """End-to-end integration tests for message processing pipeline"""
    
    @pytest.fixture
    def mock_telegram_message(self):
        """Create mock Telegram message with forwarded content"""
        # Create mock forwarded user
        forward_from = User(id=12345, first_name="John", is_bot=False)
        
        # Create mock chat
        chat = Chat(id=-1001234567890, type="supergroup")
        
        # Create mock message with URL entity
        message = Mock(spec=Message)
        message.message_id = 123
        message.chat = chat
        message.from_user = forward_from
        message.text = "Интересная статья о новых технологиях искусственного интеллекта в медицине. https://example.com/ai-medicine-article Исследователи разработали новый алгоритм для диагностики заболеваний."
        # Calculate correct offset for URL
        text = message.text
        url_start = text.find("https://example.com/ai-medicine-article")
        message.entities = [
            MessageEntity(type="url", offset=url_start, length=len("https://example.com/ai-medicine-article"))
        ]
        
        # Set forward information
        message.forward_from = forward_from
        message.forward_from_chat = None
        
        return message
    
    @pytest.fixture
    def mock_update(self, mock_telegram_message):
        """Create mock Telegram update with forwarded message"""
        update = Mock(spec=Update)
        update.message = mock_telegram_message
        return update
    
    @pytest.fixture
    def mock_context(self):
        """Create mock context with bot services"""
        context = Mock(spec=ContextTypes.DEFAULT_TYPE)
        
        # Mock database
        mock_db = Mock(spec=Database)
        mock_db.get_existing_categories.return_value = ['technology', 'news', 'medicine']
        mock_db.save_message.return_value = 456  # Mock message ID
        
        # Mock classifier
        mock_classifier = Mock(spec=MessageClassifier)
        mock_classifier.classify_message.return_value = ('medicine', 0.85)
        
        # Mock summarizer
        mock_summarizer = Mock(spec=TextSummarizer)
        mock_summarizer.summarize_text.return_value = "Исследователи разработали новый ИИ-алгоритм для диагностики заболеваний в медицине."
        
        # Store in bot_data
        context.bot_data = {
            'database': mock_db,
            'classifier': mock_classifier,
            'summarizer': mock_summarizer
        }
        
        return context
    
    @pytest.mark.asyncio
    async def test_full_message_pipeline_success(self, mock_update, mock_context):
        """Test complete message processing pipeline - success case"""
        # Mock reply_text method
        mock_update.message.reply_text = AsyncMock()
        
        # Execute the pipeline
        await handle_forwarded_message(mock_update, mock_context)
        
        # Verify database operations
        mock_db = mock_context.bot_data['database']
        mock_classifier = mock_context.bot_data['classifier']
        mock_summarizer = mock_context.bot_data['summarizer']
        
        # Verify summarizer was called with correct text
        mock_summarizer.summarize_text.assert_called_once()
        call_args = mock_summarizer.summarize_text.call_args[0]
        assert "технологиях искусственного интеллекта" in call_args[0]
        
        # Verify classifier was called with correct text
        mock_classifier.classify_message.assert_called_once()
        call_args = mock_classifier.classify_message.call_args[0]
        assert "технологиях искусственного интеллекта" in call_args[0]
        
        # Verify database save was called with correct parameters
        mock_db.save_message.assert_called_once()
        save_args = mock_db.save_message.call_args[0]
        assert save_args[0] == "https://example.com/ai-medicine-article"  # source_url
        assert "https://t.me/c/" in save_args[1]  # telegram_link
        assert save_args[2] == "Исследователи разработали новый ИИ-алгоритм для диагностики заболеваний в медицине."  # summary
        assert save_args[3] == "medicine"  # category
        
        # Verify success message was sent
        mock_update.message.reply_text.assert_called_once()
        reply_text = mock_update.message.reply_text.call_args[0][0]
        assert "✅ Сообщение обработано" in reply_text
        assert "medicine" in reply_text
        assert "https://example.com/ai-medicine-article" in reply_text
    
    @pytest.mark.asyncio
    async def test_message_pipeline_duplicate_handling(self, mock_update, mock_context):
        """Test pipeline handling of duplicate messages"""
        # Mock duplicate message (database returns None)
        mock_context.bot_data['database'].save_message.return_value = None
        mock_update.message.reply_text = AsyncMock()
        
        # Execute the pipeline
        await handle_forwarded_message(mock_update, mock_context)
        
        # Verify duplicate message response
        mock_update.message.reply_text.assert_called_once()
        reply_text = mock_update.message.reply_text.call_args[0][0]
        assert "⚠️ Сообщение уже было обработано ранее" in reply_text
    
    @pytest.mark.asyncio
    async def test_message_pipeline_error_handling(self, mock_update, mock_context):
        """Test pipeline error handling"""
        # Mock error in classifier
        mock_context.bot_data['classifier'].classify_message.side_effect = Exception("API Error")
        mock_update.message.reply_text = AsyncMock()
        
        # Execute the pipeline
        await handle_forwarded_message(mock_update, mock_context)
        
        # Verify error message was sent
        mock_update.message.reply_text.assert_called_once()
        reply_text = mock_update.message.reply_text.call_args[0][0]
        assert "❌ Ошибка при обработке сообщения" in reply_text
    
    @pytest.mark.asyncio
    async def test_non_forwarded_message_ignored(self, mock_context):
        """Test that non-forwarded messages are ignored"""
        # Create non-forwarded message
        message = Mock(spec=Message)
        message.message_id = 123
        message.chat = Chat(id=12345, type="private")
        message.text = "Regular message"
        message.forward_from = None
        message.forward_from_chat = None
        
        update = Mock(spec=Update)
        update.message = message
        
        # Execute the pipeline
        await handle_forwarded_message(update, mock_context)
        
        # Verify no processing occurred
        mock_context.bot_data['database'].save_message.assert_not_called()
        mock_context.bot_data['classifier'].classify_message.assert_not_called()
        mock_context.bot_data['summarizer'].summarize_text.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_message_without_text_ignored(self, mock_context):
        """Test that messages without text are ignored"""
        # Create forwarded message without text
        forward_from = User(id=12345, first_name="John", is_bot=False)
        message = Mock(spec=Message)
        message.message_id = 123
        message.chat = Chat(id=12345, type="private")
        message.forward_from = forward_from
        message.text = None
        
        update = Mock(spec=Update)
        update.message = message
        
        # Execute the pipeline
        await handle_forwarded_message(update, mock_context)
        
        # Verify no processing occurred
        mock_context.bot_data['database'].save_message.assert_not_called()
    
    def test_extract_url_from_message_entity(self):
        """Test URL extraction from message entities"""
        text = "Check this link: https://example.com/article"
        url_start = text.find("https://example.com/article")
        
        message = Message(
            message_id=123,
            date=None,
            chat=Chat(id=12345, type="private"),
            text=text,
            entities=[
                MessageEntity(type="url", offset=url_start, length=len("https://example.com/article"))
            ]
        )
        
        url = extract_url_from_message(message)
        assert url == "https://example.com/article"
    
    def test_extract_url_from_message_regex(self):
        """Test URL extraction using regex fallback"""
        message = Message(
            message_id=123,
            date=None,
            chat=Chat(id=12345, type="private"),
            text="Check this link: https://example.com/article",
            entities=None
        )
        
        url = extract_url_from_message(message)
        assert url == "https://example.com/article"
    
    def test_extract_url_from_message_no_url(self):
        """Test URL extraction when no URL present"""
        message = Message(
            message_id=123,
            date=None,
            chat=Chat(id=12345, type="private"),
            text="Message without URL",
            entities=None
        )
        
        url = extract_url_from_message(message)
        assert url is None
    
    def test_generate_telegram_link_supergroup(self):
        """Test Telegram link generation for supergroup"""
        message = Message(
            message_id=123,
            date=None,
            chat=Chat(id=-1001234567890, type="supergroup")
        )
        
        link = generate_telegram_link(message)
        assert link == "https://t.me/c/1234567890/123"
    
    def test_generate_telegram_link_regular_chat(self):
        """Test Telegram link generation for regular chat"""
        message = Message(
            message_id=123,
            date=None,
            chat=Chat(id=12345, type="private")
        )
        
        link = generate_telegram_link(message)
        assert link == "https://t.me/c/12345/123"
    
    @pytest.mark.asyncio
    async def test_pipeline_with_multiple_urls(self, mock_context):
        """Test pipeline with message containing multiple URLs"""
        # Create message with multiple URLs
        forward_from = User(id=12345, first_name="John", is_bot=False)
        message = Message(
            message_id=123,
            date=None,
            chat=Chat(id=-1001234567890, type="supergroup"),
            text="Multiple links: https://example.com/first and https://example.com/second",
            entities=[
                MessageEntity(type="url", offset=16, length=25),
                MessageEntity(type="url", offset=46, length=26)
            ]
        )
        message.forward_from = forward_from
        message.reply_text = AsyncMock()
        
        update = Mock(spec=Update)
        update.message = message
        
        # Execute the pipeline
        await handle_forwarded_message(update, mock_context)
        
        # Verify first URL was extracted
        mock_db = mock_context.bot_data['database']
        save_args = mock_db.save_message.call_args[0]
        assert save_args[0] == "https://example.com/first"
    
    @pytest.mark.asyncio
    async def test_pipeline_performance(self, mock_update, mock_context):
        """Test pipeline performance with reasonable execution time"""
        import time
        
        mock_update.message.reply_text = AsyncMock()
        
        start_time = time.time()
        await handle_forwarded_message(mock_update, mock_context)
        end_time = time.time()
        
        # Pipeline should complete quickly (mocked services)
        assert end_time - start_time < 0.1
    
    def test_pipeline_data_flow_integrity(self):
        """Test that data flows correctly through the pipeline"""
        # Test data consistency through the pipeline
        original_text = "Новые технологии в медицине: https://example.com/article"
        
        # Simulate each stage
        # 1. URL extraction
        mock_message = Mock()
        mock_message.text = original_text
        mock_message.entities = [MessageEntity(type="url", offset=32, length=25)]
        
        extracted_url = extract_url_from_message(mock_message)
        assert extracted_url == "https://example.com/article"
        
        # 2. Text should be preserved for summarization and classification
        assert "Новые технологии в медицине" in original_text
        
        # 3. Telegram link generation
        mock_message.chat = Chat(id=-1001234567890, type="supergroup")
        mock_message.message_id = 123
        
        telegram_link = generate_telegram_link(mock_message)
        assert telegram_link == "https://t.me/c/1234567890/123"


class TestPipelineRealWorldScenarios:
    """Test pipeline with real-world message scenarios"""
    
    @pytest.fixture
    def real_world_messages(self):
        """Real-world message examples"""
        return {
            'tech_article': {
                'text': 'Переслано от TechNews\n\nНовый прорыв в области квантовых вычислений: https://techcrunch.com/quantum-breakthrough исследователи из MIT создали более стабильный квантовый процессор.',
                'expected_category': 'technology',
                'expected_url': 'https://techcrunch.com/quantum-breakthrough'
            },
            'news_article': {
                'text': 'Важные новости дня: Президент подписал новый закон о цифровой экономике https://news.gov/digital-economy-law',
                'expected_category': 'news',
                'expected_url': 'https://news.gov/digital-economy-law'
            },
            'review_post': {
                'text': 'Отзыв о новом ресторане: Отличная кухня и обслуживание! Рекомендую всем попробовать их фирменные блюда.',
                'expected_category': 'review',
                'expected_url': None
            },
            'finance_update': {
                'text': 'Курс биткоина вырос на 15% за последние 24 часа https://coindesk.com/bitcoin-surge аналитики прогнозируют дальнейший рост.',
                'expected_category': 'finance',
                'expected_url': 'https://coindesk.com/bitcoin-surge'
            }
        }
    
    @pytest.mark.asyncio
    async def test_pipeline_with_real_world_messages(self, real_world_messages):
        """Test pipeline with various real-world message types"""
        
        for message_type, message_data in real_world_messages.items():
            # Create mock services
            mock_db = Mock(spec=Database)
            mock_db.get_existing_categories.return_value = ['technology', 'news', 'review', 'finance']
            mock_db.save_message.return_value = 123
            
            mock_classifier = Mock(spec=MessageClassifier)
            mock_classifier.classify_message.return_value = (message_data['expected_category'], 0.8)
            
            mock_summarizer = Mock(spec=TextSummarizer)
            mock_summarizer.summarize_text.return_value = f"Краткое резюме: {message_type}"
            
            # Create mock message
            forward_from = User(id=12345, first_name="Test", is_bot=False)
            telegram_message = Message(
                message_id=123,
                date=None,
                chat=Chat(id=-1001234567890, type="supergroup"),
                text=message_data['text']
            )
            telegram_message.forward_from = forward_from
            telegram_message.reply_text = AsyncMock()
            
            # Add URL entities if URL expected
            if message_data['expected_url']:
                url_offset = message_data['text'].find(message_data['expected_url'])
                telegram_message.entities = [
                    MessageEntity(type="url", offset=url_offset, length=len(message_data['expected_url']))
                ]
            
            # Create context
            context = Mock(spec=ContextTypes.DEFAULT_TYPE)
            context.bot_data = {
                'database': mock_db,
                'classifier': mock_classifier,
                'summarizer': mock_summarizer
            }
            
            update = Mock(spec=Update)
            update.message = telegram_message
            
            # Execute pipeline
            await handle_forwarded_message(update, context)
            
            # Verify processing
            mock_summarizer.summarize_text.assert_called_once()
            mock_classifier.classify_message.assert_called_once()
            
            # Verify correct data was saved
            save_args = mock_db.save_message.call_args[0]
            assert save_args[0] == message_data['expected_url']  # source_url
            assert save_args[3] == message_data['expected_category']  # category
            
            # Verify success response
            telegram_message.reply_text.assert_called_once()
            reply_text = telegram_message.reply_text.call_args[0][0]
            assert "✅ Сообщение обработано" in reply_text


class TestPipelineErrorRecovery:
    """Test pipeline error recovery and resilience"""
    
    @pytest.mark.asyncio
    async def test_pipeline_classifier_failure_recovery(self, mock_update, mock_context):
        """Test pipeline recovery from classifier failure"""
        # Mock classifier failure
        mock_context.bot_data['classifier'].classify_message.side_effect = Exception("Classifier failed")
        mock_update.message.reply_text = AsyncMock()
        
        # Execute pipeline
        await handle_forwarded_message(mock_update, mock_context)
        
        # Verify error handling
        mock_update.message.reply_text.assert_called_once()
        reply_text = mock_update.message.reply_text.call_args[0][0]
        assert "❌ Ошибка при обработке сообщения" in reply_text
        
        # Verify no database save occurred
        mock_context.bot_data['database'].save_message.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_pipeline_summarizer_failure_recovery(self, mock_update, mock_context):
        """Test pipeline recovery from summarizer failure"""
        # Mock summarizer failure
        mock_context.bot_data['summarizer'].summarize_text.side_effect = Exception("Summarizer failed")
        mock_update.message.reply_text = AsyncMock()
        
        # Execute pipeline
        await handle_forwarded_message(mock_update, mock_context)
        
        # Verify error handling
        mock_update.message.reply_text.assert_called_once()
        reply_text = mock_update.message.reply_text.call_args[0][0]
        assert "❌ Ошибка при обработке сообщения" in reply_text
    
    @pytest.mark.asyncio
    async def test_pipeline_database_failure_recovery(self, mock_update, mock_context):
        """Test pipeline recovery from database failure"""
        # Mock database failure
        mock_context.bot_data['database'].save_message.side_effect = Exception("Database failed")
        mock_update.message.reply_text = AsyncMock()
        
        # Execute pipeline
        await handle_forwarded_message(mock_update, mock_context)
        
        # Verify error handling
        mock_update.message.reply_text.assert_called_once()
        reply_text = mock_update.message.reply_text.call_args[0][0]
        assert "❌ Ошибка при обработке сообщения" in reply_text


# Performance benchmarks
class TestPipelinePerformance:
    """Performance tests for the message pipeline"""
    
    @pytest.mark.asyncio
    async def test_pipeline_processing_time(self, mock_update, mock_context):
        """Test pipeline processing time meets performance requirements"""
        mock_update.message.reply_text = AsyncMock()
        
        import time
        start_time = time.time()
        await handle_forwarded_message(mock_update, mock_context)
        end_time = time.time()
        
        # With mocked services, should be very fast
        assert end_time - start_time < 0.05
    
    @pytest.mark.asyncio
    async def test_pipeline_memory_usage(self, mock_update, mock_context):
        """Test pipeline doesn't cause memory leaks"""
        import gc
        
        mock_update.message.reply_text = AsyncMock()
        
        # Get initial memory state
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Process multiple messages
        for _ in range(10):
            await handle_forwarded_message(mock_update, mock_context)
        
        # Check memory state
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Should not have significant memory growth
        assert final_objects - initial_objects < 100