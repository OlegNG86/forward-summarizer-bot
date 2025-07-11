import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from telegram import Update, Message, User, Chat, MessageEntity
from telegram.ext import ContextTypes
from bot import handle_forwarded_message, extract_url_from_message, generate_telegram_link


class TestMessagePipelineVerification:
    """Simplified end-to-end pipeline verification tests"""
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_success_flow(self):
        """Test complete pipeline: forwarded message -> summarization -> classification -> database save"""
        
        # 1. Create mock forwarded message
        forward_from = User(id=12345, first_name="John", is_bot=False)
        chat = Chat(id=-1001234567890, type="supergroup")
        
        message = Mock()
        message.message_id = 123
        message.chat = chat
        message.text = "Статья о новых технологиях ИИ: https://example.com/ai-article Революция в медицине"
        message.entities = [
            Mock(type="url", offset=35, length=28)  # Mock entity
        ]
        message.forward_from = forward_from
        message.forward_from_chat = None
        message.reply_text = AsyncMock()
        
        # 2. Create mock services
        mock_db = Mock()
        mock_db.get_existing_categories.return_value = ['technology', 'medicine']
        mock_db.save_message.return_value = 456  # Success
        
        mock_classifier = Mock()
        mock_classifier.classify_message.return_value = ('technology', 0.9)
        
        mock_summarizer = Mock()
        mock_summarizer.summarize_text.return_value = "Новые технологии ИИ в медицине"
        
        # 3. Create context with services
        context = Mock()
        context.bot_data = {
            'database': mock_db,
            'classifier': mock_classifier,
            'summarizer': mock_summarizer
        }
        
        # 4. Create update
        update = Mock()
        update.message = message
        
        # 5. Execute pipeline
        await handle_forwarded_message(update, context)
        
        # 6. Verify pipeline execution
        # Summarizer should be called with message text
        mock_summarizer.summarize_text.assert_called_once()
        summarizer_call_args = mock_summarizer.summarize_text.call_args[0][0]
        assert "технологиях ИИ" in summarizer_call_args
        
        # Classifier should be called with message text
        mock_classifier.classify_message.assert_called_once()
        classifier_call_args = mock_classifier.classify_message.call_args[0][0]
        assert "технологиях ИИ" in classifier_call_args
        
        # Database should save the message
        mock_db.save_message.assert_called_once()
        db_call_args = mock_db.save_message.call_args[0]
        assert len(db_call_args) == 4
        # source_url, telegram_link, summary, category
        assert db_call_args[2] == "Новые технологии ИИ в медицине"  # summary
        assert db_call_args[3] == "technology"  # category
        
        # Success message should be sent
        message.reply_text.assert_called_once()
        reply_text = message.reply_text.call_args[0][0]
        assert "✅ Сообщение обработано" in reply_text
        assert "technology" in reply_text
    
    @pytest.mark.asyncio
    async def test_pipeline_duplicate_message_handling(self):
        """Test pipeline correctly handles duplicate messages"""
        
        # Create mock message
        message = Mock()
        message.message_id = 123
        message.chat = Chat(id=-1001234567890, type="supergroup")
        message.text = "Duplicate message text"
        message.entities = []
        message.forward_from = User(id=12345, first_name="John", is_bot=False)
        message.forward_from_chat = None
        message.reply_text = AsyncMock()
        
        # Mock services - database returns None (duplicate)
        mock_db = Mock()
        mock_db.save_message.return_value = None  # Duplicate detected
        
        mock_classifier = Mock()
        mock_classifier.classify_message.return_value = ('general', 0.8)
        
        mock_summarizer = Mock()
        mock_summarizer.summarize_text.return_value = "Duplicate summary"
        
        context = Mock()
        context.bot_data = {
            'database': mock_db,
            'classifier': mock_classifier,
            'summarizer': mock_summarizer
        }
        
        update = Mock()
        update.message = message
        
        # Execute pipeline
        await handle_forwarded_message(update, context)
        
        # Verify processing still occurred
        mock_summarizer.summarize_text.assert_called_once()
        mock_classifier.classify_message.assert_called_once()
        mock_db.save_message.assert_called_once()
        
        # Verify duplicate message was sent
        message.reply_text.assert_called_once()
        reply_text = message.reply_text.call_args[0][0]
        assert "⚠️ Сообщение уже было обработано ранее" in reply_text
    
    @pytest.mark.asyncio
    async def test_pipeline_error_recovery(self):
        """Test pipeline handles errors gracefully"""
        
        # Create mock message
        message = Mock()
        message.message_id = 123
        message.chat = Chat(id=-1001234567890, type="supergroup")
        message.text = "Error test message"
        message.entities = []
        message.forward_from = User(id=12345, first_name="John", is_bot=False)
        message.forward_from_chat = None
        message.reply_text = AsyncMock()
        
        # Mock services - classifier throws error
        mock_db = Mock()
        mock_classifier = Mock()
        mock_classifier.classify_message.side_effect = Exception("Classification failed")
        mock_summarizer = Mock()
        mock_summarizer.summarize_text.return_value = "Error summary"
        
        context = Mock()
        context.bot_data = {
            'database': mock_db,
            'classifier': mock_classifier,
            'summarizer': mock_summarizer
        }
        
        update = Mock()
        update.message = message
        
        # Execute pipeline
        await handle_forwarded_message(update, context)
        
        # Verify error message was sent
        message.reply_text.assert_called_once()
        reply_text = message.reply_text.call_args[0][0]
        assert "❌ Ошибка при обработке сообщения" in reply_text
        
        # Verify database save was not called due to error
        mock_db.save_message.assert_not_called()
    
    def test_url_extraction_functionality(self):
        """Test URL extraction works correctly"""
        
        # Test with entity
        message = Mock()
        message.text = "Check this: https://example.com/test"
        url_start = message.text.find("https://example.com/test")
        message.entities = [Mock(type="url", offset=url_start, length=len("https://example.com/test"))]
        
        url = extract_url_from_message(message)
        assert url == "https://example.com/test"
        
        # Test with regex fallback
        message_no_entity = Mock()
        message_no_entity.text = "Check this: https://example.com/test"
        message_no_entity.entities = None
        
        url = extract_url_from_message(message_no_entity)
        assert url == "https://example.com/test"
        
        # Test no URL
        message_no_url = Mock()
        message_no_url.text = "No URL here"
        message_no_url.entities = None
        
        url = extract_url_from_message(message_no_url)
        assert url is None
    
    def test_telegram_link_generation(self):
        """Test Telegram link generation"""
        
        # Test supergroup
        message = Mock()
        message.message_id = 123
        message.chat = Mock()
        message.chat.id = -1001234567890
        
        link = generate_telegram_link(message)
        assert link == "https://t.me/c/1234567890/123"
        
        # Test regular chat
        message_regular = Mock()
        message_regular.message_id = 456
        message_regular.chat = Mock()
        message_regular.chat.id = 12345
        
        link = generate_telegram_link(message_regular)
        assert link == "https://t.me/c/12345/456"
    
    @pytest.mark.asyncio
    async def test_pipeline_data_integrity(self):
        """Test data flows correctly through pipeline without corruption"""
        
        # Original message data
        original_text = "Важные новости: https://news.example.com/breaking новый закон принят"
        original_url = "https://news.example.com/breaking"
        expected_summary = "Краткое резюме новостей"
        expected_category = "news"
        
        # Create message
        message = Mock()
        message.message_id = 789
        message.chat = Mock()
        message.chat.id = -1001234567890
        message.text = original_text
        message.entities = []
        message.forward_from = User(id=12345, first_name="John", is_bot=False)
        message.forward_from_chat = None
        message.reply_text = AsyncMock()
        
        # Mock services
        mock_db = Mock()
        mock_db.save_message.return_value = 789
        
        mock_classifier = Mock()
        mock_classifier.classify_message.return_value = (expected_category, 0.85)
        
        mock_summarizer = Mock()
        mock_summarizer.summarize_text.return_value = expected_summary
        
        context = Mock()
        context.bot_data = {
            'database': mock_db,
            'classifier': mock_classifier,
            'summarizer': mock_summarizer
        }
        
        update = Mock()
        update.message = message
        
        # Execute pipeline
        await handle_forwarded_message(update, context)
        
        # Verify data integrity
        # 1. Summarizer received original text
        summarizer_args = mock_summarizer.summarize_text.call_args[0]
        assert summarizer_args[0] == original_text
        
        # 2. Classifier received original text
        classifier_args = mock_classifier.classify_message.call_args[0]
        assert classifier_args[0] == original_text
        
        # 3. Database received processed data
        db_args = mock_db.save_message.call_args[0]
        assert db_args[1] == "https://t.me/c/1234567890/789"  # telegram_link
        assert db_args[2] == expected_summary  # summary
        assert db_args[3] == expected_category  # category
        
        # 4. User received confirmation with correct data
        reply_args = message.reply_text.call_args[0]
        assert expected_summary in reply_args[0]
        assert expected_category in reply_args[0]
    
    @pytest.mark.asyncio
    async def test_non_forwarded_messages_ignored(self):
        """Test non-forwarded messages are ignored"""
        
        # Create non-forwarded message
        message = Mock()
        message.message_id = 123
        message.chat = Chat(id=12345, type="private")
        message.text = "Regular message"
        message.forward_from = None
        message.forward_from_chat = None
        
        # Mock services
        mock_db = Mock()
        mock_classifier = Mock()
        mock_summarizer = Mock()
        
        context = Mock()
        context.bot_data = {
            'database': mock_db,
            'classifier': mock_classifier,
            'summarizer': mock_summarizer
        }
        
        update = Mock()
        update.message = message
        
        # Execute pipeline
        await handle_forwarded_message(update, context)
        
        # Verify no processing occurred
        mock_summarizer.summarize_text.assert_not_called()
        mock_classifier.classify_message.assert_not_called()
        mock_db.save_message.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_empty_text_messages_ignored(self):
        """Test messages without text are ignored"""
        
        # Create message without text
        message = Mock()
        message.message_id = 123
        message.chat = Chat(id=12345, type="private")
        message.text = None
        message.forward_from = User(id=12345, first_name="John", is_bot=False)
        message.forward_from_chat = None
        
        # Mock services
        mock_db = Mock()
        mock_classifier = Mock()
        mock_summarizer = Mock()
        
        context = Mock()
        context.bot_data = {
            'database': mock_db,
            'classifier': mock_classifier,
            'summarizer': mock_summarizer
        }
        
        update = Mock()
        update.message = message
        
        # Execute pipeline
        await handle_forwarded_message(update, context)
        
        # Verify no processing occurred
        mock_summarizer.summarize_text.assert_not_called()
        mock_classifier.classify_message.assert_not_called()
        mock_db.save_message.assert_not_called()


# Integration test with all components
class TestPipelineIntegration:
    """Integration tests that verify component interactions"""
    
    @pytest.mark.asyncio
    async def test_full_integration_with_real_flow(self):
        """Test full integration simulating real message processing"""
        
        # Real-world message example
        message_text = """
        Переслано от TechNews
        
        Прорыв в области квантовых вычислений: https://techcrunch.com/quantum-breakthrough
        
        Исследователи из MIT создали квантовый процессор нового поколения, который может работать 
        при комнатной температуре. Это революционное достижение открывает новые возможности 
        для практического применения квантовых технологий.
        """
        
        # Create message
        message = Mock()
        message.message_id = 1001
        message.chat = Mock()
        message.chat.id = -1001234567890
        message.text = message_text.strip()
        message.entities = []
        message.forward_from = User(id=12345, first_name="TechNews", is_bot=False)
        message.forward_from_chat = None
        message.reply_text = AsyncMock()
        
        # Mock realistic service responses
        mock_db = Mock()
        mock_db.get_existing_categories.return_value = ['technology', 'science', 'news']
        mock_db.save_message.return_value = 1001
        
        mock_classifier = Mock()
        mock_classifier.classify_message.return_value = ('technology', 0.92)
        
        mock_summarizer = Mock()
        mock_summarizer.summarize_text.return_value = "MIT создал квантовый процессор для комнатной температуры"
        
        context = Mock()
        context.bot_data = {
            'database': mock_db,
            'classifier': mock_classifier,
            'summarizer': mock_summarizer
        }
        
        update = Mock()
        update.message = message
        
        # Execute pipeline
        await handle_forwarded_message(update, context)
        
        # Verify complete processing
        mock_summarizer.summarize_text.assert_called_once()
        mock_classifier.classify_message.assert_called_once()
        mock_db.save_message.assert_called_once()
        
        # Verify data passed correctly
        db_call = mock_db.save_message.call_args[0]
        assert db_call[1] == "https://t.me/c/1234567890/1001"  # telegram_link
        assert db_call[2] == "MIT создал квантовый процессор для комнатной температуры"  # summary
        assert db_call[3] == "technology"  # category
        
        # Verify user notification
        message.reply_text.assert_called_once()
        reply_text = message.reply_text.call_args[0][0]
        assert "✅ Сообщение обработано" in reply_text
        assert "MIT создал квантовый процессор" in reply_text
        assert "technology" in reply_text