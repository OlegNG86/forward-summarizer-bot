import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from classifier import MessageClassifier
import anthropic


class TestMessageClassifier:
    """Unit tests for MessageClassifier class"""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database with predefined categories"""
        db = Mock()
        db.get_existing_categories.return_value = ['technology', 'news', 'review', 'finance']
        return db
    
    @pytest.fixture
    def classifier(self, mock_db):
        """Create classifier instance with mocked database"""
        with patch('classifier.anthropic.Anthropic'):
            return MessageClassifier(mock_db)
    
    def test_simple_category_match_exact(self, classifier):
        """Test exact category match in text"""
        text = "This is about technology and innovation"
        result = classifier._simple_category_match(text, ['technology', 'news'])
        assert result == 'technology'
    
    def test_simple_category_match_case_insensitive(self, classifier):
        """Test case-insensitive category matching"""
        text = "This is about TECHNOLOGY and innovation"
        result = classifier._simple_category_match(text, ['technology', 'news'])
        assert result == 'technology'
    
    def test_simple_category_match_no_match(self, classifier):
        """Test when no category matches"""
        text = "This is about cooking and recipes"
        result = classifier._simple_category_match(text, ['technology', 'news'])
        assert result is None
    
    def test_simple_category_match_empty_categories(self, classifier):
        """Test with empty categories list"""
        text = "This is about technology"
        result = classifier._simple_category_match(text, [])
        assert result is None
    
    @patch('classifier.MessageClassifier._call_claude_with_retry')
    def test_classify_message_simple_match(self, mock_retry, classifier):
        """Test classification with simple string match"""
        text = "New iPhone technology breakthrough"
        
        category, confidence = classifier.classify_message(text)
        
        assert category == 'technology'
        assert confidence == 1.0
        # Should not call Claude API if simple match found
        mock_retry.assert_not_called()
    
    @patch('classifier.MessageClassifier._call_claude_with_retry')
    def test_classify_message_claude_classification(self, mock_retry, classifier):
        """Test classification using Claude API"""
        text = "Article about artificial intelligence research"
        mock_retry.return_value = ('ai_research', 0.85)
        
        category, confidence = classifier.classify_message(text)
        
        assert category == 'ai_research'
        assert confidence == 0.85
        mock_retry.assert_called_once()
    
    @patch('classifier.MessageClassifier._call_claude_with_retry')
    @patch('classifier.MessageClassifier._check_duplicate_category')
    def test_classify_message_new_category_check(self, mock_duplicate, mock_retry, classifier):
        """Test duplicate checking for new categories"""
        text = "Article about machine learning"
        mock_retry.return_value = ('machine_learning', 0.8)
        mock_duplicate.return_value = 'technology'  # Duplicate found
        
        category, confidence = classifier.classify_message(text)
        
        assert category == 'technology'
        assert confidence == 0.8
        mock_duplicate.assert_called_once()
    
    def test_parse_claude_response_success(self, classifier):
        """Test successful parsing of Claude response"""
        response_text = """
        Анализ: Текст о новых технологиях
        Категория: technology
        Уверенность: 0.9
        """
        
        category, confidence = classifier._parse_claude_response(response_text)
        
        assert category == 'technology'
        assert confidence == 0.9
    
    def test_parse_claude_response_malformed(self, classifier):
        """Test parsing malformed Claude response"""
        response_text = "Invalid response format"
        
        category, confidence = classifier._parse_claude_response(response_text)
        
        assert category == 'general'
        assert confidence == 0.5
    
    def test_parse_claude_response_missing_category(self, classifier):
        """Test parsing response with missing category"""
        response_text = """
        Анализ: Some analysis
        Уверенность: 0.8
        """
        
        category, confidence = classifier._parse_claude_response(response_text)
        
        assert category == 'general'
        assert confidence == 0.8
    
    def test_parse_claude_response_invalid_confidence(self, classifier):
        """Test parsing response with invalid confidence"""
        response_text = """
        Категория: technology
        Уверенность: invalid_number
        """
        
        category, confidence = classifier._parse_claude_response(response_text)
        
        assert category == 'technology'
        assert confidence == 0.5
    
    @patch('classifier.MessageClassifier._call_claude_with_retry')
    def test_check_duplicate_category_found(self, mock_retry, classifier):
        """Test duplicate category detection"""
        existing_categories = ['technology', 'news']
        mock_retry.return_value = 'technology'  # Duplicate found
        
        result = classifier._check_duplicate_category('tech', existing_categories)
        
        assert result == 'technology'
        mock_retry.assert_called_once()
    
    @patch('classifier.MessageClassifier._call_claude_with_retry')
    def test_check_duplicate_category_not_found(self, mock_retry, classifier):
        """Test when no duplicate category found"""
        existing_categories = ['technology', 'news']
        mock_retry.return_value = 'cooking'  # No duplicate
        
        result = classifier._check_duplicate_category('cooking', existing_categories)
        
        assert result == 'cooking'
        mock_retry.assert_called_once()
    
    @patch('time.sleep')
    def test_call_claude_with_retry_success(self, mock_sleep, classifier):
        """Test successful Claude API call"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Категория: technology\nУверенность: 0.9")]
        classifier.client.messages.create.return_value = mock_response
        
        def mock_parser(text):
            return ('technology', 0.9)
        
        result = classifier._call_claude_with_retry("test prompt", mock_parser, "classification")
        
        assert result == ('technology', 0.9)
        classifier.client.messages.create.assert_called_once()
        mock_sleep.assert_not_called()
    
    @patch('time.sleep')
    def test_call_claude_with_retry_rate_limit(self, mock_sleep, classifier):
        """Test retry logic on rate limit error"""
        # First call fails with rate limit, second succeeds
        mock_response = Mock()
        mock_response.content = [Mock(text="Категория: technology\nУверенность: 0.9")]
        
        classifier.client.messages.create.side_effect = [
            anthropic.RateLimitError("Rate limit exceeded"),
            mock_response
        ]
        
        def mock_parser(text):
            return ('technology', 0.9)
        
        result = classifier._call_claude_with_retry("test prompt", mock_parser, "classification")
        
        assert result == ('technology', 0.9)
        assert classifier.client.messages.create.call_count == 2
        mock_sleep.assert_called_once_with(2)  # 2^0 + 1 = 2
    
    @patch('time.sleep')
    def test_call_claude_with_retry_api_error(self, mock_sleep, classifier):
        """Test retry logic on API error"""
        classifier.client.messages.create.side_effect = [
            anthropic.APIError("API Error"),
            anthropic.APIError("API Error"),
            anthropic.APIError("API Error")
        ]
        
        def mock_parser(text):
            return ('technology', 0.9)
        
        result = classifier._call_claude_with_retry("test prompt", mock_parser, "classification")
        
        # Should return fallback after all retries fail
        assert result == ('general', 0.5)
        assert classifier.client.messages.create.call_count == 3
        assert mock_sleep.call_count == 2  # Sleep between retries
    
    @patch('time.sleep')
    def test_call_claude_with_retry_low_confidence(self, mock_sleep, classifier):
        """Test handling of low confidence classification"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Категория: uncertain\nУверенность: 0.3")]
        classifier.client.messages.create.return_value = mock_response
        
        def mock_parser(text):
            return ('uncertain', 0.3)
        
        result = classifier._call_claude_with_retry("test prompt", mock_parser, "classification")
        
        assert result == ('review', 0.3)  # Low confidence -> review
        mock_sleep.assert_not_called()
    
    @patch('time.sleep')
    def test_call_claude_with_retry_non_classification_operation(self, mock_sleep, classifier):
        """Test retry logic for non-classification operations"""
        mock_response = Mock()
        mock_response.content = [Mock(text="technology")]
        classifier.client.messages.create.return_value = mock_response
        
        def mock_parser(text):
            return 'technology'
        
        result = classifier._call_claude_with_retry("test prompt", mock_parser, "duplicate_check")
        
        assert result == 'technology'
        mock_sleep.assert_not_called()
    
    @patch('time.sleep')
    def test_call_claude_with_retry_max_retries_exceeded(self, mock_sleep, classifier):
        """Test behavior when max retries exceeded"""
        classifier.client.messages.create.side_effect = Exception("Connection error")
        
        def mock_parser(text):
            return ('technology', 0.9)
        
        result = classifier._call_claude_with_retry("test prompt", mock_parser, "classification", max_retries=2)
        
        assert result == ('general', 0.5)
        assert classifier.client.messages.create.call_count == 2
        assert mock_sleep.call_count == 1  # Sleep before second retry
    
    def test_integration_classify_message_full_flow(self, mock_db):
        """Integration test for full classification flow"""
        # Mock database with specific categories
        mock_db.get_existing_categories.return_value = ['technology', 'news']
        
        with patch('classifier.anthropic.Anthropic') as mock_anthropic:
            classifier = MessageClassifier(mock_db)
            
            # Mock Claude API response
            mock_response = Mock()
            mock_response.content = [Mock(text="Категория: artificial_intelligence\nУверенность: 0.85")]
            classifier.client.messages.create.return_value = mock_response
            
            # Test text that doesn't match existing categories
            text = "Advanced AI systems are revolutionizing healthcare"
            
            category, confidence = classifier.classify_message(text)
            
            # Should get AI classification since no simple match found
            assert category == 'artificial_intelligence'
            assert confidence == 0.85
            
            # Verify database was queried for existing categories
            mock_db.get_existing_categories.assert_called()


# Fixtures for pytest
@pytest.fixture
def sample_texts():
    """Sample texts for testing"""
    return {
        'technology': "New smartphone with advanced AI chip released",
        'news': "Government announces new economic policy",
        'review': "This movie is absolutely fantastic, great acting",
        'finance': "Stock market rises 2% amid economic recovery",
        'general': "Random text about daily life activities"
    }


@pytest.fixture
def mock_anthropic_responses():
    """Mock responses from Anthropic API"""
    return {
        'high_confidence': "Категория: technology\nУверенность: 0.9",
        'low_confidence': "Категория: uncertain\nУверенность: 0.3",
        'malformed': "Invalid response format",
        'duplicate_check': "technology"
    }


# Performance tests
class TestClassifierPerformance:
    """Performance tests for classifier"""
    
    def test_simple_matching_performance(self, classifier):
        """Test performance of simple category matching"""
        text = "technology news about innovation"
        categories = ['technology', 'news', 'review'] * 100  # Large category list
        
        start_time = time.time()
        result = classifier._simple_category_match(text, categories)
        end_time = time.time()
        
        assert result == 'technology'
        assert end_time - start_time < 0.1  # Should be fast
    
    @patch('classifier.MessageClassifier._call_claude_with_retry')
    def test_classification_caching_behavior(self, mock_retry, classifier):
        """Test that simple matches bypass API calls"""
        text = "technology breakthrough"
        
        # Multiple calls with same text
        for _ in range(5):
            category, confidence = classifier.classify_message(text)
            assert category == 'technology'
            assert confidence == 1.0
        
        # Should never call Claude API
        mock_retry.assert_not_called()