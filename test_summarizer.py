import pytest
from unittest.mock import Mock, patch
from summarizer import TextSummarizer
import anthropic


class TestTextSummarizer:
    """Unit tests for TextSummarizer class"""
    
    @pytest.fixture
    def summarizer(self):
        """Create summarizer instance with mocked Anthropic client"""
        with patch('summarizer.anthropic.Anthropic'):
            return TextSummarizer()
    
    def test_summarize_text_short_text(self, summarizer):
        """Test that short text is returned as-is"""
        short_text = "Short text"
        result = summarizer.summarize_text(short_text)
        assert result == short_text
    
    def test_summarize_text_empty_text(self, summarizer):
        """Test handling of empty text"""
        result = summarizer.summarize_text("")
        assert result == ""
    
    def test_summarize_text_none_input(self, summarizer):
        """Test handling of None input"""
        result = summarizer.summarize_text(None)
        assert result == ""
    
    @patch('summarizer.TextSummarizer._call_claude_with_retry')
    def test_summarize_text_long_text(self, mock_retry, summarizer):
        """Test summarization of long text"""
        long_text = "This is a very long text that needs summarization. " * 10
        expected_summary = "Краткое резюме длинного текста"
        mock_retry.return_value = expected_summary
        
        result = summarizer.summarize_text(long_text)
        
        assert result == expected_summary
        mock_retry.assert_called_once()
    
    def test_clean_text_excessive_whitespace(self, summarizer):
        """Test cleaning excessive whitespace"""
        text = "Text   with    excessive     whitespace"
        result = summarizer._clean_text(text)
        assert result == "Text with excessive whitespace"
    
    def test_clean_text_forwarding_markers(self, summarizer):
        """Test removal of forwarding markers"""
        text = "Forwarded from @channel\nActual message content"
        result = summarizer._clean_text(text)
        assert result == "Actual message content"
    
    def test_clean_text_russian_forwarding_markers(self, summarizer):
        """Test removal of Russian forwarding markers"""
        text = "Переслано от @channel\nСодержание сообщения"
        result = summarizer._clean_text(text)
        assert result == "Содержание сообщения"
    
    def test_clean_text_excessive_punctuation(self, summarizer):
        """Test cleaning excessive punctuation"""
        text = "Text with excessive punctuation!!! And dots... More dots...."
        result = summarizer._clean_text(text)
        assert result == "Text with excessive punctuation! And dots... More dots..."
    
    def test_clean_summary_with_prefixes(self, summarizer):
        """Test removal of summary prefixes"""
        test_cases = [
            ("Резюме: This is summary", "This is summary"),
            ("Краткое резюме: This is summary", "This is summary"),
            ("Основная идея: This is summary", "This is summary"),
            ("Суть: This is summary", "This is summary"),
            ("В кратце: This is summary", "This is summary"),
            ("Вкратце: This is summary", "This is summary"),
        ]
        
        for input_text, expected in test_cases:
            result = summarizer._clean_summary(input_text)
            assert result == expected
    
    def test_clean_summary_quoted_text(self, summarizer):
        """Test removal of quotes around summary"""
        text = '"This is a quoted summary"'
        result = summarizer._clean_summary(text)
        assert result == "This is a quoted summary"
    
    def test_clean_summary_no_changes_needed(self, summarizer):
        """Test that clean summary doesn't change already clean text"""
        text = "This is already clean summary"
        result = summarizer._clean_summary(text)
        assert result == text
    
    def test_fallback_summary_short_text(self, summarizer):
        """Test fallback summary for short text"""
        text = "Short text"
        result = summarizer._fallback_summary(text, 100)
        assert result == text
    
    def test_fallback_summary_sentence_boundary(self, summarizer):
        """Test fallback summary cuts at sentence boundary"""
        text = "First sentence. Second sentence. Third sentence."
        result = summarizer._fallback_summary(text, 20)
        assert result == "First sentence."
    
    def test_fallback_summary_word_boundary(self, summarizer):
        """Test fallback summary cuts at word boundary"""
        text = "This is a long text without sentence boundaries"
        result = summarizer._fallback_summary(text, 20)
        assert result.endswith("...")
        assert len(result) <= 23  # 20 + "..."
    
    def test_fallback_summary_hard_truncation(self, summarizer):
        """Test fallback summary with hard truncation"""
        text = "Verylongtextwithoutspaces"
        result = summarizer._fallback_summary(text, 10)
        assert result == "Verylongte..."
    
    @patch('time.sleep')
    def test_call_claude_with_retry_success(self, mock_sleep, summarizer):
        """Test successful Claude API call"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Резюме: Test summary")]
        summarizer.client.messages.create.return_value = mock_response
        
        result = summarizer._call_claude_with_retry("test prompt", 200, "original text")
        
        assert result == "Test summary"
        mock_sleep.assert_not_called()
    
    @patch('time.sleep')
    def test_call_claude_with_retry_rate_limit(self, mock_sleep, summarizer):
        """Test retry on rate limit error"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Test summary")]
        
        summarizer.client.messages.create.side_effect = [
            anthropic.RateLimitError("Rate limit exceeded"),
            mock_response
        ]
        
        result = summarizer._call_claude_with_retry("test prompt", 200, "original text")
        
        assert result == "Test summary"
        mock_sleep.assert_called_once_with(2)
    
    @patch('time.sleep')
    def test_call_claude_with_retry_api_error(self, mock_sleep, summarizer):
        """Test retry on API error"""
        summarizer.client.messages.create.side_effect = [
            anthropic.APIError("API Error"),
            anthropic.APIError("API Error"),
            anthropic.APIError("API Error")
        ]
        
        result = summarizer._call_claude_with_retry("test prompt", 200, "original text")
        
        # Should return fallback summary
        assert result == "original text"
        assert mock_sleep.call_count == 2
    
    @patch('time.sleep')
    def test_call_claude_with_retry_long_summary_truncation(self, mock_sleep, summarizer):
        """Test truncation of long summaries"""
        mock_response = Mock()
        long_summary = "This is a very long summary that exceeds the maximum length limit " * 5
        mock_response.content = [Mock(text=long_summary)]
        summarizer.client.messages.create.return_value = mock_response
        
        result = summarizer._call_claude_with_retry("test prompt", 100, "original text")
        
        assert len(result) <= 100
        assert result.endswith("...")
        mock_sleep.assert_not_called()
    
    def test_integration_summarize_full_flow(self, summarizer):
        """Integration test for full summarization flow"""
        with patch.object(summarizer, '_call_claude_with_retry') as mock_retry:
            mock_retry.return_value = "Краткое резюме статьи о технологиях"
            
            text = "This is a long article about technology trends and innovations in 2024. " * 10
            result = summarizer.summarize_text(text)
            
            assert result == "Краткое резюме статьи о технологиях"
            mock_retry.assert_called_once()
    
    def test_max_length_parameter(self, summarizer):
        """Test custom max_length parameter"""
        with patch.object(summarizer, '_call_claude_with_retry') as mock_retry:
            mock_retry.return_value = "Short summary"
            
            text = "Long text that needs summarization " * 20
            result = summarizer.summarize_text(text, max_length=50)
            
            # Verify max_length was passed to retry function
            args, kwargs = mock_retry.call_args
            assert args[1] == 50  # max_length parameter
    
    @patch('summarizer.TextSummarizer._call_claude_with_retry')
    def test_different_summary_lengths(self, mock_retry, summarizer):
        """Test summarization with different max lengths"""
        long_text = "This is a long text that needs summarization " * 20
        
        # Test different max lengths
        for max_len in [50, 100, 200, 500]:
            mock_retry.return_value = f"Summary with max length {max_len}"
            result = summarizer.summarize_text(long_text, max_length=max_len)
            assert result == f"Summary with max length {max_len}"


# Performance tests
class TestSummarizerPerformance:
    """Performance tests for summarizer"""
    
    def test_short_text_performance(self, summarizer):
        """Test that short texts are processed quickly"""
        import time
        
        short_text = "Short text"
        start_time = time.time()
        result = summarizer.summarize_text(short_text)
        end_time = time.time()
        
        assert result == short_text
        assert end_time - start_time < 0.001  # Should be very fast
    
    def test_text_cleaning_performance(self, summarizer):
        """Test performance of text cleaning"""
        import time
        
        # Large text with various formatting issues
        messy_text = ("Forwarded from @channel\n" + 
                     "Text with    excessive     whitespace!!! " +
                     "And dots.... " * 1000)
        
        start_time = time.time()
        result = summarizer._clean_text(messy_text)
        end_time = time.time()
        
        assert len(result) < len(messy_text)
        assert end_time - start_time < 0.1  # Should be reasonably fast


# Fixtures
@pytest.fixture
def sample_texts():
    """Sample texts for testing"""
    return {
        'short': "Short text",
        'medium': "This is a medium length text that might need summarization but is not too long.",
        'long': "This is a very long text that definitely needs summarization. " * 20,
        'messy': "Forwarded from @channel\nThis    text   has   formatting!!! Issues....",
        'russian': "Переслано от канала\nЭто текст на русском языке с форматированием.",
        'empty': "",
        'whitespace': "   \n\t   \n   "
    }


@pytest.fixture
def mock_claude_responses():
    """Mock responses from Claude API"""
    return {
        'good_summary': "Краткое и точное резюме текста",
        'long_summary': "Это очень длинное резюме которое превышает лимит символов " * 10,
        'with_prefix': "Резюме: Хорошее резюме текста",
        'quoted': '"Резюме в кавычках"',
        'malformed': "Некорректный ответ без структуры"
    }