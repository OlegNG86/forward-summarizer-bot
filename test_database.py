import pytest
from unittest.mock import Mock, patch, MagicMock
import psycopg2
from database import Database


class TestDatabase:
    """Unit tests for Database class"""
    
    @pytest.fixture
    def mock_connection(self):
        """Mock database connection"""
        connection = Mock()
        connection.closed = 0  # Connection is open
        return connection
    
    @pytest.fixture
    def mock_cursor(self):
        """Mock database cursor"""
        cursor = Mock()
        cursor.__enter__ = Mock(return_value=cursor)
        cursor.__exit__ = Mock(return_value=None)
        return cursor
    
    @pytest.fixture
    def database(self, mock_connection, mock_cursor):
        """Create database instance with mocked connection"""
        with patch('database.psycopg2.connect', return_value=mock_connection):
            db = Database()
            db.connection = mock_connection
            mock_connection.cursor.return_value = mock_cursor
            return db
    
    def test_connect_success(self):
        """Test successful database connection"""
        with patch('database.psycopg2.connect') as mock_connect:
            mock_connect.return_value = Mock()
            db = Database()
            mock_connect.assert_called_once()
    
    def test_connect_failure(self):
        """Test database connection failure"""
        with patch('database.psycopg2.connect') as mock_connect:
            mock_connect.side_effect = psycopg2.OperationalError("Connection failed")
            with pytest.raises(psycopg2.OperationalError):
                Database()
    
    def test_get_cursor_connection_open(self, database, mock_connection, mock_cursor):
        """Test getting cursor with open connection"""
        cursor = database.get_cursor()
        assert cursor == mock_cursor
        mock_connection.cursor.assert_called_once()
    
    def test_get_cursor_connection_closed(self, database, mock_connection, mock_cursor):
        """Test getting cursor with closed connection"""
        mock_connection.closed = 1  # Connection is closed
        
        with patch.object(database, 'connect') as mock_reconnect:
            cursor = database.get_cursor()
            mock_reconnect.assert_called_once()
    
    def test_get_existing_categories(self, database, mock_cursor):
        """Test retrieving existing categories"""
        mock_cursor.fetchall.return_value = [
            {'name': 'technology'}, 
            {'name': 'news'}, 
            {'name': 'review'}
        ]
        
        categories = database.get_existing_categories()
        
        assert categories == ['technology', 'news', 'review']
        mock_cursor.execute.assert_called_once_with("SELECT name FROM categories ORDER BY name")
    
    def test_get_existing_categories_empty(self, database, mock_cursor):
        """Test retrieving categories when none exist"""
        mock_cursor.fetchall.return_value = []
        
        categories = database.get_existing_categories()
        
        assert categories == []
    
    def test_category_exists_true(self, database, mock_cursor):
        """Test category existence check - category exists"""
        mock_cursor.fetchone.return_value = {'exists': 1}
        
        exists = database.category_exists('technology')
        
        assert exists is True
        mock_cursor.execute.assert_called_once_with(
            "SELECT 1 FROM categories WHERE LOWER(name) = LOWER(%s)",
            ('technology',)
        )
    
    def test_category_exists_false(self, database, mock_cursor):
        """Test category existence check - category doesn't exist"""
        mock_cursor.fetchone.return_value = None
        
        exists = database.category_exists('nonexistent')
        
        assert exists is False
    
    def test_category_exists_case_insensitive(self, database, mock_cursor):
        """Test category existence check is case insensitive"""
        mock_cursor.fetchone.return_value = {'exists': 1}
        
        exists = database.category_exists('TECHNOLOGY')
        
        assert exists is True
        mock_cursor.execute.assert_called_once_with(
            "SELECT 1 FROM categories WHERE LOWER(name) = LOWER(%s)",
            ('TECHNOLOGY',)
        )
    
    def test_add_category_new_category(self, database, mock_cursor, mock_connection):
        """Test adding new category"""
        # Mock category doesn't exist
        with patch.object(database, 'category_exists', return_value=False):
            with patch.object(database, 'get_existing_categories', return_value=['news']):
                result = database.add_category('technology')
                
                assert result is True
                mock_cursor.execute.assert_called_once_with(
                    "INSERT INTO categories (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
                    ('technology',)
                )
                mock_connection.commit.assert_called_once()
    
    def test_add_category_existing_category(self, database, mock_cursor, mock_connection):
        """Test adding existing category"""
        with patch.object(database, 'category_exists', return_value=True):
            result = database.add_category('technology')
            
            assert result is False
            mock_cursor.execute.assert_not_called()
            mock_connection.commit.assert_not_called()
    
    def test_add_category_similar_category_warning(self, database, mock_cursor, mock_connection):
        """Test warning for similar categories"""
        with patch.object(database, 'category_exists', return_value=False):
            with patch.object(database, 'get_existing_categories', return_value=['technology']):
                with patch('database.logger') as mock_logger:
                    result = database.add_category('tech')
                    
                    assert result is True
                    mock_logger.warning.assert_called_once()
    
    def test_save_message_success(self, database, mock_cursor, mock_connection):
        """Test successful message saving"""
        mock_cursor.fetchone.return_value = {'id': 123}
        
        with patch.object(database, 'add_category') as mock_add_category:
            with patch.object(database, '_message_exists', return_value=False):
                message_id = database.save_message(
                    'https://example.com',
                    'https://t.me/test/123',
                    'Test summary',
                    'technology'
                )
                
                assert message_id == 123
                mock_add_category.assert_called_once_with('technology')
                mock_cursor.execute.assert_called_once()
                mock_connection.commit.assert_called_once()
    
    def test_save_message_duplicate(self, database, mock_cursor, mock_connection):
        """Test saving duplicate message"""
        with patch.object(database, 'add_category') as mock_add_category:
            with patch.object(database, '_message_exists', return_value=True):
                message_id = database.save_message(
                    'https://example.com',
                    'https://t.me/test/123',
                    'Test summary',
                    'technology'
                )
                
                assert message_id is None
                mock_add_category.assert_called_once_with('technology')
                mock_cursor.execute.assert_not_called()
                mock_connection.commit.assert_not_called()
    
    def test_message_exists_by_telegram_link(self, database, mock_cursor):
        """Test message existence check by telegram link"""
        mock_cursor.fetchone.return_value = {'exists': 1}
        
        exists = database._message_exists('https://t.me/test/123', 'Test summary')
        
        assert exists is True
        mock_cursor.execute.assert_called_once_with(
            "SELECT 1 FROM messages WHERE telegram_link = %s OR summary = %s LIMIT 1",
            ('https://t.me/test/123', 'Test summary')
        )
    
    def test_message_exists_by_summary(self, database, mock_cursor):
        """Test message existence check by summary"""
        mock_cursor.fetchone.return_value = {'exists': 1}
        
        exists = database._message_exists('https://t.me/test/999', 'Existing summary')
        
        assert exists is True
    
    def test_message_not_exists(self, database, mock_cursor):
        """Test message existence check when message doesn't exist"""
        mock_cursor.fetchone.return_value = None
        
        exists = database._message_exists('https://t.me/test/999', 'New summary')
        
        assert exists is False
    
    def test_close_connection(self, database, mock_connection):
        """Test closing database connection"""
        database.close()
        mock_connection.close.assert_called_once()
    
    def test_close_connection_already_closed(self, database):
        """Test closing already closed connection"""
        database.connection = None
        database.close()  # Should not raise exception
    
    def test_database_connection_parameters(self):
        """Test database connection uses correct parameters"""
        with patch('database.psycopg2.connect') as mock_connect:
            with patch.dict('os.environ', {
                'DB_HOST': 'test_host',
                'DB_NAME': 'test_db',
                'DB_USER': 'test_user',
                'DB_PASSWORD': 'test_pass',
                'DB_PORT': '5433'
            }):
                Database()
                mock_connect.assert_called_once_with(
                    host='test_host',
                    database='test_db',
                    user='test_user',
                    password='test_pass',
                    port='5433'
                )
    
    def test_database_connection_defaults(self):
        """Test database connection uses default parameters"""
        with patch('database.psycopg2.connect') as mock_connect:
            with patch.dict('os.environ', {}, clear=True):
                Database()
                mock_connect.assert_called_once_with(
                    host='localhost',
                    database='telegram_bot',
                    user='postgres',
                    password='',
                    port='5432'
                )


class TestDatabaseIntegration:
    """Integration tests for Database class"""
    
    def test_full_message_flow(self):
        """Test full message saving and retrieval flow"""
        with patch('database.psycopg2.connect') as mock_connect:
            mock_connection = Mock()
            mock_cursor = Mock()
            mock_cursor.__enter__ = Mock(return_value=mock_cursor)
            mock_cursor.__exit__ = Mock(return_value=None)
            
            mock_connection.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_connection
            
            db = Database()
            
            # Mock category operations
            mock_cursor.fetchone.side_effect = [
                None,  # category_exists returns False
                {'id': 1}  # save_message returns id
            ]
            mock_cursor.fetchall.return_value = [{'name': 'news'}]
            
            # Test adding new category and saving message
            message_id = db.save_message(
                'https://example.com',
                'https://t.me/test/123',
                'Test summary',
                'technology'
            )
            
            assert message_id == 1
            # Should have called: get_existing_categories, category_exists, 
            # _message_exists, add_category, save_message
            assert mock_cursor.execute.call_count >= 4
    
    def test_deduplication_flow(self):
        """Test message deduplication flow"""
        with patch('database.psycopg2.connect') as mock_connect:
            mock_connection = Mock()
            mock_cursor = Mock()
            mock_cursor.__enter__ = Mock(return_value=mock_cursor)
            mock_cursor.__exit__ = Mock(return_value=None)
            
            mock_connection.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_connection
            
            db = Database()
            
            # Mock duplicate message detection
            mock_cursor.fetchone.side_effect = [
                {'exists': 1},  # category_exists returns True
                {'exists': 1}   # _message_exists returns True
            ]
            
            # Test duplicate message handling
            message_id = db.save_message(
                'https://example.com',
                'https://t.me/test/123',
                'Test summary',
                'technology'
            )
            
            assert message_id is None
            # Should not insert message if duplicate detected
            insert_calls = [call for call in mock_cursor.execute.call_args_list 
                          if 'INSERT INTO messages' in str(call)]
            assert len(insert_calls) == 0


# Error handling tests
class TestDatabaseErrorHandling:
    """Test error handling in Database class"""
    
    def test_connection_error_handling(self):
        """Test handling of connection errors"""
        with patch('database.psycopg2.connect') as mock_connect:
            mock_connect.side_effect = psycopg2.OperationalError("Connection failed")
            
            with pytest.raises(psycopg2.OperationalError):
                Database()
    
    def test_query_error_handling(self, database, mock_cursor):
        """Test handling of query errors"""
        mock_cursor.execute.side_effect = psycopg2.Error("Query failed")
        
        with pytest.raises(psycopg2.Error):
            database.get_existing_categories()
    
    def test_transaction_rollback_on_error(self, database, mock_cursor, mock_connection):
        """Test transaction rollback on error"""
        mock_cursor.execute.side_effect = psycopg2.Error("Insert failed")
        
        with patch.object(database, 'add_category'):
            with patch.object(database, '_message_exists', return_value=False):
                with pytest.raises(psycopg2.Error):
                    database.save_message(
                        'https://example.com',
                        'https://t.me/test/123',
                        'Test summary',
                        'technology'
                    )


# Performance tests
class TestDatabasePerformance:
    """Performance tests for Database class"""
    
    def test_category_check_performance(self, database, mock_cursor):
        """Test performance of category existence check"""
        import time
        
        mock_cursor.fetchone.return_value = {'exists': 1}
        
        start_time = time.time()
        for _ in range(100):
            database.category_exists('technology')
        end_time = time.time()
        
        # Should be fast even with many calls
        assert end_time - start_time < 0.1
    
    def test_bulk_category_operations(self, database, mock_cursor, mock_connection):
        """Test performance of bulk category operations"""
        mock_cursor.fetchone.return_value = None  # Categories don't exist
        mock_cursor.fetchall.return_value = []
        
        categories = [f'category_{i}' for i in range(50)]
        
        import time
        start_time = time.time()
        for category in categories:
            database.add_category(category)
        end_time = time.time()
        
        # Should handle bulk operations reasonably fast
        assert end_time - start_time < 1.0


# Fixtures
@pytest.fixture
def sample_messages():
    """Sample messages for testing"""
    return [
        {
            'source_url': 'https://example.com/article1',
            'telegram_link': 'https://t.me/test/123',
            'summary': 'Summary of article 1',
            'category': 'technology'
        },
        {
            'source_url': None,
            'telegram_link': 'https://t.me/test/124',
            'summary': 'Summary of article 2',
            'category': 'news'
        },
        {
            'source_url': 'https://example.com/article3',
            'telegram_link': 'https://t.me/test/125',
            'summary': 'Summary of article 3',
            'category': 'review'
        }
    ]