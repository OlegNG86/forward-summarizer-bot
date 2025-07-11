import os
import logging
import anthropic
import time
from typing import List, Tuple
from database import Database

logger = logging.getLogger(__name__)

class MessageClassifier:
    def __init__(self, db: Database):
        self.db = db
        self.client = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
    
    def classify_message(self, text: str) -> Tuple[str, float]:
        """
        Classify message text into category with confidence score
        Returns: (category, confidence_score)
        """
        # Get existing categories
        existing_categories = self.db.get_existing_categories()
        
        # First try simple string matching (case-insensitive)
        category_match = self._simple_category_match(text, existing_categories)
        if category_match:
            logger.info(f"Found category match: {category_match}")
            return category_match, 1.0
        
        # If no match, use Claude for classification
        category, confidence = self._classify_with_claude(text, existing_categories)
        
        # If Claude suggests a new category, verify it's not a duplicate
        if category.lower() not in [cat.lower() for cat in existing_categories]:
            category = self._check_duplicate_category(category, existing_categories)
        
        return category, confidence
    
    def _simple_category_match(self, text: str, categories: List[str]) -> str:
        """Simple case-insensitive string matching"""
        text_lower = text.lower()
        for category in categories:
            if category.lower() in text_lower:
                return category
        return None
    
    def _classify_with_claude(self, text: str, existing_categories: List[str]) -> Tuple[str, float]:
        """Classify text using Claude with chain-of-thought reasoning"""
        
        categories_str = ", ".join(existing_categories) if existing_categories else "нет существующих категорий"
        
        prompt = f"""Твоя задача - классифицировать текст сообщения в одну из существующих категорий или предложить новую.

Существующие категории: {categories_str}

Текст для классификации:
{text}

Подумай пошагово (think step by step):
1. Проанализируй основную тему текста
2. Определи ключевые слова и контекст
3. Сравни с существующими категориями
4. Выбери наиболее подходящую категорию или предложи новую

Примеры классификации:

Пример 1:
Текст: "Новый iPhone 15 получил улучшенную камеру и процессор A17"
Анализ: Речь идет о технологическом продукте, новинке в сфере мобильных устройств
Категория: technology
Уверенность: 0.9

Пример 2:
Текст: "Сегодня президент подписал новый закон о налогах"
Анализ: Политическая новость, касается государственного управления
Категория: news
Уверенность: 0.8

Пример 3:
Текст: "Отличный фильм! Актерская игра на высоте, сюжет захватывающий"
Анализ: Отзыв о кинематографическом произведении, оценка качества
Категория: review
Уверенность: 0.85

Пример 4:
Текст: "Как приготовить идеальную пасту карбонара: секреты от шеф-повара"
Анализ: Кулинарная тематика, рецепты и советы по приготовлению
Категория: cooking
Уверенность: 0.9

Пример 5:
Текст: "Курс доллара вырос на 2%, эксперты прогнозируют дальнейший рост"
Анализ: Финансовая информация, экономические показатели
Категория: finance
Уверенность: 0.85

Теперь классифицируй данный текст:

Анализ: 
Категория: 
Уверенность: """

        return self._call_claude_with_retry(prompt, self._parse_claude_response, "classification")
    
    def _parse_claude_response(self, response_text: str) -> Tuple[str, float]:
        """Parse Claude's response to extract category and confidence"""
        try:
            lines = response_text.strip().split('\n')
            category = None
            confidence = 0.5
            
            for line in lines:
                line = line.strip()
                if line.startswith('Категория:'):
                    category = line.split(':', 1)[1].strip()
                elif line.startswith('Уверенность:'):
                    confidence_str = line.split(':', 1)[1].strip()
                    try:
                        confidence = float(confidence_str)
                    except ValueError:
                        confidence = 0.5
            
            if not category:
                category = "general"
            
            return category, confidence
            
        except Exception as e:
            logger.error(f"Error parsing Claude response: {e}")
            return "general", 0.5
    
    def _check_duplicate_category(self, new_category: str, existing_categories: List[str]) -> str:
        """Check if new category is a duplicate of existing ones"""
        
        if not existing_categories:
            return new_category
        
        categories_str = ", ".join(existing_categories)
        
        prompt = f"""Проверь, является ли предложенная категория дубликатом уже существующих.

Существующие категории: {categories_str}
Предложенная категория: {new_category}

Проанализируй:
1. Совпадает ли смысл с существующими категориями?
2. Есть ли синонимы или очень похожие по смыслу категории?
3. Стоит ли использовать существующую категорию вместо новой?

Если есть дубликат, верни существующую категорию.
Если дубликата нет, верни предложенную категорию.

Ответ (только название категории):"""

        def parse_duplicate_response(response_text):
            result_category = response_text.strip()
            if result_category.lower() in [cat.lower() for cat in existing_categories]:
                logger.info(f"Using existing category '{result_category}' instead of '{new_category}'")
                return result_category
            return new_category
        
        return self._call_claude_with_retry(prompt, parse_duplicate_response, "duplicate_check", max_tokens=50)
    
    def _call_claude_with_retry(self, prompt: str, parser_func, operation_name: str, max_tokens: int = 200, max_retries: int = 3):
        """Call Claude API with retry logic and exponential backoff"""
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                response_text = response.content[0].text
                
                if operation_name == "classification":
                    category, confidence = parser_func(response_text)
                    if confidence < 0.5:
                        logger.warning(f"Low confidence ({confidence}) for category '{category}', using 'review'")
                        return "review", confidence
                    return category, confidence
                else:
                    return parser_func(response_text)
                    
            except anthropic.RateLimitError as e:
                wait_time = (2 ** attempt) + 1
                logger.warning(f"Rate limit hit for {operation_name}, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            except anthropic.APIError as e:
                logger.error(f"Claude API error for {operation_name}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                break
            except Exception as e:
                logger.error(f"Unexpected error calling Claude API for {operation_name}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                break
        
        # Fallback if all retries failed
        logger.error(f"All retries failed for {operation_name}")
        if operation_name == "classification":
            return "general", 0.5
        else:
            return "general"