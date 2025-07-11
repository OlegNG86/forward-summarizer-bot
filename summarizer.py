import os
import logging
import anthropic
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

class TextSummarizer:
    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
    
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """
        Summarize text using Claude AI with Russian language support
        Returns concise summary in Russian
        """
        if not text or len(text.strip()) < 50:
            return text.strip()
        
        # Clean and prepare text
        cleaned_text = self._clean_text(text)
        
        prompt = f"""Твоя задача - создать краткое резюме текста на русском языке.

Требования к резюме:
- Максимум {max_length} символов
- Сохрани основную идею и ключевые факты
- Используй ясный и понятный язык
- Убери лишние детали, но сохрани суть

Примеры хороших резюме:

Исходный текст: "Apple представила новый iPhone 15 Pro с титановым корпусом, улучшенными камерами и новым процессором A17 Pro. Устройство получило порт USB-C вместо Lightning, что является значительным изменением для пользователей Apple. Цена стартует от 999 долларов."
Резюме: "Apple представила iPhone 15 Pro с титановым корпусом, процессором A17 Pro и портом USB-C. Цена от $999."

Исходный текст: "Исследование показало, что регулярные физические упражнения снижают риск развития сердечно-сосудистых заболеваний на 35%. Ученые наблюдали за 10000 участников в течение 15 лет. Наиболее эффективными оказались кардио-тренировки продолжительностью 30 минут 5 раз в неделю."
Резюме: "Исследование: регулярные кардио-тренировки (30 мин, 5 раз в неделю) снижают риск сердечных заболеваний на 35%."

Исходный текст: "Компания Tesla отчиталась о рекордных продажах электромобилей в третьем квартале 2023 года. Было продано 435,000 автомобилей, что на 27% больше по сравнению с предыдущим кварталом. Илон Маск отметил, что основной рост пришелся на модели Model 3 и Model Y."
Резюме: "Tesla продала рекордные 435,000 электромобилей в Q3 2023 (+27% к предыдущему кварталу), рост за счет Model 3 и Model Y."

Теперь создай резюме для следующего текста:

{cleaned_text}

Резюме:"""

        return self._call_claude_with_retry(prompt, max_length, cleaned_text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and prepare text for summarization"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common forwarding markers
        text = re.sub(r'^(Forwarded from|Переслано от|Пересылка от).*?\n', '', text, flags=re.MULTILINE)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text
    
    def _clean_summary(self, summary: str) -> str:
        """Clean the generated summary"""
        # Remove common prefixes that Claude might add
        prefixes = [
            'Резюме:', 'Краткое резюме:', 'Основная идея:', 
            'Суть:', 'В кратце:', 'Вкратце:'
        ]
        
        for prefix in prefixes:
            if summary.startswith(prefix):
                summary = summary[len(prefix):].strip()
                break
        
        # Remove quotes if the entire summary is quoted
        if summary.startswith('"') and summary.endswith('"'):
            summary = summary[1:-1].strip()
        
        return summary
    
    def _fallback_summary(self, text: str, max_length: int) -> str:
        """Fallback summary when API fails"""
        # Simple truncation with sentence boundaries
        if len(text) <= max_length:
            return text
        
        # Try to cut at sentence boundary
        truncated = text[:max_length]
        last_sentence = truncated.rfind('.')
        if last_sentence > max_length // 2:
            return truncated[:last_sentence + 1]
        
        # Cut at word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_length // 2:
            return truncated[:last_space] + '...'
        
        return truncated + '...'
    
    def _call_claude_with_retry(self, prompt: str, max_length: int, original_text: str, max_retries: int = 3):
        """Call Claude API with retry logic and exponential backoff"""
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                summary = response.content[0].text.strip()
                
                # Clean the summary
                summary = self._clean_summary(summary)
                
                # Ensure it's not longer than max_length
                if len(summary) > max_length:
                    summary = summary[:max_length].rsplit(' ', 1)[0] + '...'
                
                return summary
                
            except anthropic.RateLimitError as e:
                wait_time = (2 ** attempt) + 1
                logger.warning(f"Rate limit hit for summarization, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            except anthropic.APIError as e:
                logger.error(f"Claude API error for summarization: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                break
            except Exception as e:
                logger.error(f"Unexpected error calling Claude API for summarization: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                break
        
        # Fallback if all retries failed
        logger.error("All retries failed for summarization")
        return self._fallback_summary(original_text, max_length)