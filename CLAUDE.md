# CLAUDE.md – Telegram‑бот с LLM‑саммари и категоризацией (с дедупликацией)

Этот файл настраивает Claude Code согласно лучшим практикам Anthropic Academy с учётом проверки существующих категорий и предотвращения дубликатов.

---

## 🚀 Обзор проекта

Бот принимает пересланные Telegram-сообщения и сохраняет в БД:
1. URL источника (если есть)
2. Ссылка на сообщение в Telegram
3. Саммари текста (через Claude)
4. Категория — выбирается из существующих или создаётся новая после проверки

---

## 🧩 Архитектура и поток данных

1. Сервер извлекает текст и URL из пересланного сообщения  
2. Берёт список existing_categories из таблицы categories(name TEXT UNIQUE)  
3. Пытается простым string-match (case-insensitive)  
4. Если не найдено — вызывает Claude с prompt:
   - Передаются существующие категории
   - Используется chain-of-thought («think step by step»)
   - Применён few-shot шаблон (3–5 примеров) :contentReference[oaicite:2]{index=2}  
5. Если Claude предлагает новую категорию — повторный prompt для проверки дублирования  
6. При низкой уверенности — категория помечается как review  
7. Только после успешной проверки — сохраняется запись в messages

БД:
```sql
CREATE TABLE categories (
  id SERIAL PRIMARY KEY,
  name TEXT UNIQUE NOT NULL
);

CREATE TABLE messages (
  id SERIAL PRIMARY KEY,
  source_url TEXT,
  telegram_link TEXT NOT NULL,
  summary TEXT NOT NULL,
  category TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 🔧 Конфигурация

Создайте файл `.env` в корневой папке проекта с токеном Telegram бота:

```env
TELEGRAM_BOT_TOKEN=7084567973:AAGfYVbhr0tIAC3kX_EdhsDZjiIMFosuI2s
```