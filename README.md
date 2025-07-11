# Telegram Bot с LLM-саммари и категоризацией

Telegram-бот для автоматической обработки пересланных сообщений с использованием Claude AI для создания саммари и категоризации контента.

## 🚀 Функциональность

- **Автоматическое саммари**: Создание кратких резюме пересланных сообщений на русском языке
- **Умная категоризация**: Автоматическое присвоение категорий с проверкой дубликатов
- **Дедупликация**: Предотвращение повторного сохранения одинаковых сообщений
- **Извлечение URL**: Автоматическое извлечение ссылок из сообщений
- **Отказоустойчивость**: Retry-логика для API-запросов с экспоненциальным backoff

## 🛠️ Технологии

- **Python 3.11** + Poetry для управления зависимостями
- **PostgreSQL** для хранения данных
- **Claude AI** (Anthropic) для суммаризации и категоризации
- **python-telegram-bot** для работы с Telegram API
- **Docker & Docker Compose** для развертывания

## 📋 Требования

- Docker и Docker Compose
- Telegram Bot Token
- Anthropic API Key

## 🚀 Запуск проекта

### 1. Клонирование и настройка

```bash
git clone <repository-url>
cd forward-summarizer-bot
```

### 2. Создание .env файла

```bash
cp .env.example .env
```

Заполните `.env` файл вашими данными:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Database settings (уже настроены для Docker)
DB_HOST=postgres
DB_NAME=telegram_bot
DB_USER=bot_user
DB_PASSWORD=bot_password
DB_PORT=5432
```

### 3. Запуск через Docker Compose

```bash
# Запуск всех сервисов
docker-compose up -d

# Просмотр логов
docker-compose logs -f bot

# Остановка
docker-compose down
```

### 4. Проверка работы

После запуска бот будет готов к работе. Перешлите любое сообщение боту, и он:
1. Создаст саммари текста
2. Присвоит категорию
3. Сохранит в базу данных
4. Отправит результат обработки

## 🏗️ Архитектура

```
forward-summarizer-bot/
├── bot.py              # Основной модуль бота
├── database.py         # Работа с PostgreSQL
├── classifier.py       # Классификация сообщений
├── summarizer.py       # Создание саммари
├── schema.sql          # Схема базы данных
├── init_db.py          # Инициализация БД
├── docker-compose.yaml # Docker Compose конфигурация
├── Dockerfile          # Docker образ приложения
└── pyproject.toml      # Poetry конфигурация
```

## 🗄️ Структура базы данных

### Таблица `categories`
- `id` - уникальный идентификатор
- `name` - название категории (уникальное)
- `created_at` - дата создания

### Таблица `messages`
- `id` - уникальный идентификатор
- `source_url` - URL источника (если есть)
- `telegram_link` - ссылка на сообщение в Telegram
- `summary` - саммари текста
- `category` - категория сообщения
- `created_at` - дата создания

## 🔧 Локальная разработка

### Установка зависимостей

```bash
# Установка Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Установка зависимостей
poetry install

# Активация virtual environment
poetry shell
```

### Запуск без Docker

```bash
# Настройка локальной PostgreSQL
# Измените .env для локальной БД:
DB_HOST=localhost
DB_USER=postgres
DB_PASSWORD=your_password

# Инициализация базы данных
poetry run python init_db.py

# Запуск бота
poetry run python bot.py
```

## 📊 Мониторинг

### Просмотр логов

```bash
# Логи бота
docker-compose logs -f bot

# Логи PostgreSQL
docker-compose logs -f postgres
```

### Подключение к базе данных

```bash
# Подключение к PostgreSQL через Docker
docker-compose exec postgres psql -U bot_user -d telegram_bot

# Просмотр таблиц
\dt

# Просмотр категорий
SELECT * FROM categories;

# Просмотр сообщений
SELECT * FROM messages ORDER BY created_at DESC LIMIT 10;
```

## 🔒 Безопасность

- Используется non-root пользователь в Docker контейнере
- Секретные данные передаются через environment variables
- База данных изолирована в Docker сети
- Логи не содержат чувствительной информации

## 🛠️ Troubleshooting

### Проблемы с запуском

1. **Ошибка подключения к PostgreSQL**: Убедитесь, что контейнер postgres запущен
2. **Ошибка Telegram API**: Проверьте правильность TELEGRAM_BOT_TOKEN
3. **Ошибка Anthropic API**: Проверьте правильность ANTHROPIC_API_KEY

### Перезапуск сервисов

```bash
# Полный перезапуск
docker-compose down
docker-compose up -d --build

# Перезапуск только бота
docker-compose restart bot
```

## 📝 Лицензия

MIT License