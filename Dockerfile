FROM python:3.9-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем pip и обновляем его до последней версии
RUN pip install --upgrade pip

# Устанавливаем зависимости Python
COPY requirements.txt .
RUN pip install -r requirements.txt

# Добавляем код приложения
COPY . /app
WORKDIR /app

# Указываем порт, используемый приложением
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
