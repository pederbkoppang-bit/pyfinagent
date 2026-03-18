FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ ./backend/

RUN adduser --disabled-password --uid 1001 appuser
USER appuser

CMD ["python", "-m", "backend.slack_bot.app"]
