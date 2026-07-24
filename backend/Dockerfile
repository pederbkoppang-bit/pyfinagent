FROM python:3.14-slim

WORKDIR /app

# phase-75.16 (deps-03/04): build context is the repo root (COPY . . below
# assumes this); the old `COPY requirements.txt .` pointed at a file that
# does not exist at the repo root -- only backend/requirements.txt does --
# so the build could never succeed. Runtime pinned to match the deployed
# 3.14.4 interpreter (was 3.11).
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as non-root user
RUN addgroup --system --gid 1001 appuser && \
    adduser --system --uid 1001 --ingroup appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
