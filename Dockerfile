FROM python:3.13-slim

WORKDIR /app

# System deps: none needed beyond what psycopg2-binary/pip provide.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command runs the API; docker-compose.yml overrides this for the
# streamlit service. Using one image for both keeps the build simple for
# a project this size — a larger system might split these into separate
# images to keep each smaller.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
