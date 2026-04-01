FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir torch==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV MODEL_ID=JDS-74/kidlearn-story-model
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache

RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('JDS-74/kidlearn-story-model'); AutoModelForCausalLM.from_pretrained('JDS-74/kidlearn-story-model')"

EXPOSE 3001

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:3001", "--timeout", "120", "--workers", "1"]
