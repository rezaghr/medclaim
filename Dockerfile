FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN groupadd --system medclaim && useradd --system --gid medclaim --create-home medclaim

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY --chown=medclaim:medclaim src /app/src
COPY --chown=medclaim:medclaim app /app/app
COPY --chown=medclaim:medclaim scripts/initialize_runtime_artifacts.py /app/scripts/initialize_runtime_artifacts.py
COPY --chown=medclaim:medclaim configs/deployment/default.yaml /app/configs/deployment/default.yaml

USER medclaim
EXPOSE 8000 8501

CMD ["uvicorn", "medclaim.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
