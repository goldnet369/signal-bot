FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "-w", "1", "--threads", "4", "-t", "120", "app:app"]
