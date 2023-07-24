FROM python:3.11.0-slim

WORKDIR /app

COPY requirements-docker.txt requirements-docker.txt

RUN pip install --no-cache-dir -r requirements-docker.txt

COPY ["utils.py", "streamlit_demo.py", "start_script.sh", ".streamlit/*", "./"]

# Streamlit
CMD ["bash", "start_script.sh"]
EXPOSE 8501
