FROM python:3.13.0

WORKDIR /opt/program
ENV PATH="/opt/program/.venv/bin:$PATH"

RUN pip install uv

# Copy project files
COPY pyproject.toml .python-version uv.lock ./

# Sync dependencies với UV
RUN uv sync --locked

COPY serve.py model.bin ./

EXPOSE 8080
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]

