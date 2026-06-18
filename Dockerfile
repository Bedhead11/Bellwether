# BELLWETHER — self-hostable drift detection. Minimal image exposing the `bellwether` CLI.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

# Run the benchmark by default; override with e.g. `docker run <img> dashboard -o /out/d.html`.
ENTRYPOINT ["bellwether"]
CMD ["benchmark", "--quick"]
