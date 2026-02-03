# Build context must be the workspace root (parent of orchestrator/).
# From workspace root: docker build -f orchestrator/Dockerfile -t qtos-api .
# Or: docker-compose -f orchestrator/docker-compose.yml up --build

FROM python:3.11-slim

WORKDIR /app

# Copy entire workspace (orchestrator + sibling agents)
COPY . .

# Install orchestrator dependencies
RUN pip install --no-cache-dir -r orchestrator/requirements.txt

# Install agent dependencies (skip if an agent or its requirements.txt is missing)
RUN for dir in Market-Regime-Agent Portfolio-Analyst-Agent Capital-Allocation-Agent \
    Execution-Discipline-Agent Capital-Guardian-Agent Sentiment-Shift-Alert-Agent \
    Equity-Insider-Intelligence-Agent Trade-Journal-Coach-Agent; do \
  if [ -f "$dir/requirements.txt" ]; then pip install --no-cache-dir -r "$dir/requirements.txt"; fi; \
done

EXPOSE 8000

# Run API; use 0.0.0.0 so it's reachable from outside the container
CMD ["uvicorn", "orchestrator.api:app", "--host", "0.0.0.0", "--port", "8000"]
