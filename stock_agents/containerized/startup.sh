# startup.sh
#!/bin/bash

echo "Starting Efficient Stock Analysis Agent..."

# Get model from environment variable (default to phi3:mini)
LLM_MODEL=${LLM_MODEL:-phi3:mini}

# Wait for Ollama to be ready
echo "Waiting for Ollama service..."
until curl -f http://ollama:11434/api/health > /dev/null 2>&1; do
    echo "Ollama not ready yet, waiting..."
    sleep 5
done

echo "Ollama is ready!"

# Pull the efficient model
echo "Pulling $LLM_MODEL model (efficient for Docker)..."
curl -X POST http://ollama:11434/api/pull \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"$LLM_MODEL\"}" || echo "Model pull failed, continuing..."

# Wait for model to be ready
echo "Waiting for model to be ready..."
sleep 15

# Verify model is available
echo "Verifying model availability..."
curl -X POST http://ollama:11434/api/generate \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$LLM_MODEL\", \"prompt\": \"Test\", \"stream\": false}" || echo "Model test failed"

# Check if CSV file exists
if [ ! -f "/app/data/NSE_ALL.csv" ]; then
    echo "Warning: NSE_ALL.csv not found in /app/data/"
    echo "Make sure to mount your CSV file to /app/data/NSE_ALL.csv"
fi

# Run the stock analysis
echo "Starting stock market analysis with $LLM_MODEL..."
cd /app
python stock_agent.py

echo "Analysis complete! Check /app/output for results."
