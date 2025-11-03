# setup ollama
setup_ollama:
	curl -fsSL https://ollama.ai/install.sh | sh
	ollama pull llama3.2:1b
	ollama pull llama3.2:3b
	ollama pull gpt-oss:20b

# download dataset
download_wikipedia_data:
	mkdir -p data/raw/
	wget https://github.com/j-labs/stfo-colbert/raw/refs/heads/main/example_data/wikipedia_summaries.txt \
    -O data/raw/wikipedia_summaries.txt

# https://github.com/j-labs/stfo-colbert
run_search_engine:
	uv run stfo-colbert \
        --dataset-path ./data/raw/wikipedia_summaries.txt \
        --model-name intfloat/multilingual-e5-small \
        --port 2017

# test search enging
test_search:
	curl "http://127.0.0.1:2017/search?query=David%20Robert%20Mitchell&k=3"
