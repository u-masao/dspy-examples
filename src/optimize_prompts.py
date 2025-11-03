import dspy
import mlflow

from dspy.datasets import HotPotQA


def init_dspy():
    mlflow.dspy.autolog(
        log_compiles=True,
        log_evals=True,
        log_traces_from_compile=True,
    )
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("DSPy-Optim")

    model_name = "ollama_chat/gpt-oss:20b"
    # model_name = "ollama_chat/llama3.2:1b"
    # model_name = "ollama_chat/llama3.2:3b"
    lm = dspy.LM(model_name, api_base="http://localhost:11434", api_key="")
    dspy.configure(lm=lm)
    return lm


def search_wikipedia(query: str) -> list[str]:
    """
    RAG のサーバーがうまく動作しない（2025/11/03）
    """
    base_url = "http://127.0.0.1:2017/search"
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(url=base_url)
    results = colbertv2_wiki17_abstracts(query, k=3)
    return [x["text"] for x in results]


def optimize_prompts_for_a_react_agent():
    trainset = [
        x.with_inputs("question")
        for x in HotPotQA(train_seed=2024, train_size=500).train
    ]
    react = dspy.ReAct("question -> answer", tools=[search_wikipedia])
    tp = dspy.MIPROv2(
        metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24
    )
    optimized_react = tp.compile(react, trainset=trainset)
    print(f"{optimized_react=}")


def main():
    init_dspy()
    optimize_prompts_for_a_react_agent()


if __name__ == "__main__":
    main()
