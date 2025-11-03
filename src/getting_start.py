import dspy
from typing import Literal
import mlflow


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


def task_math():
    math = dspy.ChainOfThought("question -> answer: float")
    question = "Two dice are tossed. What is the probability that the sum equals two?"
    response = math(question=question)
    print(f"{question=}")
    print(f"{response=}")


def search_wikipedia(query: str) -> list[str]:
    """
    RAG のサーバーがうまく動作しない（2025/11/03）
    """
    base_url = "http://127.0.0.1:2017/search"
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(url=base_url)
    results = colbertv2_wiki17_abstracts(query, k=3)
    return [x["text"] for x in results]


def task_rag():
    rag = dspy.ChainOfThought("context, question -> response")
    question = "What's the name of the castle that David Gregory inherited?"
    response = rag(context=search_wikipedia(question), question=question)
    print(f"{question=}")
    print(f"{response=}")


class Classify(dspy.Signature):
    """Classify sentiment of a given sentence."""

    sentence: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()
    confidence: float = dspy.OutputField()


def task_classify():
    classify = dspy.Predict(Classify)

    question = "This book was super fun to read, though not the last chapter."
    response = classify(sentence=question)
    print(f"{question=}")
    print(f"{response=}")

    question = "I just absolutely adore this book."
    response = classify(sentence=question)
    print(f"{question=}")
    print(f"{response=}")


class ExtractInfo(dspy.Signature):
    """Extract structured information from text."""

    text: str = dspy.InputField()
    title: str = dspy.OutputField()
    headings: list[str] = dspy.OutputField()
    entities: list[dict[str, str]] = dspy.OutputField(
        desc="a list of entities and their metadata"
    )


def task_extract():
    def print_response():
        print(f"{response.title=}")
        print(f"{response.headings=}")
        print(f"{response.entities=}")

    module = dspy.Predict(ExtractInfo)

    text = (
        "Apple Inc. announced its latest iPhone 14 today."
        "The CEO, Tim Cook, highlighted its new features in a press release."
    )
    response = module(text=text)
    print_response()

    text = """
    ## Transition to 'Next-Generation Standard' Enters Final Phase

    **Key Stakeholders Agree on "Blueprint" Details**

    The implementation process for the long-discussed "Next-Generation Standard" has entered its final stage, according to multiple stakeholders.

    At a closed-door meeting held yesterday, key stakeholders reportedly reached an agreement on the details of an execution plan, dubbed the "Blueprint." A key focus was the integration of "Project Cassiopeia" and "Algorithm V3," but sources indicate the technical hurdles have been cleared.

    One expert noted, "This agreement is a significant step toward securing 'interoperability,' but the real test will be in the implementation phase."

    Authorities are expected to release a formal roadmap soon, but specific impacts on the market remain largely unclear at this time.
    """
    response = module(text=text)
    print_response()


def task_agent():
    def evaluate_math(expression: str):
        print(f"evaluate_math(): {expression=}")
        result = dspy.PythonInterpreter({}).execute(expression)
        print(f"evaluate_math(): {result=}")
        return result

    react = dspy.ReAct(
        "question -> answer: float", tools=[evaluate_math, search_wikipedia]
    )
    # print(f"{react=}")
    question = "What is 9362158 divided by the year of birth of David Robert Mitchell"
    pred = react(question=question)
    print(f"{pred.answer=}")
    # print(f"{pred=}")


# multi stage pipeline demo
class Outline(dspy.Signature):
    """Outline a thorough overview of a topic."""

    topic: str = dspy.InputField()
    title: str = dspy.OutputField()
    sections: list[str] = dspy.OutputField()
    section_subheadings: dict[str, list[str]] = dspy.OutputField(
        desc="mapping from section headings to subheadings(日本語で出力)"
    )


class DraftSection(dspy.Signature):
    """Draft a top-level section of an article."""

    topic: str = dspy.InputField()
    section_heading: str = dspy.InputField()
    section_subheadings: list[str] = dspy.InputField()
    content: str = dspy.OutputField(desc="markdown-formatted section(日本語で出力)")


class DraftArticle(dspy.Module):
    def __init__(self):
        self.build_outline = dspy.ChainOfThought(Outline)
        self.draft_section = dspy.ChainOfThought(DraftSection)

    def forward(self, topic):
        outline = self.build_outline(topic=topic)
        sections = []
        for heading, subheadings in outline.section_subheadings.items():
            section, subheadings = (
                f"## {heading}",
                [f"### {subheading}" for subheading in subheadings],
            )
            section = self.draft_section(
                topic=outline.title,
                section_heading=section,
                section_subheadings=subheadings,
            )
            sections.append(section.content)
        return dspy.Prediction(title=outline.title, sections=sections)


def task_multi_stage_pipeline():
    draft_article = DraftArticle()
    print(f"{draft_article=}")
    article = draft_article(topic="世界ラリー選手権 2023")
    print(f"{article=}")


def main():
    lm = init_dspy()
    print(f"{dspy=}")
    print(f"{lm=}")
    # task_math()
    # task_rag()
    # task_classify()
    # task_extract()
    # task_agent()
    task_multi_stage_pipeline()


if __name__ == "__main__":
    main()
