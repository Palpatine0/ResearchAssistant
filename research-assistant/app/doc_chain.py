from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import json
from langchain.retrievers import ArxivRetriever

retriever = ArxivRetriever()

from dotenv import load_dotenv
from langchain.callbacks.tracers.langchain import wait_for_all_tracers

load_dotenv()
wait_for_all_tracers()

# Text summary pipeline
summary_template = """
{doc} 
-----------
Using the above text, answer in short the following question: 
> {question}
-----------
if the question cannot be answered using the text, imply summarize the text. 
Include all factual information, numbers, stats etc if available.
"""
summary_prompt = ChatPromptTemplate.from_template(summary_template)

scrape_and_summarize_chain = RunnablePassthrough.assign(
    summary = summary_prompt | ChatOpenAI() | StrOutputParser()
) | (lambda x: f"TITLE: {x['doc'].metadata['Title']}\n\nSUMMARY: {x['summary']}")

# Search pipeline
search_question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 google search queries to search online that form an "
            "objective opinion from the following: {question}\n"
            "You must respond with a list of strings in the following format: "
            '["query 1", "query 2", "query 3"].',
        ),
    ]
)

search_question_chain = search_question_prompt | ChatOpenAI(temperature = 0) | StrOutputParser() | json.loads

doc_search_chain = RunnablePassthrough.assign(
    docs = lambda x: retriever.get_summaries_as_docs(x["question"])
) | (lambda x: [{"question": x["question"], "doc": u} for u in x["docs"]]) | scrape_and_summarize_chain.map()

full_research_chain = search_question_chain | (lambda x: [{"question": q} for q in x]) | doc_search_chain.map()


def collapse_list_of_lists(list_of_lists):
    content = []
    for l in list_of_lists:
        content.append("\n\n".join(l))
    return "\n\n".join(content)


writer_system_prompt = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."  # noqa: E501

research_prompt_template = """Information:
--------
{research_summary}
--------
Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
The report should focus on the answer to the question, should be well structured, informative, \
in depth, with facts and numbers if available and a minimum of 1,200 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
Please do your best, this is very important to my career."""  # noqa: E501

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", writer_system_prompt),
        ("user", research_prompt_template),
    ]
)

chain = RunnablePassthrough.assign(
    research_summary = full_research_chain | collapse_list_of_lists
) | prompt | ChatOpenAI() | StrOutputParser()

# Main execution
if __name__ == "__main__":
    print(chain.invoke({
        "question": "What papers did Emil Khalisi write?"
    }))
