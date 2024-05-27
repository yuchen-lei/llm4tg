import logging
import typing

import dotenv
from openai import RateLimitError
import tenacity
import tiktoken
import yaml
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

TOKENIZERS = typing.Literal[
    "cl100k_base",
    "o200k_base",
]

MODELS = typing.Literal[
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]

llms: dict[str, ChatOpenAI] = {}
tokenizers: dict[TOKENIZERS, tiktoken.Encoding] = {}


def token_count(text: str, tokenizer: TOKENIZERS = "cl100k_base"):
    global tokenizers
    if not tokenizers.get(tokenizer):
        tokenizers[tokenizer] = tiktoken.get_encoding(tokenizer)
    return len(tokenizers[tokenizer].encode(text))


def get_llm_model(model_name: MODELS):
    global llms
    if not llms.get(model_name):
        llms[model_name] = ChatOpenAI(
            model=model_name,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    return llms[model_name]


def tmpl2prompt(file_path: str):
    with open(file_path) as f:
        tmpl = yaml.safe_load(f)
    return ChatPromptTemplate.from_messages([(x["role"], x["msg"]) for x in tmpl])


logger = logging.getLogger(__name__)


@tenacity.retry(
    wait=tenacity.wait_random(40, 80),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
)
async def aquery_once(llm: ChatOpenAI, tmpl_name: str, **prompt_params):
    prompt_tmpl = tmpl2prompt(f"prompt_tmpl/{tmpl_name}.yaml")
    chain = prompt_tmpl | llm | JsonOutputParser()
    try:
        return await chain.ainvoke(prompt_params)
    except RateLimitError as ex:
        # ex.message = f"RateLimitError: {ex.response.json()["error"]["message"]}"
        raise ex.message
    # return await chain.ainvoke(prompt_params)


@tenacity.retry(
    wait=tenacity.wait_random(40, 80),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
)
def query_once(llm: ChatOpenAI, tmpl_name: str, **prompt_params):
    prompt_tmpl = tmpl2prompt(f"prompt_tmpl/{tmpl_name}.yaml")
    chain = prompt_tmpl | llm | JsonOutputParser()
    return chain.invoke(prompt_params)
