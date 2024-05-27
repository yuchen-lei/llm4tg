# %%
import yaml
import dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
import tiktoken

dotenv.load_dotenv()


llms: dict[str, ChatOpenAI] = {}
tokenizer = tiktoken.get_encoding("cl100k_base")


def token_count(text: str):
    return len(tokenizer.encode(text))


def get_llm_model(model_name="gpt-4-turbo-preview"):
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


def query_node_info(edge_info, llm: ChatOpenAI):
    prompt_tmpl = tmpl2prompt("prompt_tmpl/query_inout.yaml")
    chain = prompt_tmpl | llm | JsonOutputParser()
    return chain.ainvoke({"edges": edge_info})


def query_graph_characteristics(graph_ref, current_graph, llm: ChatOpenAI):
    prompt_tmpl = tmpl2prompt("prompt_tmpl/query_graph_bytable.yaml")
    chain = prompt_tmpl | llm | JsonOutputParser()
    return chain.ainvoke(
        {
            "features": graph_ref,
            "graph": current_graph,
        }
    )


def query_graph_characteristics_raw(graph_ref, current_graph, llm: ChatOpenAI):
    prompt_tmpl = tmpl2prompt("prompt_tmpl/query_graph_raw.yaml")
    chain = prompt_tmpl | llm | JsonOutputParser()
    return chain.ainvoke(
        {
            "graph_ref": graph_ref,
            "graph": current_graph,
        }
    )


def categorize_by_features(graph_ref, current_graph, llm: ChatOpenAI):
    prompt_tmpl = tmpl2prompt("prompt_tmpl/categorize_by_features.yaml")
    chain = prompt_tmpl | llm | JsonOutputParser()
    return chain.ainvoke(
        {
            "features": graph_ref,
            "graph": current_graph,
        }
    )


def categorize_by_graph(graph_ref, current_graph, llm: ChatOpenAI):
    prompt_tmpl = tmpl2prompt("prompt_tmpl/categorize_by_graph.yaml")
    chain = prompt_tmpl | llm | JsonOutputParser()
    return chain.ainvoke(
        {
            "graph_ref": graph_ref,
            "graph": current_graph,
        }
    )


def distingush_graphs(graph_list, llm: ChatOpenAI, prompt_file="categorize_by_graph"):
    prompt_tmpl = tmpl2prompt(f"prompt_tmpl/{prompt_file}.yaml")
    chain = prompt_tmpl | llm | JsonOutputParser()
    return chain.ainvoke(
        {
            "graphs": graph_list,
        }
    )


# nodes = """
#     to n0 {'time': 1589440969, 'value': 0.0001}
#     to n2 {'time': 1589440969, 'value': 1.17360205}
#     to n3 {'time': 1589440969, 'value': 3.17180975}
#     to n4 {'time': 1589440969, 'value': 3.31838625}
#     to n5 {'time': 1589440969, 'value': 4.01939776}
#     to n6 {'time': 1589440969, 'value': 4.94629228}
#     from n7 {'time': 1589440969, 'value': 16.62999292}
#     """

# query_node_info(nodes)

# %%
