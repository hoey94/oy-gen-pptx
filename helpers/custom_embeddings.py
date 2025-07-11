# import re
# import os
# import jieba
# import pickle
# import string
# import nltk
# import requests
# import yaml
#
# from enum import Enum
# from typing import List, Tuple, Union
# from langchain.embeddings.base import Embeddings
# from langchain.schema import Document, BaseMessage
# from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings, OpenAIEmbeddings
# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
#
# # from nacos_config import get_new_nacos_info
#
# # try:
# #     nltk.data.path.append('./toolkit/nltk_data')
# #     nltk.data.find("stopwords")
# # except LookupError:
# #     nltk.download("stopwords")
# #
# # ChatTurnType = Union[Tuple[str, str], BaseMessage]
# # _ROLE_MAP = {"human": "Human: ", "ai": "Assistant: "}
# #
# # # 初始化分词库和词性标注库
# # jieba.initialize()
#
# # 设置环境变量以避免 tokenizers 警告
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# # get_new_nacos_info()
#
#
# class Config:
#     """Initializes configs from a YAML file."""
#
#     def __init__(self, config_file: str):
#         with open(config_file, 'r') as file:
#             self.config = yaml.safe_load(file)
#
#         # Directory
#         # self.docs_dir = self.config['directory']['DOCS_DIR']
#         # self.db_dir = self.config['directory']['DB_DIR']
#
#         # Parameters
#         self.model_name = self.config['parameters']['MODEL_NAME']
#         self.temperature = self.config['parameters']['TEMPERATURE']
#         self.max_chat_history = self.config['parameters']['MAX_CHAT_HISTORY']
#         self.max_llm_context = self.config['parameters']['MAX_LLM_CONTEXT']
#         self.max_llm_generation = self.config['parameters']['MAX_LLM_GENERATION']
#         self.embedding_name = self.config['parameters']['EMBEDDING_NAME']
#
#         self.n_gpu_layers = self.config['parameters']['N_GPU_LAYERS']
#         self.n_batch = self.config['parameters']['N_BATCH']
#
#         self.base_chunk_size = self.config['parameters']['BASE_CHUNK_SIZE']
#         self.chunk_overlap = self.config['parameters']['CHUNK_OVERLAP']
#         self.chunk_scale = self.config['parameters']['CHUNK_SCALE']
#         self.window_steps = self.config['parameters']['WINDOW_STEPS']
#         self.window_scale = self.config['parameters']['WINDOW_SCALE']
#
#         # Ensure retriever_weights is a list of floats
#         self.retriever_weights = self.config['parameters']['RETRIEVER_WEIGHTS']
#         if isinstance(self.retriever_weights, list):
#             self.retriever_weights = [float(x) for x in self.retriever_weights]
#
#         self.first_retrieval_k = self.config['parameters']['FIRST_RETRIEVAL_K']
#         self.second_retrieval_k = self.config['parameters']['SECOND_RETRIEVAL_K']
#         self.num_windows = self.config['parameters']['NUM_WINDOWS']
#         self.qa_time_limit = self.config['parameters']['qa_time_limit']
#
#         # # mysql
#         # self.database_ip = self.config['mysql_info']['database_ip']
#         # self.database_port = self.config['mysql_info']['database_port']
#         # self.database_name = self.config['mysql_info']['database_name']
#         # self.database_user = self.config['mysql_info']['database_user']
#         # self.database_password = self.config['mysql_info']['database_password']
#         # self.database_table_name = self.config['mysql_info']['table_name']
#         # self.database_sp_table_name = self.config['mysql_info']['table_sp_noun_explain']
#         #
#         # # jzss_mysql
#         # self.jzss_database_ip = self.config['jzss_mysql']['database_ip']
#         # self.jzss_database_port = self.config['jzss_mysql']['database_port']
#         # self.jzss_database_name = self.config['jzss_mysql']['database_name']
#         # self.jzss_database_user = self.config['jzss_mysql']['database_user']
#         # self.jzss_database_password = self.config['jzss_mysql']['database_password']
#
#         # llm
#         # self.llm_url = self.config['llm_info']['url']
#         # self.llm_key = self.config['llm_info']['key']
#         # self.llm_api_model = self.config['llm_info']['model']
#
#         # embedding
#         self.embedding_url = self.config['embedding_info']['url']
#         self.embedding_key = self.config['embedding_info']['key']
#         self.embedding_api_model = self.config['embedding_info']['model']
#
#         # reranker
#         # self.reranker_url = self.config['reranker_info']['url']
#         # self.reranker_key = self.config['reranker_info']['key']
#         # self.reranker_api_model = self.config['reranker_info']['model']
#
#         # child_session
#         self.session_len = self.config['child_session']['session_len']
#
#
# cur_env = os.environ.get("cur_env", "test")
# # configs = Config(f"./config/{cur_env}_configparser.ini")
# configs = Config(f"./config/{cur_env}_config.yaml")
# # configs = Config(f"../config/{cur_env}_config.yaml")
#
#
# class CustomAPIEmbeddings(Embeddings):
#     def __init__(
#         self,
#         api_url=configs.embedding_url,
#         api_key=configs.embedding_key,
#         api_model=configs.embedding_api_model,
#     ):
#         self.api_url = api_url
#         self.api_key = api_key
#         self.api_model = api_model
#
#     def get_embedding_batch(self, texts):
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json",
#         }
#         data = {"model": self.api_model, "input": texts}
#
#         try:
#             response = requests.post(self.api_url, headers=headers, json=data)
#             response.raise_for_status()  # 如果响应状态码不是200，抛出HTTPError异常
#             result = response.json()
#             # 假设响应结构为 {"data": [{"embedding": [...]}]}
#             embeddings = [item.get("embedding") for item in result.get("data", [])]
#             if len(embeddings) != len(texts):
#                 print(
#                     f"Unexpected number of embeddings returned: expected {len(texts)}, got {len(embeddings)}"
#                 )
#             return embeddings
#         except requests.exceptions.HTTPError as http_err:
#             print(f"HTTP error occurred: {http_err} - Response: {response.text}")
#         except Exception as err:
#             print(f"Other error occurred: {err}")
#
#         return [None] * len(texts)
#
#     def embed_documents_multiprocess(self, texts, batch_size=10, max_workers=None):
#         if max_workers is None:
#             max_workers = os.cpu_count() or 4
#         all_embeddings = []
#         with ProcessPoolExecutor(max_workers=max_workers) as executor:
#             futures = []
#             for i in range(0, len(texts), batch_size):
#                 batch_texts = texts[i : i + batch_size]
#                 futures.append(executor.submit(self.get_embedding_batch, batch_texts))
#             for future in futures:
#                 batch_embeddings = future.result()
#                 all_embeddings.extend(batch_embeddings)
#         return all_embeddings
#
#     def embed_documents(self, texts):
#         return self.embed_documents_multiprocess(texts)
#
#     def embed_query(self, text):
#         return self.get_embedding_batch([text])[0]
#
#
# # def choose_embeddings(embedding_name):
# #     """
# #     根据给定的嵌入名称选择相应的嵌入模型。
# #
# #     参数：
# #         embedding_name (str): 嵌入名称
# #
# #     返回：
# #         embeddings: 选择的嵌入模型
# #
# #     异常：
# #         ValueError: 如果给定的嵌入名称不受支持
# #
# #     """
# #     try:
# #         if embedding_name == "openAIEmbeddings":
# #             return OpenAIEmbeddings()
# #         elif embedding_name == "hkunlpInstructorLarge":
# #             device = check_device()
# #             return HuggingFaceInstructEmbeddings(
# #                 model_name="hkunlp/instructor-large", model_kwargs={"device": device}
# #             )
# #         elif embedding_name == "customAPIEmbeddings":
# #             return CustomAPIEmbeddings()
# #         else:
# #             embeddings = HuggingFaceEmbeddings(model_name=embedding_name)
# #             embeddings.client = sentence_transformers.SentenceTransformer(
# #                 embeddings.model_name, device="cuda:0"
# #             )
# #             return embeddings
# #     except Exception as error:
# #         print(error)
# #         raise ValueError(f"Embedding {embedding_name} not supported") from error
# #
# #
# # def load_embedding(store_name, embedding, suffix, path):
# #     """Load chroma embeddings"""
# #     # vector_store = Chroma(
# #     #     persist_directory=f"{path}/chroma_{store_name}_{suffix}",
# #     #     embedding_function=embedding,
# #     # )
# #     from langchain_community.vectorstores import FAISS
# #
# #     # vector_store = FAISS.load_local(
# #     #     f"{path}/chroma_{store_name[1:]}_{suffix}", embedding
# #     # )
# #     # 如果自定义
# #     print(f"{path}/{store_name[:]}_chunks_small")
# #     vector_store = FAISS.load_local(f"{path}/{store_name[:]}_chunks_small", embedding, allow_dangerous_deserialization=True)
# #
# #     return vector_store
# #
# #
# # def load_pickle(prefix, suffix, path, embedding_name):
# #     """Load langchain documents from a pickle file.
# #
# #     Args:
# #         store_name (str): The name of the store where data is saved.
# #         suffix (str): Suffix to append to the store name.
# #         path (str): The path where the pickle file is stored.
# #
# #     Returns:
# #         Document: documents from the pickle file
# #     """
# #     with open(f"{path}/{embedding_name}_{suffix}/{prefix}_{suffix}.pkl", "rb") as file:
# #         return pickle.load(file)
# #
# #
# # def clean_text(text):
# #     """
# #     Converts text to lowercase, removes punctuation, stopwords, and lemmatizes it
# #     for BM25 retriever.
# #
# #     Parameters:
# #         text (str): The text to be cleaned.
# #
# #     Returns:
# #         str: The cleaned and lemmatized text.
# #     """
# #
# #     # remove [SEP] in the text
# #     text = text.replace("[SEP]", "")
# #     # Tokenization
# #     # 加载停用词表
# #     stopwords = []
# #     with open("./toolkit/cn_stopwords.txt", "r", encoding="utf-8") as f:
# #         for line in f:
# #             stopwords.append(line.strip())
# #
# #     tokens = jieba.cut(text)
# #     # Lowercasing
# #     tokens = [w.lower() for w in tokens]
# #     # Remove punctuation
# #     table = str.maketrans("", "", string.punctuation)
# #     stripped = [w.translate(table) for w in tokens]
# #     # Keep tokens that are alphabetic, numeric, or contain both.
# #     words = [
# #         word
# #         for word in stripped
# #         # if word.isalpha()
# #         # or word.isdigit()
# #         # or (re.search("\d", word) and re.search("[a-zA-Z]", word))
# #     ]
# #     # Remove stopwords
# #     stop_words = stopwords
# #     words = [w for w in words if w not in stop_words]
# #     # # Lemmatization (or you could use stemming instead)
# #     # lemmatizer = WordNetLemmatizer()
# #     # lemmatized = [lemmatizer.lemmatize(w) for w in words]
# #     # # Convert list of words to a string
# #     # lemmatized_ = " ".join(lemmatized)
# #     # lemmatized_words = []
# #     # for word in words:
# #     #     pos = pynlpir.segment(word)[0][1]  # 获取词性
# #     #     if pos.startswith('n'):  # 名词
# #     #         pynlpir.
# #     #         lemmatized_words.append(pynlpir.noun_lemmatize(word))
# #     #     elif pos.startswith('v'):  # 动词
# #     #         lemmatized_words.append(pynlpir.verb_lemmatize(word))
# #     #     else:
# #     #         lemmatized_words.append(word)
# #     lemmatized_words = words
# #     lemmatized_ = " ".join(lemmatized_words)
# #
# #     return lemmatized_
# #
# #
# # class IndexerOperator(Enum):
# #     """
# #     Enumeration for different query operators used in indexing.
# #     """
# #
# #     EQ = "=="
# #     GT = ">"
# #     GTE = ">="
# #     LT = "<"
# #     LTE = "<="
# #
# #
# # class DocIndexer:
# #     """
# #     A class to handle indexing and searching of documents.
# #
# #     Attributes:
# #         documents (List[Document]): List of documents to be indexed.
# #     """
# #
# #     def __init__(self, documents):
# #         self.documents = documents
# #         self.index = self.build_index(documents)
# #
# #     def build_index(self, documents):
# #         """
# #         Build an index for the given list of documents.
# #
# #         Parameters:
# #             documents (List[Document]): The list of documents to be indexed.
# #
# #         Returns:
# #             dict: The built index.
# #         """
# #         index = {}
# #         for doc in documents:
# #             for key, value in doc.metadata.items():
# #                 if key not in index:
# #                     index[key] = {}
# #                 if value not in index[key]:
# #                     index[key][value] = []
# #                 index[key][value].append(doc)
# #         return index
# #
# #     def retrieve_metadata(self, search_dict):
# #         """
# #         Retrieve documents based on the search criteria provided in search_dict.
# #
# #         Parameters:
# #             search_dict (dict): Dictionary specifying the search criteria.
# #                                 It can contain "AND" or "OR" operators for
# #                                 complex queries.
# #
# #         Returns:
# #             List[Document]: List of documents that match the search criteria.
# #         """
# #         if "AND" in search_dict:
# #             return self._handle_and(search_dict["AND"])
# #         elif "OR" in search_dict:
# #             return self._handle_or(search_dict["OR"])
# #         else:
# #             return self._handle_single(search_dict)
# #
# #     def _handle_and(self, search_dicts):
# #         results = [self.retrieve_metadata(sd) for sd in search_dicts]
# #         if results:
# #             intersection = set.intersection(
# #                 *[set(map(self._hash_doc, r)) for r in results]
# #             )
# #             return [self._unhash_doc(h) for h in intersection]
# #         else:
# #             return []
# #
# #     def _handle_or(self, search_dicts):
# #         results = [self.retrieve_metadata(sd) for sd in search_dicts]
# #         union = set.union(*[set(map(self._hash_doc, r)) for r in results])
# #         return [self._unhash_doc(h) for h in union]
# #
# #     def _handle_single(self, search_dict):
# #         unions = []
# #         for key, query in search_dict.items():
# #             operator, value = query
# #             union = set()
# #             if operator == IndexerOperator.EQ:
# #                 if key in self.index and value in self.index[key]:
# #                     union.update(map(self._hash_doc, self.index[key][value]))
# #             else:
# #                 if key in self.index:
# #                     for k, v in self.index[key].items():
# #                         if (
# #                             (operator == IndexerOperator.GT and k > value)
# #                             or (operator == IndexerOperator.GTE and k >= value)
# #                             or (operator == IndexerOperator.LT and k < value)
# #                             or (operator == IndexerOperator.LTE and k <= value)
# #                         ):
# #                             union.update(map(self._hash_doc, v))
# #             if union:
# #                 unions.append(union)
# #
# #         if unions:
# #             intersection = set.intersection(*unions)
# #             return [self._unhash_doc(h) for h in intersection]
# #         else:
# #             return []
# #
# #     def _hash_doc(self, doc):
# #         return (doc.page_content, frozenset(doc.metadata.items()))
# #
# #     def _unhash_doc(self, hashed_doc):
# #         page_content, metadata = hashed_doc
# #         return Document(page_content=page_content, metadata=dict(metadata))
# #
# #
# # def _get_chat_history(chat_history: List[ChatTurnType]) -> str:
# #     buffer = ""
# #     for dialogue_turn in chat_history:
# #         if isinstance(dialogue_turn, BaseMessage):
# #             role_prefix = _ROLE_MAP.get(dialogue_turn.type, f"{dialogue_turn.type}: ")
# #             buffer += f"\n{role_prefix}{dialogue_turn.content}"
# #         elif isinstance(dialogue_turn, tuple):
# #             human = "Human: " + dialogue_turn[0]
# #             ai = "Assistant: " + dialogue_turn[1]
# #             buffer += "\n" + "\n".join([human, ai])
# #         else:
# #             raise ValueError(
# #                 f"Unsupported chat history format: {type(dialogue_turn)}."
# #                 f" Full chat history: {chat_history} "
# #             )
# #     return buffer
# #
# #
# # def _get_standalone_questions_list(
# #     standalone_questions_str: str, original_question: str
# # ) -> List[str]:
# #     pattern = r"\d+\.\s(.*?)(?=\n\d+\.|\n|$)"
# #
# #     matches = [
# #         match.group(1) for match in re.finditer(pattern, standalone_questions_str)
# #     ]
# #     if matches:
# #         return matches
# #
# #     match = re.search(
# #         r"(?i)standalone[^\n]*:\n(.*)", standalone_questions_str, re.DOTALL
# #     )
# #     sentence_source = match.group(1).strip() if match else standalone_questions_str
# #     sentences = sentence_source.split("\n")
# #
# #     return [
# #         re.sub(
# #             r"^\((\d+)\)\.? ?|^\d+\.? ?\)?|^(\d+)\) ?|^(\d+)\) ?|^[Qq]uery \d+: ?|^[Qq]uery: ?",
# #             "",
# #             sentence.strip(),
# #         )
# #         for sentence in sentences
# #         if sentence.strip()
# #     ]
