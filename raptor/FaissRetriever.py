import random
from concurrent.futures import ProcessPoolExecutor

import faiss
import numpy as np
import tiktoken
from tqdm import tqdm

from .EmbeddingModels import BaseEmbeddingModel, OpenAIEmbeddingModel
from .Retrievers import BaseRetriever
from .utils import split_text


class FaissRetrieverConfig:
    def __init__(
        self,
        max_tokens=100,
        max_context_tokens=3500,
        use_top_k=False,
        embedding_model=None,
        question_embedding_model=None,
        top_k=5,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        embedding_model_string=None,
    ):
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if max_context_tokens is not None and max_context_tokens < 1:
            raise ValueError("max_context_tokens must be at least 1 or None")

        if embedding_model is not None and not isinstance(
            embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel or None"
            )

        if question_embedding_model is not None and not isinstance(
            question_embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "question_embedding_model must be an instance of BaseEmbeddingModel or None"
            )

        self.top_k = top_k
        self.max_tokens = max_tokens
        self.max_context_tokens = max_context_tokens
        self.use_top_k = use_top_k
        self.embedding_model = embedding_model or OpenAIEmbeddingModel()
        self.question_embedding_model = question_embedding_model or self.embedding_model
        self.tokenizer = tokenizer
        self.embedding_model_string = embedding_model_string or "OpenAI"

    def log_config(self):
        config_summary = """
		FaissRetrieverConfig:
			Max Tokens: {max_tokens}
			Max Context Tokens: {max_context_tokens}
			Use Top K: {use_top_k}
			Embedding Model: {embedding_model}
			Question Embedding Model: {question_embedding_model}
			Top K: {top_k}
			Tokenizer: {tokenizer}
			Embedding Model String: {embedding_model_string}
		""".format(
            max_tokens=self.max_tokens,
            max_context_tokens=self.max_context_tokens,
            use_top_k=self.use_top_k,
            embedding_model=self.embedding_model,
            question_embedding_model=self.question_embedding_model,
            top_k=self.top_k,
            tokenizer=self.tokenizer,
            embedding_model_string=self.embedding_model_string,
        )
        return config_summary


class FaissRetriever(BaseRetriever):
    """
    FaissRetriever is a class that retrieves similar context chunks for a given query using Faiss.
    encoders_type is 'same' if the question and context encoder is the same,
    otherwise, encoders_type is 'different'.
    """

    def __init__(self, config):
        self.embedding_model = config.embedding_model
        self.question_embedding_model = config.question_embedding_model
        self.index = None
        self.context_chunks = None
        self.max_tokens = config.max_tokens
        self.max_context_tokens = config.max_context_tokens
        self.use_top_k = config.use_top_k
        self.tokenizer = config.tokenizer
        self.top_k = config.top_k
        self.embedding_model_string = config.embedding_model_string

    def build_from_text(self, doc_text):
        """
        Builds the index from a given text.

        :param doc_text: A string containing the document text.
        :param tokenizer: A tokenizer used to split the text into chunks.
        :param max_tokens: An integer representing the maximum number of tokens per chunk.
        """
        self.context_chunks = np.array(
            split_text(doc_text, self.tokenizer, self.max_tokens)
        )

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self.embedding_model.create_embedding, context_chunk)
                for context_chunk in self.context_chunks
            ]

        self.embeddings = []
        for future in tqdm(futures, total=len(futures), desc="Building embeddings"):
            self.embeddings.append(future.result())

        self.embeddings = np.array(self.embeddings, dtype=np.float32)

        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def build_from_leaf_nodes(self, leaf_nodes):
        """
        Builds the index from a given text.

        :param doc_text: A string containing the document text.
        :param tokenizer: A tokenizer used to split the text into chunks.
        :param max_tokens: An integer representing the maximum number of tokens per chunk.
        """

        self.context_chunks = [node.text for node in leaf_nodes]

        self.embeddings = np.array(
            [node.embeddings[self.embedding_model_string] for node in leaf_nodes],
            dtype=np.float32,
        )

        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def sanity_check(self, num_samples=4):
        """
        Perform a sanity check by recomputing embeddings of a few randomly-selected chunks.

        :param num_samples: The number of samples to test.
        """
        indices = random.sample(range(len(self.context_chunks)), num_samples)

        for i in indices:
            original_embedding = self.embeddings[i]
            recomputed_embedding = self.embedding_model.create_embedding(
                self.context_chunks[i]
            )
            assert np.allclose(
                original_embedding, recomputed_embedding
            ), f"Embeddings do not match for index {i}!"

        print(f"Sanity check passed for {num_samples} random samples.")

    def retrieve(self, query: str) -> str:
        """
        Retrieves the k most similar context chunks for a given query.

        :param query: A string containing the query.
        :param k: An integer representing the number of similar context chunks to retrieve.
        :return: A string containing the retrieved context chunks.
        """
        query_embedding = np.array(
            [
                np.array(
                    self.question_embedding_model.create_embedding(query),
                    dtype=np.float32,
                ).squeeze()
            ]
        )

        context = ""

        if self.use_top_k:
            _, indices = self.index.search(query_embedding, self.top_k)
            for i in range(self.top_k):
                context += self.context_chunks[indices[0][i]]

        else:
            range_ = int(self.max_context_tokens / self.max_tokens)
            _, indices = self.index.search(query_embedding, range_)
            total_tokens = 0
            for i in range(range_):
                tokens = len(self.tokenizer.encode(self.context_chunks[indices[0][i]]))
                context += self.context_chunks[indices[0][i]]
                if total_tokens + tokens > self.max_context_tokens:
                    break
                total_tokens += tokens

        return context
