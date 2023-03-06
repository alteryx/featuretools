import numpy as np
import pandas as pd
import tiktoken
from openai.embeddings_utils import get_embedding
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double, NaturalLanguage

from featuretools.primitives.base import TransformPrimitive
from featuretools.primitives.standard.transform.natural_language.openai.model import (
    OpenAIEmbeddingModel,
)

DEFAULT_MODEL = OpenAIEmbeddingModel(
    name="text-embedding-ada-002",
    encoding="cl100k_base",
    max_tokens=8191,
    output_dimensions=1536,
)


class OpenAIEmbeddings(TransformPrimitive):
    """Generates embeddings using OpenAI.

    Description:
        Given list of strings, determine the embeddings for each string, using
        the OpenAI model.

    Args:
        model (OpenAIEmbeddingModel, optional): The model to use to produce embeddings.
            Defaults to "text-embedding-ada-002" if not specified.

    Examples:
        >>> x = ['This is a test file', 'This is second line', 'third line $1,000', None]
        >>> openai_embeddings = OpenAIEmbeddings()
        >>> openai_embeddings(x).tolist()
        [4.0, 4.0, 5.0, nan]
    """

    name = "openai_embeddings"
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})

    def __init__(self, model=DEFAULT_MODEL):
        self.model = model
        self.number_output_features = model.output_dimensions

    def get_function(self):
        encoding = tiktoken.get_encoding(self.model.encoding)

        def is_too_many_tokens(element):
            return len(encoding.encode(element)) > self.model.max_tokens

        def get_openai_embeddings(series):
            invalid = [np.nan] * self.number_output_features
            result = []
            for element in series:
                if pd.isnull(element) or is_too_many_tokens(element):
                    result.append(invalid)
                else:
                    embedding = get_embedding(element, engine=self.model.name)
                    result.append(embedding)
            result = np.array(result).T.tolist()
            return pd.Series(result)

        return get_openai_embeddings
