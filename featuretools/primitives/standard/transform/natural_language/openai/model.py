class OpenAIModel(object):
    """A model accessible via the OpenAI API."""

    def __init__(self, name, encoding, max_tokens):
        self.name = name
        self.encoding = encoding
        self.max_tokens = max_tokens
        pass


class OpenAIEmbeddingModel(OpenAIModel):
    """A model accessible via the OpenAI API that can produce embeddings."""

    def __init__(self, name, encoding, max_tokens, output_dimensions):
        self.output_dimensions = output_dimensions
        super().__init__(name, encoding, max_tokens)
