from typing import Union, Optional, List

from transformers import BertTokenizerFast, TensorType
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, \
    EncodedInput, TruncationStrategy
from transformers.utils import PaddingStrategy


MAX_LENGTH = 512


class CustomTokenizer(BertTokenizerFast):

    def encode(self, text: Union[TextInput, PreTokenizedInput, EncodedInput],
               text_pair: Optional[
                   Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
               add_special_tokens: bool = True,
               padding: Union[bool, str, PaddingStrategy] = "max_length",
               truncation: Union[bool, str, TruncationStrategy] = True,
               max_length: Optional[int] = MAX_LENGTH, stride: int = 0,
               return_tensors: Optional[Union[str, TensorType]] = None,
               **kwargs) -> List[int]:
        if not add_special_tokens:
            padding = False
        return super().encode(text, text_pair, add_special_tokens, padding,
                              truncation, max_length, stride, return_tensors,
                              **kwargs)
