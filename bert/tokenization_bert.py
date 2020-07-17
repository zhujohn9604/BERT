import collections
import logging
import os
import unicodedata
from enum import Enum
import torch
import wget
from typing import List, Optional, Union, Tuple, Dict

logger = logging.getLogger(__name__)

# Define type aliases and NamedTuples
TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[str], List[str]]


class ExplicitEnum(Enum):

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            "%r is not a valid %s, please select one of %s"
            % (value, cls.__name__, str(list(cls._value2member_map_.keys())))
        )


class TensorType(ExplicitEnum):
    PYTORCH = "pt"  # ONLY HAS THIS ONE
    TENSORFLOW = "tf"
    NUMPY = "np"


class TruncationStrategy(ExplicitEnum):
    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"


class PaddingStrategy(ExplicitEnum):
    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
        "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
        "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
        "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
        "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
        "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
        "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
        "bert-base-german-cased": "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-vocab.txt",
        "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-vocab.txt",
        "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-vocab.txt",
        "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt",
        "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-vocab.txt",
        "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-vocab.txt",
        "bert-base-german-dbmdz-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-vocab.txt",
        "bert-base-german-dbmdz-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-vocab.txt",
        "TurkuNLP/bert-base-finnish-cased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-cased-v1/vocab.txt",
        "TurkuNLP/bert-base-finnish-uncased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-uncased-v1/vocab.txt",
        "wietsedv/bert-base-dutch-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/wietsedv/bert-base-dutch-cased/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "bert-base-uncased": 512,
    "bert-large-uncased": 512,
    "bert-base-cased": 512,
    "bert-large-cased": 512,
    "bert-base-multilingual-uncased": 512,
    "bert-base-multilingual-cased": 512,
    "bert-base-chinese": 512,
    "bert-base-german-cased": 512,
    "bert-large-uncased-whole-word-masking": 512,
    "bert-large-cased-whole-word-masking": 512,
    "bert-large-uncased-whole-word-masking-finetuned-squad": 512,
    "bert-large-cased-whole-word-masking-finetuned-squad": 512,
    "bert-base-cased-finetuned-mrpc": 512,
    "bert-base-german-dbmdz-cased": 512,
    "bert-base-german-dbmdz-uncased": 512,
    "TurkuNLP/bert-base-finnish-cased-v1": 512,
    "TurkuNLP/bert-base-finnish-uncased-v1": 512,
    "wietsedv/bert-base-dutch-cased": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "bert-base-uncased": {"do_lower_case": True},
    "bert-large-uncased": {"do_lower_case": True},
    "bert-base-cased": {"do_lower_case": False},
    "bert-large-cased": {"do_lower_case": False},
    "bert-base-multilingual-uncased": {"do_lower_case": True},
    "bert-base-multilingual-cased": {"do_lower_case": False},
    "bert-base-chinese": {"do_lower_case": False},
    "bert-base-german-cased": {"do_lower_case": False},
    "bert-large-uncased-whole-word-masking": {"do_lower_case": True},
    "bert-large-cased-whole-word-masking": {"do_lower_case": False},
    "bert-large-uncased-whole-word-masking-finetuned-squad": {"do_lower_case": True},
    "bert-large-cased-whole-word-masking-finetuned-squad": {"do_lower_case": False},
    "bert-base-cased-finetuned-mrpc": {"do_lower_case": False},
    "bert-base-german-dbmdz-cased": {"do_lower_case": False},
    "bert-base-german-dbmdz-uncased": {"do_lower_case": True},
    "TurkuNLP/bert-base-finnish-cased-v1": {"do_lower_case": False},
    "TurkuNLP/bert-base-finnish-uncased-v1": {"do_lower_case": True},
    "wietsedv/bert-base-dutch-cased": {"do_lower_case": False},
}


def load_vocab(vocab_file):
    """Load a vocabulary file into a dictionary"""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """"Run basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:  # null
        return []
    tokens = text.split()
    return tokens


def _is_punctuation(char):
    # Note: Characters such as "^", "$", and "`" are not in the Unicode Punctuation class
    # More info about General Category (Unicode): https://en.wikipedia.org/wiki/Template:General_Category_(Unicode)
    cp = ord(char)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class SpecialTokens(object):
    """This is a light version of SpecialTokensMixin See:
    https://github.com/huggingface/transformers/blob/d12ceb48bad126768e44d2bd958fa7638abd0f16/src/transformers
    /tokenization_utils_base.py#L884

    Input format:
    demo = SpecialTokens(unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]",
                 cls_token="[CLS]", mask_token="[MASK]", additional_special_tokens = ['<BOS>', '<END>'])
    """
    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ]

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == "additional_special_tokens":
                    assert isinstance(value, (list, tuple)) and all(isinstance(t, str) for t in value)
                    setattr(self, key, value)
                else:
                    setattr(self, key, value)

    @property
    def all_special_tokens(self):
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, attr, None)
            if attr_value:
                set_attr[attr] = attr_value

        special_tokens = [t if isinstance(t, (tuple, list)) else [t] for t in set_attr.values()]
        return set([item for sublist in special_tokens for item in sublist])


class PreTrainedTokenizerBase(SpecialTokens):
    # https://github.com/huggingface/transformers/blob/d12ceb48bad126768e44d2bd958fa7638abd0f16/src/transformers/tokenization_utils_base.py
    # https://huggingface.co/transformers/model_doc/bert.html
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.added_tokens_encoder: Dict[str, int] = {}
        self.added_tokens_decoder: Dict[int, str] = {}
        self.pad_token_type_id = 0

    def get_added_vocab(self):
        return self.added_tokens_encoder

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        # see the derived class, e.g., BertTokenizer
        raise NotImplementedError

    def _convert_id_to_token(self, index: int) -> str:
        return NotImplementedError

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens):
        # tokens: a sequence of tokens or a single token
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = [self._convert_token_to_id_with_added_voc(token) for token in tokens]
        return ids

    @property
    def pad_token_id(self):
        return self._convert_token_to_id(self.pad_token)

    def _tokenize(self, text):
        # see the derived class, e.g., BertTokenizer
        raise NotImplementedError

    def encode(self, text: Union[TextInput, PreTokenizedInput, EncodedInput],
               text_pair: Union[TextInputPair, PreTokenizedInputPair, EncodedInputPair] = None,
               add_special_tokens: bool = True, padding: Union[bool, str] = False,
               truncation: Union[bool, str] = False, max_length: Optional[int] = None,
               stride: int = 0,
               return_token_type_ids: Optional[bool] = None,
               return_attention_mask: Optional[bool] = None,
               return_overflowing_tokens: bool = False,
               return_special_tokens_mask: bool = False,
               return_length: bool = False,
               pad_to_max_length: bool = False,
               encode_plus: bool = False,
               return_tensors: bool = False,
               **kwargs):

        if truncation:
            truncation = 'only_first'
        if max_length is not None and not truncation:
            # given max_length without explicitly specify truncation method
            truncation = 'only_first'

        if pad_to_max_length:
            padding = True
            max_length = 512

        if padding:
            # given max_length without explicitly specify truncation method
            padding = "max_length"

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self._tokenize(text)
                return self.convert_tokens_to_ids(tokens)
            else:
                raise ValueError(f'Input should be a string')

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None
        if not encode_plus:
            encoded_inputs = self._prepare_for_model(
                ids=first_ids,
                pair_ids=second_ids,
                add_special_tokens=add_special_tokens,
                padding_strategy=PaddingStrategy.DO_NOT_PAD if not padding else padding,
                truncation_strategy=TruncationStrategy.DO_NOT_TRUNCATE if not truncation else truncation,
                max_length=max_length,
                stride=stride,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=return_tensors,
            )
            return encoded_inputs["input_ids"]
        else:
            encoded_inputs = self._prepare_for_model(
                ids=first_ids,
                pair_ids=second_ids,
                add_special_tokens=add_special_tokens,
                padding_strategy=PaddingStrategy.DO_NOT_PAD if not padding else padding,
                truncation_strategy=TruncationStrategy.DO_NOT_TRUNCATE if not truncation else truncation,
                max_length=max_length,
                stride=stride,
                return_token_type_ids=True,
                return_attention_mask=True,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=return_tensors,
            )
            return encoded_inputs

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        # see the derived class, e.g., BertTokenizer
        raise NotImplementedError

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        # see the derived class, e.g., BertTokenizer
        raise NotImplementedError

    def num_special_tokens_to_add(self, pair_exist=False):
        # return the number of special tokens
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair_exist else None))

    def truncate_sequences(self, ids: List[int], pair_ids: Optional[List[int]] = None,
                           num_tokens_to_remove: int = 0,
                           truncation_strategy: Union[str, TruncationStrategy] = 'only_first',
                           stride: int = 0,
                           ) -> Tuple[List[int], List[int], List[int]]:

        pair_exist = bool(pair_ids is not None)

        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if isinstance(truncation_strategy, str):
            truncation_strategy = TruncationStrategy(truncation_strategy)

        overflowing_tokens = []
        if truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            for _ in range(num_tokens_to_remove):
                if not pair_exist or len(ids) > len(pair_ids):
                    ids = ids[:-1]
                else:
                    pair_ids = pair_ids[:-1]
        elif truncation_strategy == TruncationStrategy.ONLY_FIRST:
            assert len(ids) > num_tokens_to_remove
            window_len = min(len(ids), stride + num_tokens_to_remove)
            overflowing_tokens = ids[-window_len:]
            ids = ids[:-num_tokens_to_remove]
        elif truncation_strategy == TruncationStrategy.ONLY_SECOND:
            assert len(pair_ids) > num_tokens_to_remove
            window_len = min(len(pair_ids), stride + num_tokens_to_remove)
            overflowing_tokens = pair_ids[-window_len:]
            pair_ids = pair_ids[:-num_tokens_to_remove]

        return ids, pair_ids, overflowing_tokens

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, ) -> List[int]:
        raise NotImplementedError

    def pad(self, encoded_inputs: Dict[str, EncodedInput], max_length: Optional[int] = None,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            return_attention_mask: Optional[bool] = None):

        if isinstance(padding_strategy, str):
            padding_strategy = PaddingStrategy(padding_strategy)

        needs_to_be_padded = (
                padding_strategy != PaddingStrategy.DO_NOT_PAD and max_length and len(
            encoded_inputs["input_ids"]) != max_length)

        if needs_to_be_padded:
            padding_len = max_length - len(encoded_inputs["input_ids"])
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"]) + [0] * padding_len
            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * padding_len
                )
            if "special_tokens_mask" in encoded_inputs:
                encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * padding_len
            encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [self.pad_token_id] * padding_len
        else:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])
        return encoded_inputs

    def _prepare_for_model(self, ids: List[int], pair_ids: Optional[List[int]] = None,
                           add_special_tokens: bool = True,
                           padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
                           truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
                           max_length: Optional[int] = None,
                           stride: int = 0,
                           return_token_type_ids: Optional[bool] = None,
                           return_attention_mask: Optional[bool] = None,
                           return_overflowing_tokens: bool = False,
                           return_special_tokens_mask: bool = False,
                           return_length: bool = False,
                           return_tensors: bool = False,
                           ):
        pair_exist = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair_exist else 0
        encoded_inputs = {}

        # Truncation
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair_exist) if add_special_tokens else 0)
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids=ids, pair_ids=pair_ids, num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy, stride=stride, )
            if return_overflowing_tokens:
                encoded_inputs["overflowing_tokens"] = overflowing_tokens
                encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair_exist else ids
            token_type_ids = [0] * len_ids + [1] * len_pair_ids

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Padding
        encoded_inputs = self.pad(encoded_inputs, max_length=max_length,
                                  padding_strategy=padding_strategy,
                                  return_attention_mask=return_attention_mask)

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        if return_tensors:
            for key, value in encoded_inputs.items():
                encoded_inputs[key] = torch.tensor([value])
        return encoded_inputs

    def decode(self, token_ids: List[int]):
        tokens = [self._convert_id_to_token(ids) for ids in token_ids]
        return self.convert_tokens_to_string(tokens)


class BertTokenizer(PreTrainedTokenizerBase):
    """BERT tokenizer based on WordPiece
        For interpretations of args below, please refer to:
        https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_bert.py
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True,
                 never_split=None, unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]",
                 cls_token="[CLS]", mask_token="[MASK]", **kwargs):
        super(BertTokenizer, self).__init__(unk_token=unk_token, sep_token=sep_token,
                                            pad_token=pad_token, cls_token=cls_token,
                                            mask_token=mask_token)

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )

        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if self.do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, never_split=never_split)
        self.wordpiece_tokenizer = WordPieceTokenizer(vocab=self.vocab, unk_token=unk_token)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        # Note: added_tokens_encoder was removed
        return dict(self.vocab)

    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)

        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    @property
    def cls_token_id(self):
        return self._convert_token_to_id(self.cls_token)

    @property
    def sep_token_id(self):
        return self._convert_token_to_id(self.sep_token)

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """Build BERT inputs:
        1. single sequence: [CLS] [SENT] [SEP]
        2. pair of sequences: [CLS] [SENT1] [SEP] [SENT2] [SEP]

        SENT1: token_ids_0, list of ids to which the special tokens will be added.
        SENT2: token_ids_1, optional only for sequence pairs.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        | - special token
        """
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, ) -> List[int]:
        if token_ids_1 is not None:
            return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]
        return [1] + [0] * len(token_ids_0) + [1]

    def save_vocabulary(self, vocab_path):
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as f:
            for token, token_index in sorted(self.vocab.items(), key=lambda x: x[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                f.write(token + "\n")
                index += 1
        return vocab_file

    @classmethod
    def from_pretrained(cls, vocab_file_path):
        if vocab_file_path is not None:
            url = PRETRAINED_VOCAB_FILES_MAP['vocab_file'][vocab_file_path]
            if not os.path.exists(url.split('/')[-1]):
                vocab_file = wget.download(url)  # download
            else:
                vocab_file = url.split('/')[-1]
            return cls(vocab_file=vocab_file)


class BasicTokenizer(object):
    """Run basic tokenization (punctuation splitting, lower casing, etc.).
        Denote: 'tokenize_chinese_chars' functions was removed
        For complete info, see:
        https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_bert.py
    """

    def __init__(self, do_lower_case=True, never_split=None):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)

    def tokenize(self, text, never_split=None):
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, token):
        token = unicodedata.normalize("NFD", token)  # e.g., 'Ç' -> 'C' '̧'
        output = []
        for char in token:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, token, never_split=None):
        if never_split is not None and token in never_split:
            return [token]
        chars = list(token)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]


class WordPieceTokenizer(object):
    """Run WordPiece tokenization"""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    if start > 0:
                        substr = "##" + "".join(chars[start:end])
                    else:
                        substr = "".join(chars[start:end])
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)

        return output_tokens


if __name__ == '__main__':
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    text = "Engineer Will Knowles explained that the first attempt at recreating the theremin sound was fairly " \
           "straightforward: just a 'single oscillator producing a wave at a given frequency.'"
    print(bert_tokenizer.encode(text, return_tensors=True))
    """
    basic_tokenizer = BasicTokenizer()
    text = "Engineer Will Knowles explained that the first attempt at recreating the theremin sound was fairly " \
           "straightforward: just a 'single oscillator producing a wave at a given frequency.'"
    print(basic_tokenizer.tokenize(text))

    import wget

    url = PRETRAINED_VOCAB_FILES_MAP['vocab_file']['bert-base-uncased']
    if not os.path.exists(url.split('/')[-1]):
        vocab_file = wget.download(url)  # download
    else:
        vocab_file = url.split('/')[-1]
    bert_tokenizer = BertTokenizer(vocab_file=vocab_file)
    print(bert_tokenizer._tokenize(text))
    print(bert_tokenizer.encode(text, padding='max_length', max_length=100))
    print(bert_tokenizer.encode(text, padding='max_length', max_length=100))
    print(bert_tokenizer.encode(text, max_length=20))
    print(bert_tokenizer.encode(text, max_length=20, return_tensors=True))

    prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    next_sentence = "The sky is blue due to the shorter wavelength of blue light."
    encoding = bert_tokenizer.encode(prompt, next_sentence, return_tensors=True, encode_plus=True)
    """
