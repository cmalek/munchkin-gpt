from functools import cached_property
from pathlib import Path
import pickle
from typing import Any, Literal, Tuple, Optional, List, Dict, cast, Type, Set

from datasets import load_dataset
import numpy as np
import requests
import tiktoken
import torch
from tqdm import tqdm

from .settings import DatasetSettings, TrainingSettings


class CharTokenizer:
    """
    This is a simple character-level tokenizer.  It's used for the ``shakespeare_char`` dataset.
    """

    def __init__(self) -> None:
        #: The name of the tokenizer
        self.name: str = 'char'
        #: The end of text token
        self.eot_token = 0
        #: The maximum token value
        self.max_token_value = 255
        #: The size of the vocabulary
        self.n_vocab: Optional[int] = None
        #: The vocabulary
        self.vocab: Optional[Dict[str, int]] = None
        #: The inverse vocabulary
        self.inverse_vocab: Optional[Set[str]] = None

    def build_vocabulary(self, corpus: str) -> None:
        """
        Given our corpus of text, build our vocabulary.

        Args:
            corpus: the corpus of text
        """
        # get all the unique characters that occur in this text
        chars = sorted(list(set(corpus)))
        self.n_vocab = len(chars)
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self.inverse_vocab = {i: ch for i, ch in enumerate(chars)}

    def load_vocabulary(self, dataset_name: str) -> None:
        """
        Load the vocabulary for the given dataset.

        Args:
            dataset_name: the name of the dataset
        """
        meta_path = Path(__file__).parent.parent / 'etc/dataset' / dataset_name / 'meta.pkl'
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        self.n_vocab = meta['vocab_size']
        self.vocab = meta['stoi']
        self.inverse_vocab = meta['itos']

    def encode_ordinary(
        self,
        text: str,
        allowed_special: Optional[Dict[str, int]] = None
    ) -> List[int]:
        """
        Given a string, return a list of integers representing the string.

        Args:
            text: The string to encode

        Returns:
            A list of integers representing the string
        """
        vocab = cast(Dict[str, int], self.vocab)
        return [vocab[c] for c in text]

    encode = encode_ordinary

    def decode(self, indices: List[int]) -> str:
        """
        Given a list of token indices, return a string.

        Args:
            ids: The list of token indices to decode

        Returns:
            The decoded string
        """
        inverse_vocab = cast(Dict[int, str], self.inverse_vocab)
        return "".join(
            [inverse_vocab[idx] for idx in indices]
        )


class DatasetLoader:
    """
    A base class for loading datasets from an external source, tokenizing them
    and saving them to disk.   We need to preprocess the datasets because
    they're raw text and we need to convert them to a format that's suitable
    for training and validation.

    Keyword Args:
        val_size: The size of the 'val' split as a percentage; the rest is the
            'train' split
        output_dtype: The data type to use when saving the tokenized dataset to
            disk
    """

    #: The encoding function
    encoder: Any = tiktoken.get_encoding('gpt2')
    # The name of the dataset
    dataset_name: Literal['openwebtext', 'shakespeare', 'shakespeare_char']

    def __init__(
        self,
        settings: DatasetSettings,
        **kwargs
    ):
        #: Our dataset settings
        self.settings = settings
        #: The raw dataset itself
        self.dataset: Any = None
        #: The tokenized dataset
        self.tokenized: Any = None
        #: The size of the 'val' split as a percentage; the rest is the 'train' split
        self.val_size = self.settings.val_size
        #: The data type to use when saving the tokenized dataset to disk
        self.output_dtype = self.settings.output_dtype

    @cached_property
    def data(self) -> Any:
        """
        Return the training data.
        """
        if self.tokenized is None:
            self.tokenize()
        return self.tokenized

    @cached_property
    def vocab_size(self) -> int:
        """
        Return the size of the vocabulary.
        """
        if self.meta:
            return self.meta['vocab_size']
        # if we don't have a meta file, assume the GPT-2 124M vocabulary size
        return 50304

    @cached_property
    def meta(self) -> Optional[Dict[str, Any]]:
        """
        Return the metadata for the dataset.
        """
        if self.settings.meta_path.exists():
            with open(self.settings.meta_path, 'rb') as f:
                return pickle.load(f)
        return None

    def load(self) -> None:
        """
        Load the dataset from an external source.
        """
        if not self.settings.train_path.exists() or not self.settings.val_path.exists():
            self.tokenize()
        self.tokenized = {}
        self.tokenized['train'] = np.memmap(  # type: ignore
            self.settings.train_path,
            dtype=self.output_dtype,
            mode='r'
        )
        self.tokenized['val'] = np.memmap(  # type: ignore
            self.settings.val_path,
            dtype=self.output_dtype,
            mode='r'
        )

    def tokenize(self) -> None:
        """
        Tokenize the dataset.  This should call :py:meth:`export` to save the
        tokenized dataset to disk.
        """
        raise NotImplementedError

    def export(self) -> None:
        """
        Export the tokenized dataset to disk.
        """
        raise NotImplementedError


class HuggingFaceDatasetLoader(DatasetLoader):
    """
    Load and process a HuggingFace dataset.

    Keyword Args:
        num_proc: The number of workers to use when tokenizing the dataset
        num_proc_load_dataset: The number of workers to use when loading the
            dataset from HuggingFace
        val_size: The size of the 'val' split as a percentage; the rest is the
            'train' split
        seed: The random seed to use when splitting the dataset
        shuffle: Whether to shuffle the dataset before splitting
    """

    def __init__(
        self,
        settings: DatasetSettings,
        num_proc: int = 8,
        num_proc_load_dataset: Optional[int] = None,
        seed: int = 2357,
        shuffle: bool = True,
        **kwargs
    ):
        super().__init__(settings, **kwargs)
        #: The number of workers to use in :py:meth:`datasets.Dataset.map`
        self.num_proc = num_proc
        #: The number of workers to use in :py:func:`datasets.load_dataset`
        self.num_proc_load_dataset = num_proc_load_dataset
        if num_proc_load_dataset is None:
            self.num_proc_load_dataset = num_proc
        #: The random seed to use when splitting the dataset
        self.seed = seed
        #: Whether to shuffle the dataset before splitting
        self.shuffle = shuffle

    def tokenize(self) -> None:
        """
        Tokenize the dataset.
        """
        dataset = load_dataset(
            self.dataset_name,
            num_proc=self.num_proc_load_dataset
        )
        self.dataset = dataset['train'].train_test_split(
            test_size=self.val_size,
            seed=self.seed,
            shuffle=self.shuffle
        )
        # Rename the test split to val
        self.dataset['val'] = self.dataset.pop('test')

        def process(block) -> Dict[str, Any]:
            """
            This internal function is used to tokenize the dataset in parallel.

            Args:
                block: the block of text to tokenize

            Returns:
                A dictionary with the following keys:

                - ``ids``: the tokenized text
                - ``len``: the length of the tokenized text

            """
            ids = self.encoder.encode_ordinary(block['text'])
            ids.append(self.encoder.eot_token)
            out = {'ids': ids, 'len': len(ids)}
            return out

        self.tokenized = self.dataset.map(
            process,
            remove_columns=['text'],
            desc=f'{self.dataset_name}: tokenizing the splits with {self.encoder.name}, '
                 f'{self.vocab_size} tokens, {self.num_proc} workers',
            num_proc=self.num_proc,
        )
        self.export()

    def export(self) -> None:
        """
        Write our tokenized dataset to disk.

        This will save the tokenized dataset to disk as ``train.bin`` and
        ``val.bin``.
        """
        # Concatenate all the ids in each dataset into one large file we can use
        # for training
        for split, dset in self.tokenized.items():
            filename = self.settings.output_dir / f'{split}.bin'

            arr_len = np.sum(dset['len'], dtype=np.uint64)
            arr = np.memmap(filename, dtype=self.output_dtype, mode='w+', shape=(arr_len,))  # type: ignore
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'{self.dataset_name}: writing {filename}'):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches,
                    index=batch_idx,
                    contiguous=True
                ).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                # Write into mmap
                arr[idx:idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()


class OpenWebTextDatasetLoader(HuggingFaceDatasetLoader):

    dataset_name = 'openwebtext'


class ShakespeareDatasetLoader(DatasetLoader):
    """
    The Shakespeare dataset loader.   It's not a HuggingFace dataset, so we
    have to do some different things to load and tokenize it.
    """

    #: The name of the dataset
    dataset_name = 'shakespeare'
    #: The URL to download the dataset from
    DATASET_URL: str = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

    def tokenize(self) -> None:
        """
        Tokenize the dataset.
        """
        datafile = self.settings.output_dir / 'input.txt'
        if not datafile.exists():
            with open(datafile, 'w', encoding='utf-8') as f:
                f.write(requests.get(self.DATASET_URL, timeout=60).text)
        with open(datafile, 'r', encoding='utf-8') as f:
            self.dataset = f.read()

        n = len(self.dataset)
        train_size = int(n * (1 - self.val_size))
        train_data = self.dataset[:train_size]
        val_data = self.dataset[train_size:]
        self.tokenized = {
            'train': self.encoder.encode_ordinary(train_data),
            'val': self.encoder.encode_ordinary(val_data)
        }
        self.export()

    def export(self) -> None:
        """
        Export the tokenized dataset to disk.
        """
        train_ids = np.array(self.tokenized['train'], dtype=self.settings.output_dtype)  # type: ignore
        val_ids = np.array(self.tokenized['val'], dtype=self.settings.output_dtype)  # type: ignore
        train_ids.tofile(self.settings.train_path)
        val_ids.tofile(self.settings.val_path)


class ShakespeareCharDatasetLoader(ShakespeareDatasetLoader):

    #: The encoding function
    encoder: Any = CharTokenizer()
    #: The name of the dataset
    dataset_name = Literal['shakespeare_char']

    def tokenize(self) -> None:
        """
        Tokenize the dataset.
        """
        datafile = self.settings.output_dir / 'input.txt'
        if not datafile.exists():
            with open(datafile, 'w', encoding='utf-8') as f:
                f.write(requests.get(self.DATASET_URL, timeout=60).text)
        with open(datafile, 'r', encoding='utf-8') as f:
            self.dataset = f.read()
        self.encoder.build_vocabulary(self.dataset)
        n = len(self.dataset)
        train_size = int(n * (1 - self.val_size))
        train_data = self.dataset[:train_size]
        val_data = self.dataset[train_size:]
        self.tokenized = {
            'train': self.encoder.encode_ordinary(train_data),
            'val': self.encoder.encode_ordinary(val_data)
        }
        self.export()

    def export(self) -> None:
        """
        Save the tokenized dataset to disk.
        """
        super().export()
        meta = {
            'vocab_size': self.encoder.n_vocab,
            'itos': self.encoder.inverse_vocab,
            'stoi': self.encoder.vocab
        }
        with open(self.settings.meta_path, 'wb') as f:
            pickle.dump(meta, f)


class Dataset:
    """
    A class representing a dataset for training or validation.

    This wraps a :py:class:`DatasetLoader` and provides a few convenience methods for
    accessing and batching the data.
    """

    #: A mapping from dataset names to dataset loaders
    LOADERS: Dict[str, Type[DatasetLoader]] = {
        'openwebtext': OpenWebTextDatasetLoader,
        'shakespeare': ShakespeareDatasetLoader,
        'shakespeare_char': ShakespeareCharDatasetLoader,
    }

    @classmethod
    def get_tokenizer(cls, dataset_name: str) -> Any:
        """
        Return the tokenizer for the given dataset.

        Args:
            dataset_name: the name of the dataset

        Returns:
            The tokenizer class
        """
        return cls.LOADERS[dataset_name].encoder

    def __init__(
        self,
        settings: DatasetSettings,
    ):
        #: Our dataset settings
        self.settings = settings
        #: The dataset loader.  This will tokenize the dataset and save it to
        #: disk if it hasn't already been done.
        self.loader = self.LOADERS[settings.dataset](settings, **settings.dataset_loader_kwargs)
        self.loader.load()

    @property
    def data(self) -> Any:
        """
        Return the dataset itself.
        """
        return self.loader.data

    @property
    def vocab_size(self) -> int:
        """
        Return the size of the vocabulary.
        """
        return self.loader.vocab_size

    def get_batch(
        self,
        split: Literal['train', 'val'],
        settings: TrainingSettings
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of data from the dataset to use in training or validation.

        This pins the data to the GPU if
        :py:attr:`nanoGPT.settings.TrainingSettings.device_type` is ``cuda``.

        Args:
            split: The dataset split to get the batch from.  Must be either ``train``
                or ``val``.
            block_size: The size of the input sequence
            batch_size: The number of sequences in the batch

        Returns:
            A tuple of two tensors: the input sequence and the next word after
            each input sequence.
        """
        data = self.data[split]
        # Get a vector of random indices into the data, length
        # :py:attr:`nanoGPT.settings.TrainingSettings.batch_size`
        _idx = torch.randint(
            len(data) - settings.block_size,
            (settings.batch_size,)
        )
        # This is the input sequence for each batch element
        input = torch.stack(
            [
                torch.from_numpy((data[i:i + settings.block_size]).astype(np.int64))
                for i in _idx
            ]
        )
        # This is the next word after each sequence in ``input``
        targets = torch.stack(
            [
                torch.from_numpy((data[i + 1:i + 1 + settings.block_size]).astype(np.int64))
                for i in _idx
            ]

        )
        if settings.device_type == 'cuda':
            # Pin arrays x, y which allows us to move them to the GPU
            # asynchroously (non_blocking=True)
            input = input.pin_memory().to(settings.device, non_blocking=True)
            targets = targets.pin_memory().to(settings.device, non_blocking=True)
        else:
            input = input.to(settings.device)
            targets = targets.to(settings.device)
        return input, targets
