from pathlib import Path
from typing import Optional, Union
from masking_dataset_class import MaskDataset
from torch.utils.data import DataLoader

__docformat__ = 'reStructuredText'
__all__ = ['get_dataset', 'get_data_loader']


def get_dataset(data_split: Union[str, Path],
                data_dir: Optional[str] = '',
                load_into_memory: Optional[bool] = True) \
        -> MaskDataset:
    """Creates and returns a dataset, according to `MyDataset` class.

    :param data_dir: Directory to read data from.
    :type data_dir: str|pathlib.Path
    :param data_parent_dir: Parent directory of the data, defaults\
                            to ``.
    :type data_parent_dir: str
    :param key_features: Key to use for getting the features,\
                         defaults to `features`.
    :type key_features: str
    :param key_class: Key to use for getting the class, defaults\
                      to `class`.
    :type key_class: str
    :param load_into_memory: Load the data into memory? Default to True
    :type load_into_memory: bool
    :return: Dataset.
    :rtype: dataset_class.MyDataset
    """
    return MaskDataset(data_split=data_split,
                     data_dir=data_dir,
                     load_into_memory=load_into_memory)


def get_data_loader(dataset: MaskDataset,
                    batch_size: int,
                    shuffle: bool,
                    drop_last: bool) \
        -> DataLoader:
    """Creates and returns a data loader.

    :param dataset: Dataset to use.
    :type dataset: dataset_class.MyDataset
    :param batch_size: Batch size to use.
    :type batch_size: int
    :param shuffle: Shuffle the data?
    :type shuffle: bool
    :return: Data loader, using the specified dataset.
    :rtype: torch.utils.data.DataLoader
    """
    return DataLoader(dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle, 
                    drop_last=drop_last,
                    num_workers=1)