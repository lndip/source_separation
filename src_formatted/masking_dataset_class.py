from typing import Tuple, Optional, Union, Dict
from pickle import load as pickle_load
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from utils import get_files_from_dir_with_pathlib

__docformat__ = 'reStructuredText'
__all__ = ['MaskDataset']


class MaskDataset(Dataset):
    def __init__(self,
                data_split: Union[Path, str],
                data_dir: Optional[Union[Path,str]]="",
                load_into_memory: Optional[int]=True) \
            -> None:

        """An example of an object of class torch.utils.data.Dataset

        :param data_split: The data split.
        :type data_split: str
        :param data_dir: Directory to read data from.
        :type data_dir: str
        :param key_features: Key to use for getting the features,\
                             defaults to `features`.
        :type key_features: str
        :param key_class: Key to use for getting the class, defaults\
                          to `class`.
        :type key_class: str
        :param load_into_memory: Load the data into memory? Default to True
        :type load_into_memory: bool
        """
        super().__init__()
        data_path = Path(data_dir, data_split)
        self.files = get_files_from_dir_with_pathlib(data_path)
        self.load_into_memory = load_into_memory
        self.key_feature = "source"
        self.key_class = "target"
        self.speech = "speech"
        if self.load_into_memory:
            for i, file in enumerate(self.files):
                self.files[i] = self._load_file(file)

    @staticmethod
    def _load_file(file_path: Path) \
            -> Dict[str, Union[int, np.ndarray]]:
        """Load the dictonary containing the data from a pickle file

        :param file_path: File path from which to load the dict
        :type file_path: Path
        :return: The dict containing the data
        :rtype: dict(str, int|np.ndarray)
        """
        with file_path.open('rb') as file:
            return pickle_load(file)


    def __len__(self) -> int:
        """Returns the length of the dataset.

        :return: Length of the dataset.
        :rtype: int
        """
        return len(self.files)
        

    def __getitem__(self, index: int) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Returns an item from the dataset.

        :param index: Index of the item.
        :type index: int
        :return: Features and class of the item.
        :rtype: (np.ndarray, np.ndarray)
        """
        if self.load_into_memory:
            the_item: Dict[str, Union[int, np.ndarray]] = self.files[index]
        else:
            the_item = self._load_file(self.files[index])
        return the_item[self.key_feature], the_item[self.key_class], the_item[self.speech]
        

       
       