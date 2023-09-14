from torch.utils.data import Dataset
import torch

class DataSet(Dataset):
    """
    A custom dataset for reading and loading data into a dataloader for use in finetuning a model.
    """
    def __init__(self, dataframe, tokenizer, source_len, target_len, source_text, target_text):
        """
        Initialize the dataset with the given dataframe, tokenizer, and text column names.
        :param dataframe: a pandas DataFrame containing the data
        :param tokenizer: a tokenizer object to use for encoding text
        :param source_len: the maximum length of source text sequences
        :param target_len: the maximum length of target text sequences
        :param source_text: the name of the column in the dataframe containing source text
        :param target_text: the name of the column in the dataframe containing target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # Clean data to ensure it is in string type
        source_text = ' '.join(source_text.split())
        target_text = ' '.join(target_text.split())

        # Encode text and create tensors
        source = self.tokenizer.batch_encode_plus([source_text], max_length=self.source_len,
                                                   pad_to_max_length=True, truncation=True, padding="max_length",
                                                   return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([target_text], max_length=self.target_len,
                                                   pad_to_max_length=True, truncation=True, padding="max_length",
                                                   return_tensors='pt')

        # Extract input and attention masks from encoded text
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        # Return the encoded text and masks as tensors
        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_mask': target_mask.to(dtype=torch.long)
        }
