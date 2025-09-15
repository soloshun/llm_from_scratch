from torch import tensor
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        # the max_length here is the same as the context window / context size
        self.input_ids = []
        self.target_ids = []
        
        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        # We divide the text into overlapping chunks of `max_length` tokens using a sliding window.
        # For each chunk:
        #   - input_chunk is tokens[i : i + max_length]
        #   - target_chunk is tokens[i+1 : i + max_length + 1], which is input shifted right by one
        # The subtraction by max_length ensures the final chunk ends cleanly without going out of bounds.
        # The `stride` controls how much overlap there is between consecutive chunks.
        # ...stops before the final window that would exceed len(token_ids) when grabbing target_chunk.
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(tensor(input_chunk))
            self.target_ids.append(tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_ids)
       
    # this function tells the dataloader what kind of input and output should we have... this is for the pytorch dataloader to do its own stuff     
    def __getitem__(self,idx):
        return self.input_ids[idx], self.target_ids[idx]
    
    
def create_dataloader_v1(
    txt, 
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0
):
    
    # initialize the tokenizer
    tokenizer = tiktoken.get_encoding(encoding_name='gpt2')
    
    # create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
    # create dataloader
    # this dataloader is going to access the __getitem__ in the dataset and create input-output tensors
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    
    return dataloader