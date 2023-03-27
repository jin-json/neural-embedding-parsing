from transformers import RobertaTokenizer, RobertaForMaskedLM
from torch.nn import functional as F
import torch
tokenizer = tok #RobertaTokenizer.from_pretrained('roberta-base')
#model = RobertaForMaskedLM.from_pretrained('roberta-base', return_dict = True)

total = 0
counter = 0
rank_total = 0
# for entry in dataset['train']:
#     if counter > 1000: 
#         break
for i in tqdm(range(900)):
    entry = dataset['train'][i]
    example = entry['example'].lower()
    word = entry['lowercase_word']
    text = example.replace(word, tokenizer.mask_token, 1)
    ## should not be a problem since preprocessing!
    if text == example: 
        print('no change in sentence error')
        continue
     
    element_index = None
    tokens = tokenizer(word)
    ids = tokens['input_ids']
    if (len(ids) != 3):
        print('token length error')
    else:
        element_index = ids[1]
    
    input_tokens = tokenizer.encode(text, return_tensors="pt")
    mask_index = torch.where(input_tokens[0] == tokenizer.mask_token_id)
    #mask_index = torch.where(input_tokens["input_ids"][0] == tokenizer.mask_token_id)[1].item()
    sentences = [input_tokens[0].clone() for _ in range(tokenizer.vocab_size)]
    for i, sentence in enumerate(sentences):
        sentence[mask_index] = i
    sentences_tensor = torch.stack(sentences)
    
    with torch.no_grad():
        outputs = model(sentences_tensor, labels=sentences_tensor)
    log_likelihoods = outputs[0]
    sentence_probabilities = torch.exp(log_likelihoods.sum(dim=-1))
    sorted_indices = torch.argsort(sentence_probabilities, descending=True)
    ranked_tokens = [tokenizer.decode(torch.tensor([token_id])) for token_id in sorted_indices]
    
    token = tokenizer.decode(torch.tensor([element_index]))
    rank = ranked_tokens.index(token) + 1
    rank_total += element_rank
    if element_rank < 1000:
        total += 1
    counter += 1
        
print(total)
print(counter)
print(rank_total)