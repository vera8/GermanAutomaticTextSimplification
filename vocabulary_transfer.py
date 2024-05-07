import torch
from torch import nn


def initialize_embeddings(model, tokenizer_old, tokenizer_new):
    """Implements the initialization of the new embeddings based on the Vocabulary Transfer method proposed by
    Mosin et al. (https://doi.org/10.1016/j.artint.2023.103860)"""
    new_vocab = tokenizer_new.get_vocab()
    old_vocab = tokenizer_old.get_vocab()
    input_embed_weights_old = model.get_input_embeddings().weight.clone().detach()
    output_embed_weights_old = model.get_output_embeddings().weight.clone().detach()

    # resize embedding matrix for the size of the new vocabulary
    model.resize_token_embeddings(tokenizer_new.vocab_size)

    def init_embedding(emb, emb_weights_old):
        for item in new_vocab.items():
            token = item[0]
            index_new = item[1]
            if token in old_vocab:
                # keep the embeddings of tokens that are the same in new and old vocabulary
                index_old = old_vocab[token]
                embedding = emb_weights_old[index_old]
            else:
                # for new tokens, partition with the old tokenizer
                subtokens = tokenizer_old.tokenize(token)
                subtoken_ids = tokenizer_old.convert_tokens_to_ids(subtokens)
                # get the embeddings from the old embedding matrix of each token in the partition
                subtoken_embeddings = torch.stack([emb_weights_old[idx] for idx in subtoken_ids])
                # take the mean of all embeddings
                embedding = torch.mean(subtoken_embeddings, 0)
            # initialize new token in new embedding
            with torch.no_grad():
                emb.weight[index_new] = embedding

    input_embed_new = nn.Embedding(tokenizer_new.vocab_size, model.model_dim)
    init_embedding(input_embed_new, input_embed_weights_old)
    model.set_input_embeddings(new_embeddings=input_embed_new)

    output_embed_new = nn.Linear(model.model_dim, tokenizer_new.vocab_size, bias=False)
    init_embedding(output_embed_new, output_embed_weights_old)
    model.set_output_embeddings(new_embeddings=output_embed_new)
    return model
