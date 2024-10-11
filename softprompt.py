from result import Result
from helper import get_embedding_matrix, get_tokens, create_one_hot_and_embeddings, find_closest_embeddings, calc_ce_loss, get_nonascii_toks
import torch
import torch.nn as nn
import config
import torch.optim as optim
from transformers import LlamaForCausalLM, FalconForCausalLM, MistralForCausalLM, MptForCausalLM

def calc_loss(model, embeddings, embeddings_attack, embeddings_target, targets):
        full_embeddings = torch.hstack([embeddings, embeddings_attack, embeddings_target])
        logits = model(inputs_embeds=full_embeddings).logits
        loss_slice_start = len(embeddings[0]) + len(embeddings_attack[0])
        loss = nn.CrossEntropyLoss()(logits[0, loss_slice_start - 1 : -1, :], targets)
        return loss, logits[:, loss_slice_start - 4 : -1, :]

def run(model, tokenizer, messages, target, rrconfig):
    
    if not config.rrconfig.allow_non_ascii: 
        non_ascii_toks = get_nonascii_toks(tokenizer, config.device)
    else:
        non_ascii_toks = None
    embed_weights = get_embedding_matrix(model)
    user_prompt_ids = get_tokens(messages, tokenizer, config.device)
    adv_string_init_ids = get_tokens(config.adv_string_init, tokenizer, config.device)
    target_ids = get_tokens(target, tokenizer, config.device)
    if not isinstance(model, FalconForCausalLM):
        adv_string_init_ids = adv_string_init_ids[1:]
        target_ids = target_ids[1:]
    _, embeddings_user = create_one_hot_and_embeddings(user_prompt_ids, embed_weights, config.device)
    _, embeddings_adv = create_one_hot_and_embeddings(adv_string_init_ids, embed_weights, config.device)
    one_hot_target, embeddings_target = create_one_hot_and_embeddings(target_ids, embed_weights, config.device)
    
    adv_pert = torch.zeros_like(embeddings_adv, requires_grad=True, device=config.device)
    step_size = 0.1
    best_discrete_loss = float("inf")  # Initialize best_discrete_loss
    best_suffix = ""
    losses = []
    suffixes = []

    for i in range(config.num_steps):
        loss, logits = calc_loss(model, embeddings_user, embeddings_adv + adv_pert, embeddings_target, one_hot_target)
        loss.backward()
        grad = adv_pert.grad.data
        adv_pert.data -= torch.sign(grad) * step_size

        model.zero_grad()
        adv_pert.grad.zero_()

        tokens_pred = logits.argmax(2)
        output_str = tokenizer.decode(tokens_pred[0][3:].cpu().numpy())

        effective_adv = embeddings_adv + adv_pert
        _, closest_indices, closest_embeddings = find_closest_embeddings(effective_adv.to(dtype=model.dtype), embed_weights, model.device, config.rrconfig.allow_non_ascii, non_ascii_toks)
        current_suffix = tokenizer.decode(closest_indices[0])
        discrete_loss = calc_ce_loss(model, embeddings_user, closest_embeddings, embeddings_target, target_ids)
        success = output_str == target
        if success:
            best_discrete_loss = discrete_loss.detach().cpu().item()
            best_suffix = current_suffix
            result = Result(best_discrete_loss, best_suffix, losses, suffixes)
            return result
        

    if best_suffix == "":
        best_suffix = tokenizer.decode(closest_indices[0])
        best_discrete_loss = discrete_loss.detach().cpu().item()
    result = Result(best_discrete_loss, best_suffix, losses, suffixes)
    return result