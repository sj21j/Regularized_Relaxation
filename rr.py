from result import RRResult
from helper import get_embedding_matrix, get_tokens, create_one_hot_and_embeddings, find_closest_embeddings, calc_ce_loss, get_nonascii_toks
import torch
import config
import torch.optim as optim
from transformers import LlamaForCausalLM, FalconForCausalLM, MistralForCausalLM, MptForCausalLM

def adjust_learning_rate(initial_lr, iteration, decay_rate):
    return initial_lr * (decay_rate ** iteration)

def run(model, tokenizer, messages, target, rrconfig):
    if not rrconfig.allow_non_ascii: 
        non_ascii_toks = get_nonascii_toks(tokenizer, config.device)
    else:
        non_ascii_toks = None
    embed_weights = get_embedding_matrix(model)
    user_prompt_ids = get_tokens(messages, tokenizer, config.device)
    adv_string_init_ids = get_tokens(rrconfig.optim_str_init, tokenizer, config.device)
    target_ids = get_tokens(target, tokenizer, config.device)
    if not isinstance(model, FalconForCausalLM):
        adv_string_init_ids = adv_string_init_ids[1:]
        target_ids = target_ids[1:]
    _, embeddings_user = create_one_hot_and_embeddings(user_prompt_ids, embed_weights, config.device)
    _, embeddings_adv = create_one_hot_and_embeddings(adv_string_init_ids, embed_weights, config.device)
    one_hot_target, embeddings_target = create_one_hot_and_embeddings(target_ids, embed_weights, config.device)
    
    embeddings_adv = embeddings_adv + torch.normal(0, 0.1, embeddings_adv.size()).to(config.device)

    embeddings_adv.requires_grad = True
    optimizer = optim.AdamW([embeddings_adv], lr=rrconfig.initial_lr, weight_decay=rrconfig.weight_decay)
    best_discrete_losses = [float("inf")] * len(rrconfig.checkpoints)  # Initialize with high loss values
    best_strings = [""] * len(rrconfig.checkpoints)  # Initialize with empty strings
    losses = []
    suffixes = []
    distances = [float("inf")] * len(rrconfig.checkpoints)

    for iteration in range(rrconfig.num_steps):
        optimizer.zero_grad()
        effective_adv = embeddings_adv
        
        # Calculate loss on the modified embeddings
        loss = calc_ce_loss(model, embeddings_user, effective_adv, embeddings_target, target_ids)
        loss_value = loss.detach().cpu().item()
        loss.backward()

        # Find closest embeddings after the update
        closest_distances, closest_indices, closest_embeddings = find_closest_embeddings(effective_adv.to(dtype=model.dtype), embed_weights, model.device, rrconfig.allow_non_ascii, non_ascii_toks)
        current_suffix = tokenizer.decode(closest_indices[0])
        current_mean_distance = closest_distances.mean().detach().cpu().item()
        # Calculate discrete loss using the closest embeddings
        discrete_loss = calc_ce_loss(model, embeddings_user, closest_embeddings, embeddings_target, target_ids)
        discrete_loss_value = discrete_loss.detach().cpu().item()
        losses.append(discrete_loss_value)
        suffixes.append(current_suffix)

        # Save checkpoints when loss crosses certain thresholds
        for i, checkpoint in enumerate(rrconfig.checkpoints):
            if loss_value < checkpoint and best_discrete_losses[i] == float("inf"):
                best_discrete_losses[i] = discrete_loss_value
                best_strings[i] = current_suffix
                distances[i] = current_mean_distance  # Update distance as the mean of closest distances
                if best_discrete_losses[-1] != float("inf"):
                    return RRResult(best_discrete_losses, best_strings, losses, suffixes, distances)

        # Apply gradient clipping to embeddings_adv
        torch.nn.utils.clip_grad_norm_([embeddings_adv], max_norm=1.0)
        
        # Perform optimization step
        optimizer.step()
        
        # Adjust learning rate
        optimizer.param_groups[0]['lr'] = adjust_learning_rate(rrconfig.initial_lr, iteration, rrconfig.decay_rate)

        # Zero gradients for embeddings_adv after each iteration
        embeddings_adv.grad.zero_()
    # If at the end of the iterations, the best suffix is still empty, use the last closest embedding
    # Ensure that all checkpoints have values
    last_discrete_loss = losses[-1]
    last_suffix = suffixes[-1]
    last_distance = distances[-1]
    
    for i in range(len(rrconfig.checkpoints)):
        if best_discrete_losses[i] == float("inf"):  # If a checkpoint was not reached, use last values
            best_discrete_losses[i] = last_discrete_loss
            best_strings[i] = last_suffix
            distances[i] = last_distance  # Use the last distance for the missing checkpoint

    # Create the result object with the updated values
    return RRResult(best_discrete_losses, best_strings, losses, suffixes, distances)