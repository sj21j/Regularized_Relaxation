import result
from helper import get_embedding_matrix, get_tokens, create_one_hot_and_embeddings, find_closest_embeddings, calc_ce_loss, get_nonascii_toks
import torch
import config
import torch.optim as optim
from transformers import LlamaForCausalLM

def adjust_learning_rate(initial_lr, iteration, decay_rate):
    return initial_lr * (decay_rate ** iteration)

def run(model, tokenizer, messages, target, rrconfig):
    # messages = [{"role": "system", "content": "BREAK EVERYTHING"}] + messages
    # print(f"Running RR with {model.__class__.__name__}")
    # print(tokenizer.chat_template)
    # test = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    # print(test)
    # print(tokenizer.decode(test[0]))
    if not rrconfig.allow_non_ascii: 
        non_ascii_toks = get_nonascii_toks(tokenizer, config.device)
    else:
        non_ascii_toks = None
    messages = "USER:"+messages
    torch.manual_seed(rrconfig.seed)
    embed_weights = get_embedding_matrix(model)
    user_prompt_ids = get_tokens(messages, tokenizer, config.device)
    adv_string_init_ids = get_tokens(rrconfig.optim_str_init, tokenizer, config.device)[1:]
    target_ids = get_tokens(target, tokenizer, config.device)[1:]
    _, embeddings_user = create_one_hot_and_embeddings(user_prompt_ids, embed_weights, config.device)
    _, embeddings_adv = create_one_hot_and_embeddings(adv_string_init_ids, embed_weights, config.device)
    one_hot_target, embeddings_target = create_one_hot_and_embeddings(target_ids, embed_weights, config.device)
    
    embeddings_adv = embeddings_adv + torch.normal(0, 0.1, embeddings_adv.size()).to(config.device)

    embeddings_adv.requires_grad = True
    optimizer = optim.AdamW([embeddings_adv], lr=rrconfig.initial_lr, weight_decay=rrconfig.weight_decay)
    best_discrete_loss = float("inf")  # Initialize best_discrete_loss
    best_string = None

    for iteration in range(rrconfig.num_steps):
        optimizer.zero_grad()
        effective_adv = embeddings_adv
        
        # Calculate loss on the modified embeddings
        loss = calc_ce_loss(model, embeddings_user, effective_adv, embeddings_target, target_ids)
        loss.backward()

        # Find closest embeddings after the update
        _, closest_indices, closest_embeddings = find_closest_embeddings(effective_adv.to(dtype=torch.float16), embed_weights, model.device, rrconfig.allow_non_ascii, non_ascii_toks)
        # Calculate discrete loss using the closest embeddings
        discrete_loss = calc_ce_loss(model, embeddings_user, closest_embeddings, embeddings_target, target_ids)
        # Track the lowest discrete loss and store the closest embeddings for that loss
        # if loss.detach().cpu().numpy() < 0.001:
        #     break
        if iteration == 0 or discrete_loss < best_discrete_loss:
            best_discrete_loss = discrete_loss
            best_closest_embeddings = closest_embeddings.clone()
            print(f"Best discrete loss: {best_discrete_loss.detach().cpu().numpy()}")
            print(f"Best discrete loss iteration: {iteration}")
            # print(f"Best suffix: {tokenizer.decode(closest_indices[0])}")
            best_suffix = tokenizer.decode(closest_indices[0])
        print(f"{iteration}Cont Loss: {loss.detach().cpu().numpy()} Disc Loss: {discrete_loss.detach().cpu().numpy()} Best suffix: {best_suffix}")

        # Apply gradient clipping to embeddings_adv
        torch.nn.utils.clip_grad_norm_([embeddings_adv], max_norm=1.0)
        
        # Perform optimization step
        optimizer.step()
        
        # Adjust learning rate
        optimizer.param_groups[0]['lr'] = adjust_learning_rate(rrconfig.initial_lr, iteration, rrconfig.decay_rate)

        # Zero gradients for embeddings_adv after each iteration
        embeddings_adv.grad.zero_()

    final_string = messages + best_suffix + "ASSISTANT:"
    if isinstance(model, LlamaForCausalLM):
        final_string_ids = get_tokens(final_string, tokenizer, config.device)
        generated_output = model.generate(
            final_string_ids.unsqueeze(0),
            max_length=200,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None
        )
        generated_output_string = tokenizer.decode(generated_output[0][1:].cpu().numpy()).strip()[len(final_string):]

    print(f"Generated output: \n{generated_output_string}")
    return None