import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm.notebook import tqdm
import numpy as np
from transformers import LlamaForCausalLM
from result import Result
from helper import get_embedding_matrix, get_tokens, create_one_hot_and_embeddings
from math import * 
import math
import pandas as pd
import torch.nn.functional as F


# A safer version of the calc_loss function.
def calc_loss(model, embeddings_user, embeddings_adv, embeddings_target, targets):
    full_embeddings = torch.hstack([embeddings_user, embeddings_adv, embeddings_target]).to(dtype=model.dtype)
    
    # Check for NaN or extreme values in the embeddings
    if torch.isnan(full_embeddings).any():
        raise ValueError("NaN detected in input embeddings")
    if torch.max(full_embeddings).item() > 1e6 or torch.min(full_embeddings).item() < -1e6:
        print("Warning: Extreme values detected in embeddings")

    # Pass through model and get logits
    logits = model(inputs_embeds=full_embeddings).logits

    # Check for NaN or extreme values in logits
    if torch.isnan(logits).any():
        raise ValueError("NaN detected in logits")
    if torch.max(logits).item() > 1e6 or torch.min(logits).item() < -1e6:
        print("Warning: Extreme values detected in logits")
    
    # Apply optional clipping to logits
    logits = torch.clamp(logits, min=-1e4, max=1e4)

    # Compute the cross-entropy loss
    loss_slice_start = len(embeddings_user[0]) + len(embeddings_adv[0])
    loss = nn.CrossEntropyLoss()(logits[0, loss_slice_start - 1 : -1, :], targets)

    return loss, logits[:, loss_slice_start - 1:, :]


# Simplex Projection
def simplex_projection(s, epsilon=1e-12):
    if s.numel() == 0:
        raise ValueError("Input tensor s must not be empty")
    
    # Step 1: Sort s into mu in descending order
    mu, _ = torch.sort(s, descending=True)
    
    # Step 2: Compute rho
    cumulative_sum = torch.cumsum(mu, dim=0)
    arange = torch.arange(1, s.size(0) + 1, device=s.device)
    condition = mu - (cumulative_sum - 1) / (arange + epsilon) > 0

    nonzero_indices = torch.nonzero(condition, as_tuple=False)
    if nonzero_indices.size(0) == 0:
        rho = 1
    else:
        rho = nonzero_indices[-1].item() + 1

    # Step 3: Compute psi
    psi = (cumulative_sum[rho - 1] - 1) / rho
    
    # Step 4: Compute p
    p = torch.clamp(s - psi, min=0)
    
    return p


def project_rows_to_simplex(matrix):
    """
    Apply the simplex projection row-wise to a 2D tensor.

    Parameters:
    - matrix: 2D tensor (PyTorch tensor)

    Returns:
    - projected_matrix: Row-wise simplex projected 2D tensor (PyTorch tensor)
    """
    projected_matrix = torch.zeros_like(matrix)
    for i in range(matrix.size(0)):
        projected_matrix[i] = simplex_projection(matrix[i])
    return projected_matrix


# Entropy Projection
def entropy_projection(s, target_entropy=2, epsilon=1e-12):
    mask = (s > 0).float()
    non_zero_count = torch.sum(mask) + epsilon  # Prevent division by zero
    c = mask / non_zero_count

    # Step 2: Compute radius R
    gini_index = 1 - torch.square(s).sum()  # Ensure gini_index >= 0 ; PGD uses 's', NOT 'mask'
    gini_index = torch.clamp(gini_index, min=0, max=1)  # Keep it in valid range
    R = torch.sqrt(1.0 - (gini_index - 1.0) / non_zero_count) 
    # Compute Euclidean norm of (s - c)
    norm_s_c = torch.norm(s - c)

    # Check if R >= ||s - c||
    if R >= norm_s_c:
        return s
    else:
        scaled_s = R / (norm_s_c * (s - c) + epsilon) + c
        return simplex_projection(scaled_s)


def project_rows_to_entropy(matrix):
    """
    Apply the simplex projection row-wise to a 2D tensor.

    Parameters:
    - matrix: 2D tensor (PyTorch tensor)

    Returns:
    - projected_matrix: Row-wise simplex projected 2D tensor (PyTorch tensor)
    """
    projected_matrix = torch.zeros_like(matrix)
    for i in range(matrix.size(0)):
        projected_matrix[i] = entropy_projection(matrix[i])
    return projected_matrix


# BEGIN ATTACK HERE


def run(model, tokenizer, messages, target, pgdconfig):
    losses = []
    strings = []
    # Get all these values from the config object.
    device: str = pgdconfig.device
    num_steps: int = pgdconfig.num_steps
    # num_tokens: int = pgdconfig.num_tokens
    step_size: float = pgdconfig.step_size
    # seed: int = pgdconfig.seed

    embed_weights = get_embedding_matrix(model)
    user_prompt = messages
    print(user_prompt)
    user_prompt_tokens = get_tokens(user_prompt, tokenizer, device=device)
    target_tokens = get_tokens(target, tokenizer, device=device)[1:]
    one_hot_inputs, embeddings_user = create_one_hot_and_embeddings(user_prompt_tokens, embed_weights, device=device)
    one_hot_target, embeddings_target = create_one_hot_and_embeddings(target_tokens, embed_weights, device=device)
    # one_hot_adv, _ = create_one_hot_and_embeddings(adv_string_tokens, embed_weights)
    one_hot_adv = F.softmax(torch.randn(20, embed_weights.shape[0], dtype=torch.float16).to(device=device), dim=1)
    one_hot_adv.requires_grad_()
    
    best_disc_loss = np.inf
    best_adv_suffix = None
    
    # Initialize Adam optimizer with user-defined epsilon value
    optimizer = optim.Adam([one_hot_adv], lr=step_size, eps=1e-4)
    # Initialize lr scheduler (Cosine Annealing with Warm Restarts)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min = 0.0001)
    scheduler_cycle = 0
                    
    # PHASE_2: bool=False
    BREAK_IT: bool=False

    for epoch_no in tqdm(range(num_steps)):
        optimizer.zero_grad()

        embeddings_adv = (one_hot_adv @ embed_weights).unsqueeze(0)
        cross_entropy_loss, logits = calc_loss(model, embeddings_user, embeddings_adv, embeddings_target, one_hot_target)
        continuous_loss = cross_entropy_loss.detach().cpu().item()
        # Break the loop if loss becomes "NaN"
        if math.isnan(continuous_loss):
            print("The value is NaN. Skipping behavior.")
            BREAK_IT=True # Go to the Next Behavior

        # # Discretization part
        max_values = torch.max(one_hot_adv, dim=1)
        adv_token_ids = max_values.indices
        one_hot_discrete = torch.zeros(
            adv_token_ids.shape[0], embed_weights.shape[0], device=device, dtype=embed_weights.dtype
        )
        one_hot_discrete.scatter_(
            1,
            adv_token_ids.unsqueeze(1),
            torch.ones(one_hot_adv.shape[0], 1, device=device, dtype=embed_weights.dtype),
        )
        # Use one_hot_discrete to print Tokens
        # Use discrete tokens to calculate loss
        embeddings_adv_discrete = (one_hot_discrete @ embed_weights).unsqueeze(0)
        disc_loss, _ = calc_loss(model, embeddings_user, embeddings_adv_discrete, embeddings_target, one_hot_target)
        # If loss improves, save it as x_best
        discrete_loss =  disc_loss.detach().cpu().item()
        losses.append(discrete_loss)
        adv_suffix = tokenizer.decode(adv_token_ids.cpu().numpy())
        strings.append(adv_suffix)
        if discrete_loss < best_disc_loss:
            best_disc_loss = discrete_loss
            best_adv_suffix = adv_suffix
        cross_entropy_loss.backward()
        # Get the scheduler's state to get learning_rate
        scheduler_state = scheduler.state_dict()
        torch.nn.utils.clip_grad_norm_([one_hot_adv], max_norm=1.0)
        # update parameters
        optimizer.step()
        # update scheduler
        scheduler.step()
        model.zero_grad()

        # Apply Simplex & Entropy
        simplices = project_rows_to_simplex(one_hot_adv)
        regularized_simplices = project_rows_to_entropy(simplices)
        one_hot_adv.data = regularized_simplices.data 
        
        one_hot_adv.grad.zero_()
        # Compute Cycle
        if scheduler_state['T_cur'] == 0:
                scheduler_cycle += 1
        
        
        # Something went wrong; so break it and go to the next behavior.
        if BREAK_IT==True:
            break
    # Create a Result object with the provided values
    best_loss = best_disc_loss
    best_string = best_adv_suffix

    result = Result(best_loss, best_string, losses, strings)
    return result