import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    
    Temperature controls the randomness of predictions:
    - temperature = 1.0: no change
    - temperature < 1.0: sharper distribution (more confident)
    - temperature > 1.0: softer distribution (more random)
    
    Args:
        logits: Shape (batch_size, vocab_size) or (vocab_size,)
        temperature: Temperature value (must be > 0)
    
    Returns:
        Temperature-scaled logits with the same shape as input
    """
    if temperature == 1.0:
        return logits
    return logits / temperature


def top_p_sampling(logits: torch.Tensor, top_p: float = 1.0) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling to logits.
    
    Top-p sampling selects from the smallest set of tokens with cumulative probability
    >= top_p, then samples from this filtered distribution.
    
    Args:
        logits: Shape (batch_size, vocab_size) or (vocab_size,)
        top_p: Cumulative probability threshold (0.0 to 1.0)
               - 1.0: no filtering (sample from all tokens)
               - 0.9: sample from top 90% of probability mass
    
    Returns:
        Logits with non-top-p tokens set to -inf, same shape as input
    """
    if top_p >= 1.0:
        return logits
    
    # Ensure we're working with probabilities (not logits)
    probs = F.softmax(logits, dim=-1)
    
    # Handle both batched and unbatched inputs
    if logits.dim() == 1:
        probs = probs.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, vocab_size = probs.shape
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # Compute cumulative probabilities
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find the cutoff index where cumulative probability exceeds top_p
    # We want to keep probabilities where cumsum <= top_p, plus one more
    cutoff_mask = cumsum_probs <= top_p
    
    # Always keep at least the top token
    cutoff_mask[:, 0] = True
    
    # Find the last True value in each row (the cutoff index)
    # We need to handle the edge case where we might want to include more
    cumsum_with_top_p = torch.cat(
        [torch.ones(batch_size, 1, device=probs.device),
         cumsum_probs[:, :-1]],
        dim=-1
    )
    cutoff_mask = cumsum_with_top_p <= top_p
    
    # For each batch, find the last index where cumsum <= top_p
    cutoff_indices = cutoff_mask.long().argmax(dim=-1)
    # If argmax returns 0 and the first element is False, it means no elements qualify
    # In that case, use at least the first element
    for i in range(batch_size):
        if not cutoff_mask[i, 0]:
            cutoff_indices[i] = 0
    
    # Create a mask for tokens to keep
    keep_mask = torch.zeros_like(sorted_probs, dtype=torch.bool)
    for i in range(batch_size):
        keep_mask[i, :cutoff_indices[i] + 1] = True
    
    # Set probabilities of filtered tokens to 0
    filtered_probs = sorted_probs.clone()
    filtered_probs[~keep_mask] = 0.0
    
    # Renormalize
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    
    # Convert back to logits (with a small epsilon to avoid log(0))
    epsilon = 1e-10
    filtered_logits = torch.log(filtered_probs + epsilon)
    
    # Unsort back to original order
    unsort_indices = sorted_indices.argsort(dim=-1)
    result = torch.gather(filtered_logits, -1, unsort_indices)
    
    if squeeze_output:
        result = result.squeeze(0)
    
    return result


def generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int = 50256,  # Default GPT-2 end-of-text token
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate a completion for a given prompt using the language model.
    
    This function generates tokens autoregressively until either:
    - The end-of-sequence token is sampled
    - max_new_tokens is reached
    
    Args:
        model: A language model with a forward method that takes input_ids 
               and returns logits of shape (batch_size, seq_length, vocab_size)
        prompt_ids: Input prompt token IDs, shape (batch_size, prompt_length)
                   or (prompt_length,) for a single prompt
        max_new_tokens: Maximum number of new tokens to generate (default: 100)
        temperature: Temperature for controlling randomness (default: 1.0)
                    - Values > 1.0 increase randomness
                    - Values < 1.0 decrease randomness
        top_p: Nucleus sampling threshold (default: 1.0, no filtering)
               - Only tokens in the top p cumulative probability are sampled
        eos_token_id: The token ID marking end-of-sequence (default: 50256)
        device: Device to run computation on (if None, uses model's device)
    
    Returns:
        Generated token IDs including the original prompt and new tokens,
        shape (batch_size, prompt_length + num_new_tokens) or 
        (prompt_length + num_new_tokens,) for single prompt
    """
    # Handle single prompt case
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Move to device
    if device is None:
        device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize output with the prompt
    output_ids = prompt_ids.clone()
    batch_size = prompt_ids.shape[0]
    
    # Generation loop
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get model predictions
            logits = model(output_ids)  # (batch_size, seq_length, vocab_size)
            
            # Take the last token's logits
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # Apply temperature scaling
            next_token_logits = apply_temperature(next_token_logits, temperature)
            
            # Apply top-p sampling
            if top_p < 1.0:
                next_token_logits = top_p_sampling(next_token_logits, top_p)
            
            # Convert logits to probabilities
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_tokens = torch.multinomial(next_token_probs, num_samples=1)
            
            # Append to output
            output_ids = torch.cat([output_ids, next_tokens], dim=-1)
            
            # Check if we've hit the end-of-sequence token for all sequences
            if (next_tokens == eos_token_id).all():
                break
    
    if squeeze_output:
        output_ids = output_ids.squeeze(0)
    
    return output_ids


def greedy_decode(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 100,
    eos_token_id: int = 50256,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate a completion using greedy decoding (always pick the most likely token).
    
    This is a deterministic version of generate() that always selects the token
    with the highest probability, regardless of the full distribution.
    
    Args:
        model: A language model with a forward method that takes input_ids
        prompt_ids: Input prompt token IDs
        max_new_tokens: Maximum number of new tokens to generate
        eos_token_id: The token ID marking end-of-sequence
        device: Device to run computation on
    
    Returns:
        Generated token IDs including the original prompt
    """
    # Handle single prompt case
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Move to device
    if device is None:
        device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize output with the prompt
    output_ids = prompt_ids.clone()
    
    # Generation loop
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get model predictions
            logits = model(output_ids)  # (batch_size, seq_length, vocab_size)
            
            # Take the last token's logits and argmax
            next_token_ids = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            
            # Append to output
            output_ids = torch.cat([output_ids, next_token_ids], dim=-1)
            
            # Check if we've hit the end-of-sequence token for all sequences
            if (next_token_ids == eos_token_id).all():
                break
    
    if squeeze_output:
        output_ids = output_ids.squeeze(0)
    
    return output_ids
