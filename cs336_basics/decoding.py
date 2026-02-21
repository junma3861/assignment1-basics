import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Optional


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


# ---------------------------------------------------------------------------
# Perceive Decoding
# ---------------------------------------------------------------------------

@contextmanager
def _capture_hidden_states(module: nn.Module, attr_names: list[str]):
    """
    Context manager that registers a forward hook on the first found submodule
    whose name matches one of ``attr_names``. The hook captures the *output*
    tensor of that submodule and stores it in the returned list so the caller
    can read it after the forward pass.

    Args:
        module: Parent module to search for the target submodule.
        attr_names: Ordered list of attribute names to try. The first one that
                    exists on the module is used.

    Yields:
        A list of length 0 (before the forward pass) or 1 (after), holding
        the captured output tensor.
    """
    target = None
    for name in attr_names:
        if hasattr(module, name):
            target = getattr(module, name)
            break

    captured: list[torch.Tensor] = []

    if target is None:
        # Cannot hook; caller will fall back to full logit computation.
        yield captured
        return

    def hook(_, __, output):
        captured.append(output)

    handle = target.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def _aux_top_p_candidates(
    aux_logits: torch.Tensor,
    aux_top_p: float,
) -> torch.Tensor:
    """
    Select the smallest set of token indices whose cumulative probability mass
    under the auxiliary model is at least ``aux_top_p``.

    Args:
        aux_logits: Shape (batch_size, vocab_size). Logits from the aux model
                    for the *last* position only.
        aux_top_p: Cumulative-probability threshold in (0, 1].

    Returns:
        1-D LongTensor of *unique* candidate token IDs pooled across the batch.
        For batch_size == 1 this is simply the per-sequence candidate set.
    """
    probs = F.softmax(aux_logits, dim=-1)                 # (B, V)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)           # (B, V)

    # Shift cumsum by one so we always include the token that pushes the
    # cumulative mass *past* the threshold (standard nucleus convention).
    shifted = torch.cat(
        [torch.zeros(probs.shape[0], 1, device=probs.device), cumsum[:, :-1]], dim=-1
    )
    keep = shifted < aux_top_p                            # (B, V)

    # Collect candidate indices across the batch and deduplicate.
    candidate_ids: list[torch.Tensor] = []
    for b in range(probs.shape[0]):
        candidate_ids.append(sorted_indices[b][keep[b]])

    # Always include the top-1 token from every sequence in the batch.
    candidate_ids.append(sorted_indices[:, 0])            # (B,)

    unique_candidates = torch.unique(torch.cat(candidate_ids, dim=0))
    return unique_candidates                              # (C,)


def _partial_logits(
    main_model: nn.Module,
    input_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Run ``main_model`` and compute logits **only** for ``candidate_ids``.

    Efficient path (when the model has a standard pre-output norm layer):
      1. Hook the pre-output normalisation layer to capture the last-position
         hidden state ``h``.
      2. Project ``h`` onto *only* the rows of the output weight matrix that
         correspond to ``candidate_ids``.  This costs O(C · d_model) instead
         of O(V · d_model).

    Fallback:
      If none of the expected attribute names are found, execute a standard
      forward pass and index-select the full logits to the candidate set.

    Args:
        main_model: Large language model; forward returns (B, S, V).
        input_ids: Shape (B, S).
        candidate_ids: 1-D LongTensor of candidate token indices, shape (C,).

    Returns:
        Logits restricted to candidates, shape (B, C).
    """
    PRE_OUTPUT_NORM_NAMES = ["layer_norm_final", "norm_f", "ln_f", "final_norm"]
    OUTPUT_PROJ_NAMES = ["output_linear", "lm_head", "head", "output_projection"]

    with torch.no_grad():
        with _capture_hidden_states(main_model, PRE_OUTPUT_NORM_NAMES) as captured:
            full_logits = main_model(input_ids)     # (B, S, V)

        if captured:
            # captured[0]: (B, S, d_model) – output of the pre-output norm
            hidden = captured[0][:, -1, :]          # (B, d_model)

            proj: Optional[nn.Linear] = None
            for name in OUTPUT_PROJ_NAMES:
                if hasattr(main_model, name):
                    proj = getattr(main_model, name)
                    break

            if proj is not None and isinstance(proj, nn.Linear):
                W = proj.weight[candidate_ids, :]   # (C, d_model)
                candidate_logits = hidden @ W.T     # (B, C)
                if proj.bias is not None:
                    candidate_logits = candidate_logits + proj.bias[candidate_ids]
                return candidate_logits

    # Fallback: slice the full result.
    return full_logits[:, -1, :][:, candidate_ids]  # (B, C)


def perceive_decode(
    main_model: nn.Module,
    aux_model: nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    aux_top_p: float = 0.9,
    top_p: float = 1.0,
    eos_token_id: int = 50256,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Perceive Decoding: use a small auxiliary model as a cheap first-pass filter
    to identify likely next tokens, then score only those candidates with the
    large main model.

    **Algorithm (per step):**

    1. **Auxiliary pass** — run ``aux_model`` and collect the nucleus of tokens
       that together account for ``aux_top_p`` of the predicted probability
       mass.  Call this set *C* (typically |C| << vocab_size).
    2. **Main-model partial pass** — run ``main_model`` through all transformer
       layers to get accurate contextual hidden states.  Project the
       last-position hidden state onto *only* the |C| rows of the output
       embedding corresponding to *C*, skipping ``vocab_size − |C|`` dot
       products in the final linear layer.
    3. **Sample** — apply temperature scaling and optional top-p filtering to
       the main model's restricted logits, then draw the next token.
    4. Append the token; stop when ``eos_token_id`` is generated or
       ``max_new_tokens`` is reached.

    The key trade-off is ``aux_top_p``:  a smaller value produces a tighter
    candidate set (faster main-model projection) at the cost of potentially
    missing tokens that the large model would have ranked highly.

    Args:
        main_model: Large, accurate language model.
                    Forward signature: ``(input_ids) -> (B, S, vocab_size)``.
        aux_model: Small, fast language model sharing the *same* vocabulary.
                   Forward signature: ``(input_ids) -> (B, S, vocab_size)``.
        prompt_ids: Token IDs of the prompt, shape ``(B, T)`` or ``(T,)``.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Temperature applied to the **main** model's restricted
                     logits before sampling.
        aux_top_p: Nucleus threshold for the **auxiliary** model's candidate
                   selection (default 0.9).  Lower → fewer candidates → faster.
        top_p: Additional nucleus filtering on the **main** model's restricted
               logits after temperature scaling (default 1.0 = disabled).
        eos_token_id: Token ID that signals end-of-sequence.
        device: Target device; inferred from ``main_model`` if not given.

    Returns:
        Generated token IDs (prompt + new tokens), shape ``(B, T+N)`` or
        ``(T+N,)`` for a single-prompt input.
    """
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    if device is None:
        device = next(main_model.parameters()).device

    prompt_ids = prompt_ids.to(device)
    main_model.eval()
    aux_model.eval()

    output_ids = prompt_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # ------------------------------------------------------------------
            # Step 1: Auxiliary model → candidate token set C
            # ------------------------------------------------------------------
            aux_logits = aux_model(output_ids)           # (B, S, V)
            aux_last = aux_logits[:, -1, :]              # (B, V)
            candidate_ids = _aux_top_p_candidates(aux_last, aux_top_p)  # (C,)

            # ------------------------------------------------------------------
            # Step 2: Main model partial forward → logits over C only
            # ------------------------------------------------------------------
            candidate_logits = _partial_logits(main_model, output_ids, candidate_ids)  # (B, C)

            # ------------------------------------------------------------------
            # Step 3: Temperature + optional top-p → sample
            # ------------------------------------------------------------------
            candidate_logits = apply_temperature(candidate_logits, temperature)

            if top_p < 1.0:
                candidate_logits = top_p_sampling(candidate_logits, top_p)

            candidate_probs = F.softmax(candidate_logits, dim=-1)             # (B, C)
            sampled_pos = torch.multinomial(candidate_probs, num_samples=1)   # (B, 1)

            # Map local position back to the actual vocabulary index.
            next_tokens = candidate_ids[sampled_pos]                          # (B, 1)

            # ------------------------------------------------------------------
            # Step 4: Append and check stopping condition
            # ------------------------------------------------------------------
            output_ids = torch.cat([output_ids, next_tokens], dim=-1)

            if (next_tokens == eos_token_id).all():
                break

    if squeeze_output:
        output_ids = output_ids.squeeze(0)

    return output_ids
