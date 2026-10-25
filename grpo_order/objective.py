"""Single-use on-policy GRPO, response mean of generated-token means."""


def response_loss(log_probs, advantage, group_size, clip_epsilon):
    # No updates between sampling and this forward, no dropout, no group reuse.
    # Thus old policy == current policy: detaching is exact for the one allowed step.
    # Keep ratio form and clipping explicit; clipping is inactive at ratio == 1.
    ratio = (log_probs - log_probs.detach()).exp()
    surrogate = (ratio * advantage).minimum(ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage)
    return -surrogate.mean() / group_size


def token_statistics(logits, targets, temperature, chunk_tokens):
    import torch
    from torch.utils.checkpoint import checkpoint
    logps, entropies = [], []

    def selected_logp(x, target):
        logp = (x.float() / temperature).log_softmax(-1)
        return logp.gather(-1, target[:, None]).squeeze(-1)

    for start in range(0, len(targets), chunk_tokens):
        x, y = logits[start:start + chunk_tokens], targets[start:start + chunk_tokens]
        logps.append(checkpoint(selected_logp, x, y, use_reentrant=False))
        with torch.no_grad():
            lp = (x.float() / temperature).log_softmax(-1)
            entropies.append(-(lp.exp() * lp).sum(-1))
    return torch.cat(logps), torch.cat(entropies)


def accumulate_group(model, prompt, responses, adv, config, pad_id):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    device = next(model.parameters()).device
    group_size = len(responses)
    micro = config["response_microbatch"]
    total_loss, entropy_sum, token_count = 0.0, 0.0, 0
    model.train()
    for start in range(0, group_size, micro):
        chunk = responses[start:start + micro]
        seqs = [torch.tensor(prompt + r["token_ids"], device=device) for r in chunk]
        x = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
        mask = torch.arange(x.shape[1], device=device)[None, :] < torch.tensor([len(s) for s in seqs], device=device)[:, None]
        logits = model(input_ids=x, attention_mask=mask.long(), use_cache=False).logits
        loss = torch.zeros((), device=device)
        for j, response in enumerate(chunk):
            n = len(response["token_ids"])
            # logits at prompt_end - 1 predict first generated token; prompt/pad excluded.
            lp, entropy = token_statistics(logits[j, len(prompt) - 1:len(prompt) - 1 + n],
                                           x[j, len(prompt):len(prompt) + n],
                                           config["temperature"], config["logit_chunk_tokens"])
            loss = loss + response_loss(lp, adv[start + j], group_size, config["clip_epsilon"])
            entropy_sum += entropy.sum().item()
            token_count += n
        loss.backward()  # Sum already scaled by full group size, never divide again by microbatches.
        total_loss += loss.detach().item()
        del logits, loss
    return {"loss": total_loss, "token_entropy_nats": entropy_sum / token_count,
            "entropy_definition": "token-weighted full-vocabulary Shannon entropy of logits/temperature at generated positions, including EOS"}
