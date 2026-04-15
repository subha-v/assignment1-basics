import torch

def decode(prompt: str, model, tokenizer, eot_id: int, max_new_tokens: int = 256, temperature: float = 1.0, top_p: float = 1.0, device: str = "cpu") -> str:
    model.eval()

    promptIds = tokenizer.encode(prompt)
    generatedIds = list(promptIds)
    contextLength = model.context_length

    with torch.no_grad():
        for _ in range(max_new_tokens):
            windowedIds = generatedIds[-contextLength:]
            x = torch.tensor(windowedIds, dtype=torch.long, device=device)
            x = x.unsqueeze(0)

            # ffwd pas
            allLogits = model(x)
            lastLogits = allLogits[0, -1, :]

            if temperature == 0.0:
                nextId = int(torch.argmax(lastLogits).item())
            else:
                scaledLogits = lastLogits / temperature
                shifted = scaledLogits - torch.max(scaledLogits)
                expShifted = torch.exp(shifted)
                probs = expShifted / torch.sum(expShifted)

                if top_p < 1.0:
                    sortedProbs, sortedIndices = torch.sort(probs, descending=True)
                    cumProbs = torch.cumsum(sortedProbs, dim=-1)

                    cumBefore = cumProbs - sortedProbs
                    keepMask = cumBefore < top_p

                    truncated = torch.zeros_like(probs)
                    keptIndices = sortedIndices[keepMask]
                    truncated[keptIndices] = probs[keptIndices]
                    probs = truncated / torch.sum(truncated)

                sampled = torch.multinomial(probs, num_samples=1)
                nextId = int(sampled.item())

            generatedIds.append(nextId)

            if nextId == eot_id:
                break

    newIds = generatedIds[len(promptIds):]
    return tokenizer.decode(newIds)
