import regex as re
import time
import os
from multiprocessing import Pool, cpu_count

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def countPreTokensInChunk(textChunk):
    localCounts = {}
    for match in re.finditer(PAT, textChunk):
        token = tuple(bytes([b]) for b in match.group().encode("utf-8"))
        localCounts[token] = localCounts.get(token, 0) + 1
    return localCounts


def bpe_tokenize(input_path, vocab_size, special_tokens):

    vocab = {i: bytes([i]) for i in range (256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    chunkSize = 50 * 1024 * 1024
    splitPattern = re.compile("|".join(re.escape(t) for t in special_tokens)) if special_tokens else None

    numWorkers = cpu_count()
    pre_token_counts = {}

    fileSizeBytes = os.path.getsize(input_path)
    fileSizeMB = fileSizeBytes / (1024 * 1024)
    print(f"Starting pre-tokenization of {input_path} ({fileSizeMB:.0f} MB) with {numWorkers} workers", flush=True)
    preTokenStartTime = time.time()
    bytesProcessed = 0
    chunkNumber = 0

    with open(input_path, "r") as f:
        leftover = ""
        while True:
            rawBlock = f.read(chunkSize)
            if not rawBlock:
                if leftover:
                    localCounts = countPreTokensInChunk(leftover)
                    for token, count in localCounts.items():
                        pre_token_counts[token] = pre_token_counts.get(token, 0) + count
                break

            rawBlock = leftover + rawBlock
            leftover = ""

            lastSpecialTokenPosition = -1
            if splitPattern:
                for match in splitPattern.finditer(rawBlock):
                    lastSpecialTokenPosition = match.end()

            if lastSpecialTokenPosition == -1:
                leftover = rawBlock
                continue

            processableText = rawBlock[:lastSpecialTokenPosition]
            leftover = rawBlock[lastSpecialTokenPosition:]

            if splitPattern:
                documents = splitPattern.split(processableText)
            else:
                documents = [processableText]

            documents = [doc for doc in documents if doc]

            with Pool(numWorkers) as pool:
                chunkResults = pool.map(countPreTokensInChunk, documents)

            for localCounts in chunkResults:
                for token, count in localCounts.items():
                    pre_token_counts[token] = pre_token_counts.get(token, 0) + count

            chunkNumber += 1
            bytesProcessed += len(processableText.encode("utf-8"))
            percentDone = (bytesProcessed / fileSizeBytes) * 100
            print(f"Pre-tokenization chunk {chunkNumber}: {percentDone:.1f}% done ({bytesProcessed / (1024*1024):.0f}/{fileSizeMB:.0f} MB)", flush=True)

    preTokenElapsed = time.time() - preTokenStartTime
    print(f"Pre-tokenization complete in {preTokenElapsed:.1f}s | {len(pre_token_counts)} unique pre-tokens", flush=True)

    wordList = []
    for tokenSequence, frequency in pre_token_counts.items():
        wordList.append((list(tokenSequence), frequency))

    print(f"Building initial pair frequencies from {len(wordList)} words", flush=True)
    pairFrequencies = {}
    pairToWordIndices = {}

    for wordIndex, (sequence, frequency) in enumerate(wordList):
        for i in range(len(sequence) - 1):
            currentPair = (sequence[i], sequence[i + 1])
            pairFrequencies[currentPair] = pairFrequencies.get(currentPair, 0) + frequency
            if currentPair not in pairToWordIndices:
                pairToWordIndices[currentPair] = set()
            pairToWordIndices[currentPair].add(wordIndex)

    print(f"Initial pair frequencies built: {len(pairFrequencies)} unique pairs", flush=True)

    merges = []
    numberOfMerges = vocab_size - 256 - len(special_tokens)
    mergeStartTime = time.time()

    print(f"Starting {numberOfMerges} merges", flush=True)

    for mergeIndex in range(numberOfMerges):
        if not pairFrequencies:
            break

        bestPair = max(pairFrequencies, key=lambda p: (pairFrequencies[p], p))

        merges.append(bestPair)
        mergedToken = bestPair[0] + bestPair[1]
        vocab[len(vocab)] = mergedToken

        affectedWordIndices = pairToWordIndices.pop(bestPair, set())
        del pairFrequencies[bestPair]

        for wordIndex in affectedWordIndices:
            sequence, frequency = wordList[wordIndex]

            # Build new sequence and find merge positions in one pass
            newSequence = []
            mergePositionsOld = []
            newMergedPositions = []
            i = 0
            while i < len(sequence):
                if i < len(sequence) - 1 and sequence[i] == bestPair[0] and sequence[i + 1] == bestPair[1]:
                    mergePositionsOld.append(i)
                    newMergedPositions.append(len(newSequence))
                    newSequence.append(mergedToken)
                    i += 2
                else:
                    newSequence.append(sequence[i])
                    i += 1

            if not mergePositionsOld:
                continue

            # Decrement only neighbor pairs of merge sites in old sequence
            decrementPositions = set()
            for pos in mergePositionsOld:
                if pos > 0:
                    decrementPositions.add(pos - 1)
                if pos + 2 < len(sequence):
                    decrementPositions.add(pos + 1)

            for pos in decrementPositions:
                oldPair = (sequence[pos], sequence[pos + 1])
                if oldPair == bestPair:
                    continue
                pairFrequencies[oldPair] = pairFrequencies.get(oldPair, 0) - frequency
                if pairFrequencies[oldPair] <= 0:
                    del pairFrequencies[oldPair]

            # Increment only neighbor pairs of merged tokens in new sequence
            incrementPositions = set()
            for j in newMergedPositions:
                if j > 0:
                    incrementPositions.add(j - 1)
                if j + 1 < len(newSequence):
                    incrementPositions.add(j)

            for pos in incrementPositions:
                newPair = (newSequence[pos], newSequence[pos + 1])
                pairFrequencies[newPair] = pairFrequencies.get(newPair, 0) + frequency
                if newPair not in pairToWordIndices:
                    pairToWordIndices[newPair] = set()
                pairToWordIndices[newPair].add(wordIndex)

            wordList[wordIndex] = (newSequence, frequency)

        if (mergeIndex + 1) % 1000 == 0 or mergeIndex == numberOfMerges - 1:
            mergeElapsed = time.time() - mergeStartTime
            mergesPerSecond = (mergeIndex + 1) / mergeElapsed if mergeElapsed > 0 else 0
            estimatedRemainingSeconds = (numberOfMerges - mergeIndex - 1) / mergesPerSecond if mergesPerSecond > 0 else 0
            estimatedRemainingHours = estimatedRemainingSeconds / 3600
            totalElapsedHours = (time.time() - preTokenStartTime) / 3600
            print(
                f"Merge {mergeIndex + 1}/{numberOfMerges} "
                f"({(mergeIndex + 1) / numberOfMerges * 100:.1f}%) | "
                f"{mergesPerSecond:.1f} merges/s | "
                f"ETA: {estimatedRemainingHours:.2f}h | "
                f"Total elapsed: {totalElapsedHours:.2f}h",
                flush=True
            )

    totalTime = time.time() - preTokenStartTime
    print(f"Training complete! Total time: {totalTime / 3600:.2f} hours ({totalTime:.0f}s)", flush=True)

    return vocab, merges


if __name__ == "__main__":
    import sys
    import json

    dataset = sys.argv[1] if len(sys.argv) > 1 else "tinystories"

    if dataset == "owt":
        inputPath = "data/owt_train.txt"
        vocabSize = 32000
        outputPrefix = "data/owt"
    else:
        inputPath = "data/TinyStoriesV2-GPT4-train.txt"
        vocabSize = 10000
        outputPrefix = "data/ts"

    specialTokens = ["<|endoftext|>"]

    print("=" * 60, flush=True)
    print(f"BPE Training: dataset={dataset}, vocab_size={vocabSize}", flush=True)
    print(f"Input: {inputPath}", flush=True)
    print(f"CPUs available: {cpu_count()}", flush=True)
    print("=" * 60, flush=True)

    trainedVocab, trainedMerges = bpe_tokenize(inputPath, vocabSize, specialTokens)

    print(f"\n{'=' * 60}", flush=True)
    print("RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"Vocab size: {len(trainedVocab)}", flush=True)

    longestToken = max(trainedVocab.values(), key=len)
    print(f"Longest token: {longestToken} ({len(longestToken)} bytes)", flush=True)

    serializedVocab = {}
    for tokenId, tokenBytes in trainedVocab.items():
        serializedVocab[tokenId] = list(tokenBytes)

    with open(f"{outputPrefix}_vocab.json", "w") as f:
        json.dump(serializedVocab, f)

    with open(f"{outputPrefix}_merges.txt", "w") as f:
        for tokenA, tokenB in trainedMerges:
            f.write(f"{tokenA} {tokenB}\n")

    print(f"Saved vocab to {outputPrefix}_vocab.json", flush=True)
    print(f"Saved merges to {outputPrefix}_merges.txt", flush=True)

