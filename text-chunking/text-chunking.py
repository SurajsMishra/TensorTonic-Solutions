def text_chunking(tokens: list, chunk_size: int, overlap: int) -> list:
    """
    Returns fixed-size token chunks with the requested overlap.
    """
    step = chunk_size - overlap
    chunks = []
    i =0
    while i < len(tokens):
        chunk = tokens[i : i+chunk_size]
        chunks.append(chunk)
        if i + chunk_size >= len(tokens):
            break
        i += step
    return chunks
    
    pass