def chunk_resume_text(text, headers):
    '''
    Function to chunk a given text based on the list of headers 

    Input: 
        + text: string, a single string represented text of the whole resume 
        + headers: List[] of headers 

    Output:
        + List[String]: each string is a partition / a chunk after being chunked by headers 
    '''
    # 1. Find start index of each header in the text
    positions = []
    for h in headers:
        idx = text.find(h)
        if idx != -1:
            positions.append((idx, h))
    # 2. Sort headers by their appearance order
    positions.sort(key=lambda x: x[0])

    # 3. Slice the text into chunks
    chunks = []
    for i, (start_idx, header) in enumerate(positions):
        end_idx = positions[i+1][0] if i+1 < len(positions) else len(text)
        chunk = text[start_idx:end_idx].strip()
        chunks.append(chunk)

    return chunks


