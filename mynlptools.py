# My NLP Tools
import re


def strlist2RegexPat(wordlist, mode='and'):
    """Converts a list of string into a regex pattern for AND or OR search.
       Arguments:
       wordlist: list of words
       mode: 'and' (default) or 'or'

       Note:
       'and' considers partial matches of given words
       'or' considers full matches of given words
       both consider a single-line string"""

    if mode == 'and':
        pat = '^' + ''.join(fr'(?=.*{w.strip()})' for w in wordlist)

    elif mode == 'or':
        pat = r'\b(?:{})\b'.format('|'.join(wordlist))

    return pat
    # TO-DO: add options for fullmatch, partial match, single-line/whole-text


def find_words_in_text(text, words_list):
    """Find in the text the words given in the list. Ignores Case.
        Returns iterator yielding match objects"""

    pat_string = '|'.join(fr'\b{w}\b' for w in words_list)
    pat = re.compile(pat_string, flags=re.I | re.X)

    return re.finditer(pat, text)


def clean_text(txt, lower=True, stopwords=None, verbose=0):
    """Prepares text for text processing removing unwanted characters,
       stopwords and lowering it.
       Arguments:
       txt: string to be processed
       lower: boolean
       stopwords: list of words or path to file containing these one per line
       verbose: choose verbosity information"""

    if verbose >= 1:
        print('Initial number of words:', len(txt.split()))

    if lower:
        if verbose >= 2:
            print('Lowering...')
        txt = txt.lower()

    if verbose >= 2:
        print('Removing unwanted characters...')
    txt_p = re.sub(r'[^\wäöüÄÖÜß]+', ' ', txt)
    # txt_p = re.sub(r'\W+', ' ', txt) # if text not german

    if stopwords is not None:

        if verbose >= 2:
            print('Removing stopwords...')

        if isinstance(stopwords, list):
            swds = stopwords

        elif isinstance(stopwords, str):
            with open(stopwords, 'r') as f:
                swds = f.read().splitlines()

        else:
            raise ValueError('Enter the stopword as a list of strings or'
                             ' a file containing these one per line')
        # Remove stopwords
        # ALTERNATIVE to pattern way:
        # (has the advantage that no extra spaces removal step is required)
        txt_p = ' '.join([w for w in txt_p.split() if w not in swds])

        # Pattern way
        # swds_pat = r'\b(?:{})\b'.format('|'.join(swds)) # OR pattern
        # txt_p = re.sub(swds_pat, '', txt_p)

    # Remove extra spaces and strip
    if stopwords is None:  # Otherwise not needed because it happens automatically
        if verbose >= 2:
            print('Removing unnecessary spaces, tabs, line breaks...')
        txt_p = re.sub(r'\s+', ' ', txt_p).strip()  # \s matches any space character (spaces, tabs, line breaks)

    if verbose >= 1:
        print('Number of words after text-clean-up:', len(txt_p.split()))

    return txt_p


def convert_umlauts(text):
    """Replace umlauts with non-diacritic equivalent"""

    text = text.replace('ä', 'ae')
    text = text.replace('ö', 'oe')
    text = text.replace('ü', 'ue')
    text = text.replace('Ä', 'Ae')
    text = text.replace('Ö', 'Oe')
    text = text.replace('Ü', 'Ue')
    text = text.replace('ß', 'ss')

    return text


def get_chunks_basic(text, max_words=256):
    """Split text in chunks with less than max_words"""

    # List of lines skipping empty lines
    lines = [l for l in text.splitlines(True) if l.strip()]

    chunks = []
    chunk = ''
    for l in lines:
        if len(chunk.split() + l.split()) <= max_words:
            chunk += l  # if splitlines(False) do += "\n" + l
            continue
        chunks.append(chunk)
        chunk = l

    if chunk:
        chunks.append(chunk)

    return chunks


def get_chunks_basic_fast(text, max_words=256):  # optimized for performance
    """Split text in chunks with less than max_words"""

    # List of lines skipping empty lines
    lines = [l for l in text.splitlines(True) if l.strip()]

    chunks = []
    chunk = []
    chunk_length = 0
    for l in lines:
        line_length = len(l.split())
        if chunk_length + line_length <= max_words:
            chunk.append(l)
            chunk_length += line_length
            continue
        chunks.append(''.join(chunk))  # if splitlines(False) do "\n".join()
        chunk = [l]
        chunk_length = len(l.split())

    if chunk:
        chunks.append(''.join(chunk))

    return chunks


def get_chunks(text, max_words=256, max_title_words=4):
    """Split text in trivial context-awared chunks with less than max_words"""

    # List of lines skipping empty lines
    lines = [l for l in text.splitlines(True) if l.strip()]

    chunks = []
    chunk = ''
    for l in lines:
        nwords = len(l.split())
        if len(chunk.split()) + nwords <= max_words and (
            nwords > max_title_words
            or all(len(s.split()) <= max_title_words for s in chunk.splitlines())
        ):
            chunk += l  # if splitline(False) do += "\n" + l
            continue
        chunks.append(chunk)
        chunk = l

    if chunk:
        chunks.append(chunk)

    return chunks


def get_chunks_fast(text, max_words=256, max_title_words=4):  # optimized for performance
    """Split text in trivial context-awared chunks with less than max_words"""

    # List of lines skipping empty lines
    lines = [l for l in text.splitlines(True) if l.strip()]

    chunks = []
    chunk = []
    chunk_length = 0
    for l in lines:
        line_length = len(l.split())
        if chunk_length + line_length <= max_words and (
            line_length > max_title_words
            or all(len(s.split()) <= max_title_words for s in chunk)
        ):
            chunk.append(l)
            chunk_length += line_length
            continue
        chunks.append(''.join(chunk))  # if splitlines(False) do "\n".join()
        chunk = [l]
        chunk_length = len(l.split())

    if chunk:
        chunks.append(''.join(chunk))

    return chunks
