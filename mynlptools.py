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


def cleanText(txt, lower=True, stopwords=None, verbose=0):
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
        if verbose >= 2: print('Lowering...')
        txt = txt.lower()

    if verbose >= 2: print('Removing unwanted characters...')
    txt_p = re.sub(r'[^\w\säöüÄÖÜß]+', '', txt)

    if stopwords is not None:

        if verbose >= 2: print('Removing stopwords...')

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
        #(has the advantage that no extra spaces removal step is required)
        txt_p = ' '.join([w for w in txt_p.split() if w not in swds])

        # Pattern way
        #swds_pat = r'\b(?:{})\b'.format('|'.join(swds)) # OR pattern
        #txt_p = re.sub(swds_pat, '', txt_p)

    # Remove extra spaces and strip
    if stopwords is None: # Otherwise not needed because it happens automatically
        if verbose >= 2: print('Removing unnecessary spaces, tabs, line breaks...')
        txt_p = re.sub(r'\s+', ' ', txt_p).strip() # \s matches any space character (spaces, tabs, line breaks)

    if verbose >= 1:
        print('Number of words after text-clean-up:', len(txt_p.split()))

    return txt_p


def convertUmlauts(word):
    """
    Replace umlauts for a given text    
    :param word: text as string
    :return: manipulated text as str
    """
    tempVar = word
    
    tempVar = tempVar.replace('ä', 'ae')
    tempVar = tempVar.replace('ö', 'oe')
    tempVar = tempVar.replace('ü', 'ue')
    tempVar = tempVar.replace('Ä', 'Ae')
    tempVar = tempVar.replace('Ö', 'Oe')
    tempVar = tempVar.replace('Ü', 'Ue')
    tempVar = tempVar.replace('ß', 'ss')
    
    return tempVar
