import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import re
import pandas as pd

from collections import Counter
from scipy import stats
import spacy
from textstat.textstat import textstatistics, legacy_round
import random


def avg_term_entropy(text):
    """
    Entropy of terms in a stackoverflow post, essentially calculating the entropy of term distribution with a given set of probabilities.
    """
    labels = [i for i in text.split(" ")]
    ate_val = stats.entropy(list(Counter(labels).values()), base=2)
    return ate_val.item()


def automated_reading_index(text):
    """
    It is a readability test for calculating the understandability of a text on a US grade level.
    """
    num_words = word_count(text)
    num_sentences = sentence_count(text)
    if num_words != 0 and num_sentences != 0:
        return 4.71 * (character_count(text) / word_count(text) ) + 0.5 * (word_count(text) / sentence_count(text)) - 21.43
    return 0.0


def coleman_liau_index(text):
    """
     Metric for understanding the readability of the text. Moreover, it also relies on the number of characters per words as opposed to syllables.
    """
    L = avgLetters(text)
    S = avgSentences(text)
    return  0.588 * L - 0.296 * S  - 15.8


def flesch_kincaid_grade_level(text):
    """
    This is a modified version of Flesch Reading Ease score which is easier to use and directly provides the reading grade level. 
    """
    num_words = word_count(text)
    num_sentences = sentence_count(text)
    num_syllables = total_syllables_count(text)
    if num_words != 0 and num_sentences != 0:
        return 0.39 * (word_count(text) / sentence_count(text)) + 11.8 * (total_syllables_count(text) / word_count(text)) - 15.9
    return 0.0


def flesch_reading_ease_score(text):
    """
    This metric tells the readability of the text as well. It assigns a score between 1 to 100, with a score of 70-80 being that of a school going grade 8 student.
    """
    num_words = word_count(text)
    num_sentences = sentence_count(text)
    num_syllables = total_syllables_count(text)
    if num_words != 0 and num_sentences != 0:    
        return 206.835 - 1.015 * (word_count(text) / sentence_count(text)) - 84.6 *(total_syllables_count(text) / word_count(text))
    return 0.0


def gunning_fox_index(text):
    """
    Gunning Fox Index is mainly used because of its simplicity. It assigns a grading to the text between 0 and 20, with a grading of 9 indicating that the text passage can easily be understood 
    """
    num_words = word_count(text)
    num_sentences = sentence_count(text)
    try:
    	num_diff_words = difficult_words(text)
    	if num_words != 0 and num_sentences != 0:    
        	return 0.4 * (word_count(text) / sentence_count(text) + 100 * (difficult_words(text)/ word_count(text)))
    except:
    	return 0.0
    return 0.0


def LOC(code, body):
    """
    Lines of code
    """
    code_len = len(code)
    body_len = len(body)
    if code_len + body_len != 0:
        return len(code)/float(len(code)+len(body))
    return 0.0

def metric_entropy(text):
    """
    Average term entropy per sentence
    """
    return avg_term_entropy(text) / len(text)

def metric_entropy_temp(text):
    """
    Used for main metric evaluation
    """
    labels = [i for i in text.split(" ")]
    ate_val = stats.entropy(list(Counter(labels).values()), base=2)
    return (ate_val / len(text)).item()

def smog_index(text):
    """
    Estimates the years of education a person needs to comprehend a piece of writing,
    """
    if sentence_count(text) >= 3:
        SMOG = (1.043 * (30*(poly_syllable_count(text) / sentence_count(text)))**0.5) + 3.1291
        return SMOG
    else:
        return 0.0

  
def tag_count(tags):
    """
    Counts the tags in the sentence
    """
    pattern = r"<.*?>"
    if tags is not None:
        tags_list = re.findall(pattern, tags)
        return len(tags_list)
    return 0


def text_similarity(text_1, text_2):
    """
    Measures similarity between texts
    """
    if text_1 is None or text_2 is None:
        return 0.0
    v1 = text_to_vector(text_1)
    v2 = text_to_vector(text_2)    
    return get_cosine(v1, v2)

def fetch_sentence_count(text):
    """
    Counts the total sentences
    """
    return sentence_count(text)


def fetch_word_count(text):
    """
    Counts the total text
    """
    return word_count(text)


def fetch_sentences(text):
    """
    returns sentences from paragraphs
    """
    output = list()
    sentence = ""
    for c in text:
        sentence += c
        if c == '.' or c == '!' or c == '?':
            output.append(sentence)
            sentence = ""
            continue
    return output

def word_count(text):
    """
    Returns word count
    """
    sentences = fetch_sentences(text)
    words = 0
    for sentence in sentences:
        words += len([token for token in sentence])
    return words

def sentence_count(text):
    # returns len of sentences
    return len(fetch_sentences(text))
 
def avg_sentence_length(text):
    # returns avgerage sentence length
    return float(word_count(text) / sentence_count(text))

def character_count(text):
    # returns total character count
    return sum(len(word) for word in text.split(" "))

def syllables_count(word):
    # returns total syllables count
    return textstatistics().syllable_count(word)

def syllables_count_temp(word):
    # returns temporary syllables count
    word = word.lower()
    return len(
        re.findall('(?!e$)[aeiouy]+', word, re.I) +
        re.findall('^[^aeiouy]*e$', word, re.I)
    )

def total_syllables_count(text):
    # returns total syllables count
    return sum(syllables_count_temp(word) for word in text.split(" "))

def avg_syllables_per_word(text):
    # return average syllables per word
    return legacy_round(float(syllables_count_temp(text)) / float(word_count(text), 1))

def poly_syllable_count(text):
    # returns poly syllable count
    count = 0
    words = []
    sentences = fetch_sentences(text)
    for sentence in sentences:
        words += [token for token in sentence]
    for word in words:
        syllable_count = syllables_count_temp(str(word))
        if syllable_count >= 3:
            count += 1
    return count

def difficult_words(text):
    # Find all words in the text
    words = []
    sentences = fetch_sentences(text)
    for sentence in sentences:
        words += [str(token) for token in sentence]
 
    # difficult words are those with syllables >= 2
    # easy_word_set is provide by Textstat as
    # a list of common words
    diff_words_set = set()
     
    for word in words:
        syllable_count = syllables_count_temp(word)
        if word not in STOP_WORDS and syllable_count >= 3:
            diff_words_set.add(word)
 
    return len(diff_words_set)

def avgLetters(text):
    #returns average number of letters per 100 words in a text file
    #uses a list of 100 word chunks to calculate this

    word_list = text.split()
    intervals = range(0, len(word_list), 100)
    word_chunks = [word_list[n:n+100] for n in intervals]
    lettersList = []
    for n in range(0, len(intervals)):
        words = [len(i) for i in word_chunks[n]]
        letters = sum(words)
        lettersList.extend([letters])
    L = sum(lettersList)
    if len(word_chunks) != 0:
        return float(L/len(word_chunks))
    return 0.0

def avgSentences(text):
    #takes entire text and returns average number of sentences per 100 words
    word_list = text.split()
    intervals = range(0, len(word_list), 100)
    word_chunks = [word_list[n:n+100] for n in intervals]
    sentencesList = []
    for n in range(0,len(intervals)):
        sentences = [word.count(".") for word in word_chunks[n]]
        total = sum(sentences)
        sentencesList.extend([total])
    S = sum(sentencesList)
    if len(word_chunks) != 0:
        return float(S/len(word_chunks))
    return 0.0

def text_to_vector(text):
    WORD = re.compile(r"\w+")
    words = WORD.findall(text)
    return Counter(words)

def get_cosine(vec1, vec2):
    """
    returns cosine distance
    """
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = sum1*0.5 * sum2*0.5

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator