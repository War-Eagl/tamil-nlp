import nltk
import streamlit as st
import dill as pickle
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer


nltk.download('punkt')

with open('thirukural_model.pkl','rb') as f:
    thirukural_model = pickle.load(f)

with open('translation.pkl','rb') as f2:
    translation_model = pickle.load(f2)

st.title('Thirukural Generator')



detokenize = TreebankWordDetokenizer().detokenize

def generate_sent(model, num_words, random_seed=42):
    """
    :param model: An ngram language model from `nltk.lm.model`.
    :param num_words: Max no. of words to generate.
    :param random_seed: Seed value for random.
    """
    content = []
    for token in model.generate(num_words, random_seed=random_seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)


if st.button('Generate random thirukural'):
    seed = random.randint(0,2000)
    st.write(generate_sent(thirukural_model, num_words=20, random_seed=seed))


if st.button('Generate random translation'):
    seed = random.randint(0,2000)
    st.write(generate_sent(translation_model, num_words=20, random_seed=seed))