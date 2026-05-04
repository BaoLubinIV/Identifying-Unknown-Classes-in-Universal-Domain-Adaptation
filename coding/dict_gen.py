import os
from nltk import pos_tag, download
from nltk.corpus import brown, gutenberg, wordnet as wn
from collections import Counter
from nltk.stem import WordNetLemmatizer
import clip
import torch
import numpy as np

def is_concrete_noun(word):
    synsets = wn.synsets(word, pos=wn.NOUN)
    score = 0
    for synset in synsets:
        hypernyms = set(hyp.name().split('.')[0] for hyp in synset.hypernym_paths()[0])
        
        
        if 'physical_entity' in hypernyms:
            score += 10
        if 'artifact' in hypernyms:
            score += 5
        if 'plant' in hypernyms:
            score += 5
        if 'animal' in hypernyms:
            score += 5
        if 'phenomenon' in hypernyms: 
            score += 8
        if 'structure' in hypernyms: 
            score += 3
        if 'container' in hypernyms: 
            score += 5
        if 'tool' in hypernyms:  
            score += 5
        if 'food' in hypernyms:  
            score += 5
        if 'instrument' in hypernyms: 
            score += 5
        if 'vehicle' in hypernyms: 
            score += 5
        if 'body_part' in hypernyms:  
            score += 5
        if 'organ' in hypernyms:
            score += 5
        if 'clothing' in hypernyms:
            score += 5
        if 'furniture' in hypernyms:
            score += 5
        if 'equipment' in hypernyms:
            score += 5
        if 'shape' in hypernyms:
            score += 3
        
        
        if 'abstraction' in hypernyms:
            score -= 3
        if 'attribute' in hypernyms:
            score -= 3
        if 'psychological_feature' in hypernyms:
            score -= 3
        if 'event' in hypernyms:
            score -= 1
        if 'time' in hypernyms:
            score -= 3
        if 'group' in hypernyms:
            score -= 2
        if 'state' in hypernyms:
            score -= 3
    return score


def get_high_freq_words(limit):
    brown_freq = Counter(brown.words())
    gutenberg_freq = Counter(gutenberg.words())
    all_synsets = wn.all_synsets(pos='n')  
    words_with_scores = []

    for synset in all_synsets:
        for lemma in synset.lemmas():
            word = lemma.name() 
            freq = 10*lemma.count()+ brown_freq[word] + gutenberg_freq[word]
            score = is_concrete_noun(word)
            if score > 0:  
                words_with_scores.append((word, freq, score))


    unique_words = {}
    for word, freq, score in words_with_scores:
        if word not in unique_words or freq > unique_words[word][0]:
            unique_words[word] = (freq, score)


    deduplicated_words = [(word, freq, score) for word, (freq, score) in unique_words.items()]
    deduplicated_words = sorted(deduplicated_words, key=lambda x: (x[1], x[2]), reverse=True)

    nouns = [word for word, freq, score in deduplicated_words]

    if limit < len(nouns):
        nouns = nouns[:limit]
    return nouns
    
    
    
    
def generate_label_candidates(limit = 20000):
    print("generating candidates labels for unknown sample definition")
    download('wordnet')
    download('brown')
    download('gutenberg')

    lemmatizer = WordNetLemmatizer()
    
    wordnet_words = get_high_freq_words(limit)
    candidates = set(wordnet_words)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)    
    
    candidates_dict = {}
    for label in candidates:
        prompt = f"This is a picture of {label}"
        with torch.no_grad():
            text_features = model.encode_text(clip.tokenize([prompt]).to(device)).cpu().numpy()
        candidates_dict[label] = text_features.flatten()    
    np.savez(file = 'candidates_dict.npz', candidates_dict=candidates_dict)