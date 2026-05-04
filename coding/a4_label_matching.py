import os
from nltk import pos_tag, download
from nltk.corpus import brown, gutenberg, wordnet as wn
from collections import Counter
from nltk.stem import WordNetLemmatizer
import clip



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


def get_class_names_from_folder(dataset_path):
    return [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]


def generate_label_candidates(limit = 20000, domainnet_path = "../../../../data/public/DomainNet/real"  ):
    download('wordnet')
    download('brown')
    download('gutenberg')

    lemmatizer = WordNetLemmatizer()
    
    wordnet_words = get_high_freq_words(limit)
    
    wordnet_set = set(wordnet_words)
    
    candidates = wordnet_set
    
    print(f"Candidates labels number: {len(candidates)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)    
    
    candidates_dict = {}
    for label in candidates:
        prompt = f"This is a picture of {label}"
        with torch.no_grad():
            text_features = model.encode_text(clip.tokenize([prompt]).to(device)).cpu().numpy()
        candidates_dict[label] = text_features.flatten()    
    print(type(candidates_dict))
    filename = 'candidates_dict_40000.npz'
    np.savez(filename, candidates_dict=candidates_dict)
    return filename
    
    
def distance_based_label_generation(features, candidates_dir = '', domainnet_path = "../../../../data/public/DomainNet/real"):
    if candidates_dir == '':
        candidates_dir = generate_label_candidates(limit = 40000)
    loaded_data = np.load(candidates_dir,allow_pickle=True)
    candidates_dict = loaded_data['candidates_dict'].item()
    
    labels = list(candidates_dict.keys()) 
    
    wordnet_set = set(labels)
    domainnet_labels = get_class_names_from_folder(domainnet_path)
    domainnet_set = set(domainnet_labels)
    
    included_labels = domainnet_set & wordnet_set
    #print(f"DomainNet labels number: {len(domainnet_set)}")
    #print(f"including rate: {len(included_labels)/len(domainnet_set)}")
    
    
    candidate_features = np.stack([candidates_dict[label] for label in labels])
    candidate_features = torch.tensor(candidate_features).float().to(features.device)  # shape: (num_candidates, 512)
    
    features = torch.tensor(features).float()  # shape: (batch_size, 512)
    
    similarity = torch.matmul(features, candidate_features.T)  # shape: (batch_size, num_candidates)
    similarity = similarity / (features.norm(dim=1, keepdim=True) * candidate_features.norm(dim=1))
    
    top_k = 3  
    top_k_indices = torch.topk(similarity, k=top_k, dim=1, largest=True).indices
    
    
    batch_labels = [[labels[idx] for idx in indices] for indices in top_k_indices]  # shape: (batch_size, 3)

    return batch_labels
    
    
    
###############here integrated in adaption.py#################
import clip
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score



def load_domainnet_images(domainnet_path, batch_size=32, num_workers=4):
    preprocess = clip.load("ViT-B/32", device="cpu")[1]
    dataset = ImageFolder(domainnet_path, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader, dataset.classes


def compute_top_k_accuracy(predictions, targets, k=3):
    top_k_hits = 0
    total = len(targets)
    for i, target in enumerate(targets):
        if target in predictions[i][:k]:  
            top_k_hits += 1
    return top_k_hits / total



def run_distance_based_label_generation(domainnet_path, candidates_dir='', k=3):
    # load dataset
    dataloader, class_names = load_domainnet_images(domainnet_path)

    # load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # label matching
    all_predictions = []
    all_targets = []
    
    
    for images, labels in tqdm(dataloader, desc="Processing Images"):
        images = images.to(device)
        # CLIP features
        with torch.no_grad():
            features = model.encode_image(images).float()  # shape: (batch_size, 512)

        # generate top-k label
        predictions = distance_based_label_generation(features, candidates_dir=candidates_dir, domainnet_path = domainnet_path)
        all_predictions.extend(predictions)
        #print('generated label: ', predictions)
        # original label 
        targets = [class_names[label] for label in labels]
        all_targets.extend(targets)
        #print('original label: ', targets)

    # top-k acc
    accuracy = compute_top_k_accuracy(all_predictions, all_targets, k=k)
    print(f"Top-{k} Accuracy: {accuracy * 100:.2f}%")
    return accuracy


domainnet_painting = "../../../../data/public/DomainNet/painting"
domainnet_real = "../../../../data/public/DomainNet/real"
visda = "../../../../data/public/visda-2017/train"
print('5000 candidates')
run_distance_based_label_generation(domainnet_path = "../../../../data/public/visda-2017/train", candidates_dir ='candidates/candidates_dict_5000.npz')

print('10000 candidates')
run_distance_based_label_generation(domainnet_path = "../../../../data/public/visda-2017/train", candidates_dir ='candidates/candidates_dict_10000.npz')

print('20000 candidates')
run_distance_based_label_generation(domainnet_path = "../../../../data/public/visda-2017/train", candidates_dir ='candidates/candidates_dict_20000.npz')

print('40000 candidates')
run_distance_based_label_generation(domainnet_path = "../../../../data/public/visda-2017/train", candidates_dir ='candidates/candidates_dict_40000.npz')