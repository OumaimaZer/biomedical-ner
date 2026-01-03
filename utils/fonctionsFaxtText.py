import os
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk.tokenize import sent_tokenize
from gensim.models import FastText
from collections import Counter, defaultdict
import random

# Créer les dossiers de sortie si absents
os.makedirs("./models", exist_ok=True)
os.makedirs("./plots", exist_ok=True)

# ─────────────────────────────────────────────
# CONFIGURATION NLTK
# ─────────────────────────────────────────────
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ─────────────────────────────────────────────
# CHARGEMENT DU DATASET JNLPBA
# ─────────────────────────────────────────────
def load_jnlpba_dataset(base_path):
    print(f"Chargement du dataset JNLPBA depuis: {base_path}")

    all_sentences = []

    for filename in ['train.tsv', 'devel.tsv', 'test.tsv']:
        file_path = os.path.join(base_path, filename)
        if not os.path.exists(file_path):
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            sentence = []
            for line in f:
                line = line.strip()

                if not line:
                    if sentence:
                        all_sentences.append(sentence)
                        sentence = []
                    continue

                if line.startswith('-DOCSTART-'):
                    continue

                parts = line.split('\t')
                if len(parts) >= 2:
                    token, label = parts[0], parts[1]
                    sentence.append((token, label))

            if sentence:
                all_sentences.append(sentence)

    classes = [
        'B-DNA', 'I-DNA',
        'B-cell_line', 'I-cell_line',
        'B-protein', 'I-protein',
        'B-cell_type', 'I-cell_type',
        'B-RNA', 'I-RNA',
        'O'
    ]

    print(f"- phrases: {len(all_sentences)}")
    print(f"- classes: {classes}")

    return all_sentences, classes


# ─────────────────────────────────────────────
# CHARGEMENT DU DATASET NCBI
# ─────────────────────────────────────────────
def load_ncbi_dataset(folder_path):
    print(f"Chargement du dataset NCBI depuis: {folder_path}")

    data = []
    pattern = r'<category="([^"]+)">([^<]+)</category>'

    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) < 3:
                    continue

                doc_id, title, text = parts[0], parts[1], parts[2]

                entities = []
                offset = 0

                for match in re.finditer(pattern, text):
                    category = match.group(1)
                    mention = match.group(2)

                    start = match.start() - offset
                    end = match.end() - offset

                    entities.append({
                        "start": start,
                        "end": end,
                        "text": mention,
                        "type": category
                    })

                    offset += len(match.group(0)) - len(mention)

                clean_text = re.sub(pattern, r'\2', text)

                data.append({
                    "id": doc_id,
                    "title": title,
                    "text": clean_text,
                    "entities": entities
                })

    print(f"Documents chargés: {len(data)}")
    return data


# ─────────────────────────────────────────────
# ENTRAÎNEMENT DES EMBEDDINGS FASTTEXT
# ─────────────────────────────────────────────
def train_fasttext_embeddings(sentences, vector_size=200, window=5, min_count=2, workers=4, min_n=3, max_n=6):
    """
    Entraîne FastText sur les phrases, compatible avec :
    - JNLPBA : liste de [(token, label), ...]
    - NCBI : liste de (tokens, labels) — tokens = [str], labels = [str]
    - Augmenté : liste de [(token, label, *extras), ...]
    """
    tokenized = []
    for sent in sentences:
        if isinstance(sent, (list, tuple)) and len(sent) == 2 and isinstance(sent[1], list):
            # Format NCBI : (liste_tokens, liste_labels)
            tokens, labels = sent
            tokenized.append([t.lower() for t in tokens])
        elif isinstance(sent, list) and sent:
            # JNLPBA ou augmenté : liste de tokens (tuples ou listes)
            tokens = []
            for tok in sent:
                if isinstance(tok, (list, tuple)):
                    # Prend le premier élément comme token, ignore le reste
                    tokens.append(tok[0].lower())
                else:
                    # Cas de secours : suppose une chaîne
                    tokens.append(str(tok).lower())
            tokenized.append(tokens)
        else:
            print(f" Phrase invalide ignorée : {type(sent)} {sent[:3] if hasattr(sent, '__len__') else sent}")
            continue

    print(f"Entraînement de FastText sur {len(tokenized)} phrases...")
    from gensim.models import FastText
    model = FastText(
        sentences=tokenized, # Texte brut tokenisé
        vector_size=vector_size, # 200
        window=window, # La taille de la fenêtre de contexte
        min_count=min_count, # Nombre minimal d’occurrences d’un mot
        workers=workers, # Nombre de threads CPU utilisés pour l’entraînement.
        # Un mot est décomposé en n-grammes de caractères (sous-mots)
        min_n=min_n,
        max_n=max_n,
        sg=1, # Skip-gram = 1 sinon CBOW
        epochs=10 # Suffisant pour un corpus de taille moyenne
    )
    return model


# ─────────────────────────────────────────────
# SAUVEGARDE / CHARGEMENT DE FASTTEXT
# ─────────────────────────────────────────────
def save_fasttext_model(model, filepath):
    if not filepath.endswith(".model"):
        filepath += ".model"

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    print(f"FastText sauvegardé : {filepath}")
    return filepath


def load_fasttext_model(filepath):
    if not filepath.endswith(".model"):
        filepath += ".model"

    if not os.path.exists(filepath):
        print("Modèle introuvable.")
        return None

    model = FastText.load(filepath)
    print(f"FastText chargé : {filepath}")
    return model


# ─────────────────────────────────────────────
# CRÉATION DE LA MATRICE D'EMBEDDING
# ─────────────────────────────────────────────
# Ajout de validation du dataset
def create_embedding_matrix_from_fasttext(model, vocab, vector_size=200):
    embedding_matrix = np.zeros((len(vocab), vector_size))
    
    oov_words = [] # Liste des mots absents de fastText.
    oov_count = 0 # nombre de mots OOV.
    
    for word, idx in vocab.items():
        if word == "<PAD>":
            embedding_matrix[idx] = np.zeros(vector_size)
        elif word in ["<UNK>", "<NUM>"]:
            # Initialisation améliorée pour les tokens spéciaux
            embedding_matrix[idx] = np.random.uniform(-0.1, 0.1, size=vector_size)
        else:
            # FastText gère les OOV via les sous-mots, mais on les trace
            embedding_matrix[idx] = model.wv[word.lower()]
            if word.lower() not in model.wv:
                oov_words.append(word)
                oov_count += 1
    
    if oov_count > 0:
        print(f"Avertissement : {oov_count} mots absents du vocabulaire FastText")
        if oov_count < 20:
            print(f"Mots OOV : {oov_words}")
    
    return embedding_matrix



def analyze_dataset_statistics(sentences, dataset_name="JNLPBA"):
    """Analyse détaillée du dataset de NER"""
    total_tokens = 0
    entity_counts = Counter()
    sentence_lengths = []
    
    for tokens, labels in sentences:
        total_tokens += len(tokens)
        sentence_lengths.append(len(tokens))
        
        current_entity = None
        for label in labels:
            if label != 'O':
                entity_counts[label] += 1
    
    print(f"\n=== Statistiques {dataset_name} ===")
    print(f"Phrases totales : {len(sentences)}")
    print(f"Tokens totaux : {total_tokens}")
    print(f"Longueur moyenne d'une phrase : {np.mean(sentence_lengths):.1f}")
    print(f"Distribution des entités :")
    for entity, count in entity_counts.most_common():
        print(f"  {entity}: {count}")


def visualize_dataset_distribution_ncbi(results, dataset_name="Dataset"):
    """
    Affiche des statistiques détaillées sur le dataset (compatible JNLPBA et NCBI)
    
    Args:
        results: Dictionnaire contenant 'train_sentences', etc.
                 - JNLPBA: liste de [(mot, tag), ...]
                 - NCBI: liste de ([mots], [tags])
        dataset_name: Nom du dataset
    """
    colors = {'train': 'skyblue', 'dev': 'orange', 'test': 'green'}
    
    print("=" * 60)
    print(f"ANALYSE DU DATASET : {dataset_name}")
    print("=" * 60)
    
    splits = [split for split in ['train', 'dev', 'test'] if f'{split}_sentences' in results]
    if not splits:
        print("ERREUR : Aucune donnée trouvée.")
        return
    
    # Détection automatique du format via la première phrase non vide
    def detect_format(sentences):
        for sent in sentences:
            if not sent:
                continue
            if isinstance(sent, list) and len(sent) > 0:
                if isinstance(sent[0], tuple) and len(sent[0]) == 2:
                    return "JNLPBA"  # [(mot, tag), ...]
                elif isinstance(sent[0], str):
                    return "JNLPBA_FLAT"  # ancienne liste plate (peu probable)
            elif isinstance(sent, tuple) and len(sent) == 2:
                tokens, tags = sent
                if isinstance(tokens, list) and isinstance(tags, list):
                    return "NCBI"  # ([mots], [tags])
        return "INCONNU"

    def extract_tags(sent):
        """Extraction unifiée des tags pour les deux formats"""
        if isinstance(sent, list):
            # JNLPBA : [(mot, tag), ...]
            if sent and isinstance(sent[0], tuple):
                return [tag for _, tag in sent]
            # Cas de secours : suppose une liste plate de tags (à éviter)
            return [tag for tag in sent if isinstance(tag, str)]
        elif isinstance(sent, tuple) and len(sent) == 2:
            # NCBI : ([mots], [tags])
            tokens, tags = sent
            return list(tags) if isinstance(tags, (list, tuple)) else []
        return []

    def extract_token_count(sent):
        if isinstance(sent, list):
            return len(sent)
        elif isinstance(sent, tuple) and len(sent) == 2:
            tokens, _ = sent
            return len(tokens)
        return 0

    # 2. Statistiques par partition
    print("\n1. RÉPARTITION DES DONNÉES")
    print("-" * 40)
    
    stats = {}
    format_detected = None
    
    for split in splits:
        sentences = results[f'{split}_sentences']
        if not sentences:
            print(f"{split.upper()}: vide")
            continue
            
        # Détection du format (une seule fois)
        if format_detected is None:
            format_detected = detect_format(sentences)
            print(f"Format détecté : {format_detected}")

        num_sentences = len(sentences)
        num_tokens = sum(extract_token_count(sent) for sent in sentences)
        num_entities = sum(1 for sent in sentences for tag in extract_tags(sent) if tag != 'O')
        
        stats[split] = {
            'sentences': num_sentences,
            'tokens': num_tokens,
            'entities': num_entities
        }
        
        print(f"\n{split.upper()}:")
        print(f"  Phrases : {num_sentences:,}")
        print(f"  Tokens : {num_tokens:,}")
        print(f"  Entités nommées : {num_entities:,}")
        if num_tokens > 0:
            print(f"  Densité d'entités : {num_entities/num_tokens*100:.1f}%")
    
    # 3. Longueur des phrases
    print("\n2. LONGUEUR DES PHRASES")
    print("-" * 40)
    
    fig, axes = plt.subplots(1, len(splits), figsize=(15, 4))
    if len(splits) == 1:
        axes = [axes]
    
    for idx, split in enumerate(splits):
        sentences = results[f'{split}_sentences']
        lengths = [extract_token_count(sent) for sent in sentences]
        
        mean_len = np.mean(lengths) if lengths else 0
        median_len = np.median(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        min_len = min(lengths) if lengths else 0
        
        print(f"\n{split.upper()}:")
        print(f"  Moyenne : {mean_len:.1f} tokens")
        print(f"  Médiane : {median_len:.1f} tokens")
        print(f"  Min-Max : {min_len}-{max_len} tokens")
        print(f"  >100 tokens : {sum(1 for l in lengths if l > 100):,}")
        
        ax = axes[idx]
        ax.hist(lengths, bins=30, color=colors[split], edgecolor='black', alpha=0.7)
        ax.axvline(mean_len, color='red', linestyle='--', label=f'Moyenne : {mean_len:.1f}')
        ax.axvline(median_len, color='green', linestyle='--', label=f'Médiane : {median_len:.1f}')
        ax.set_xlabel('Nombre de tokens')
        ax.set_ylabel('Nombre de phrases')
        ax.set_title(f'Longueur des phrases - {split}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Distribution des longueurs - {dataset_name}', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 4. Distribution des classes
    print("\n3. DISTRIBUTION DES CLASSES D'ENTITÉS")
    print("-" * 40)
    
    all_classes = set()
    class_distributions = {}
    
    for split in splits:
        sentences = results[f'{split}_sentences']
        labels = [tag for sent in sentences for tag in extract_tags(sent)]
        counter = Counter(labels)
        class_distributions[split] = counter
        all_classes.update(counter.keys())
    
    sorted_classes = sorted([c for c in all_classes if c != 'O'])
    if 'O' in all_classes:
        sorted_classes = ['O'] + sorted_classes
    
    # Affichage du tableau
    print("\nFréquences absolues :")
    header = f"{'Classe':<20} " + " ".join([f"{s.upper():>10}" for s in splits])
    print(header)
    print("-" * (20 + 11 * len(splits)))
    
    for cls in sorted_classes:
        row = f"{cls:<20}"
        for split in splits:
            count = class_distributions[split].get(cls, 0)
            row += f" {count:>10,}"
        print(row)
    
    # Pourcentages
    print("\nPourcentages (par partition) :")
    header = f"{'Classe':<20} " + " ".join([f"{s.upper():>10}" for s in splits])
    print(header)
    print("-" * (20 + 11 * len(splits)))
    
    for cls in sorted_classes:
        row = f"{cls:<20}"
        for split in splits:
            total = sum(class_distributions[split].values())
            count = class_distributions[split].get(cls, 0)
            percentage = (count / total * 100) if total > 0 else 0
            row += f" {percentage:>9.1f}%"
        print(row)
    
    # 5. Graphiques (sans 'O')
    print("\n4. VISUALISATION DES ENTITÉS (sans 'O')")
    
    entity_classes = [c for c in sorted_classes if c != 'O']
    if entity_classes:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Diagramme en barres
        x = np.arange(len(entity_classes))
        width = 0.25
        
        for i, split in enumerate(splits):
            counts = [class_distributions[split].get(cls, 0) for cls in entity_classes]
            axes[0].bar(x + i*width, counts, width=width, 
                       color=colors[split], label=split, edgecolor='black')
        
        axes[0].set_xlabel('Classes d\'entités')
        axes[0].set_ylabel('Nombre d\'occurrences')
        axes[0].set_title(f'Distribution des entités par partition - {dataset_name}')
        axes[0].set_xticks(x + width*(len(splits)-1)/2)
        axes[0].set_xticklabels(entity_classes, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Camembert (train uniquement)
        if 'train' in splits:
            train_counts = [class_distributions['train'].get(cls, 0) for cls in entity_classes]
            nonzero = [(cls, cnt) for cls, cnt in zip(entity_classes, train_counts) if cnt > 0]
            if nonzero:
                labs, vals = zip(*nonzero)
                axes[1].pie(vals, labels=labs, autopct='%1.1f%%', startangle=90)
                axes[1].set_title(f'Distribution des entités - Train')
            else:
                axes[1].text(0.5, 0.5, 'Aucune entité', ha='center', va='center')
        
        plt.tight_layout()
        plt.show()
    
    # 6. Entités par phrase
    print("\n5. ENTITÉS PAR PHRASE")
    print("-" * 40)
    
    fig, axes = plt.subplots(1, len(splits), figsize=(15, 4))
    if len(splits) == 1:
        axes = [axes]
    
    for idx, split in enumerate(splits):
        sentences = results[f'{split}_sentences']
        entities_per_sentence = [
            sum(1 for tag in extract_tags(sent) if tag != 'O')
            for sent in sentences
        ]
        
        mean_ent = np.mean(entities_per_sentence) if entities_per_sentence else 0
        median_ent = np.median(entities_per_sentence) if entities_per_sentence else 0
        zero_ent = sum(1 for e in entities_per_sentence if e == 0)
        
        print(f"\n{split.upper()}:")
        print(f"  Entités/phrase (moy) : {mean_ent:.2f}")
        print(f"  Entités/phrase (méd) : {median_ent:.1f}")
        print(f"  Phrases sans entité : {zero_ent:,} ({zero_ent/len(sentences)*100:.1f}%)")
        
        ax = axes[idx]
        ax.hist(entities_per_sentence, bins=20, color=colors[split], edgecolor='black', alpha=0.7)
        ax.set_xlabel('Nombre d\'entités')
        ax.set_ylabel('Nombre de phrases')
        ax.set_title(f'Entités par phrase - {split}')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Entités par phrase - {dataset_name}', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 7. Analyse des tags BIO
    print("\n6. ANALYSE DES TAGS BIO")
    print("-" * 40)
    
    bio_stats = {'B': 0, 'I': 0, 'O': 0, 'autres': 0}
    for split in splits:
        for sent in results[f'{split}_sentences']:
            for tag in extract_tags(sent):
                if tag == 'O':
                    bio_stats['O'] += 1
                elif tag.startswith('B-'):
                    bio_stats['B'] += 1
                elif tag.startswith('I-'):
                    bio_stats['I'] += 1
                else:
                    bio_stats['autres'] += 1
    
    total = sum(bio_stats.values())
    print(f"Total des tags : {total:,}")
    for k, v in bio_stats.items():
        pct = v/total*100 if total else 0
        print(f"  {k}: {v:,} ({pct:.1f}%)")
    
    # 8. Informations supplémentaires
    print("\n7. INFORMATIONS SUPPLÉMENTAIRES")
    print("-" * 40)
    if 'vocab' in results:
        print(f"Vocabulaire : {len(results['vocab']):,} mots")
    if 'tag_to_idx' in results:
        print(f"Classes : {list(results['tag_to_idx'].keys())}")
    
    print("\n" + "=" * 60)
    print("ANALYSE TERMINÉE")
    print("=" * 60)
    

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Augmentation des données
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 1. Filtre de plausibilité biologique
# ─────────────────────────────────────────────
def is_biologically_plausible(word: str) -> bool:
    word = word.strip()
    if not word or len(word) < 2:
        return False
    if not word.replace('-', '').replace('_', '').isalpha():
        return False
    blacklisted = {'the', 'and', 'of', 'in', 'to', 'for', 'with', 'that', 'this'}
    if word.lower() in blacklisted:
        return False
    return True

# ─────────────────────────────────────────────
# 2. Entraînement de FastText
# ─────────────────────────────────────────────
def train_domain_fasttext(sentences, model_path=None, vector_size=200, min_count=1):
    tokenized = []
    for sent in sentences:
        if isinstance(sent, list):  # JNLPBA
            tokens = [tok[0].lower() for tok in sent if isinstance(tok, tuple)]
        elif isinstance(sent, tuple) and len(sent) == 2:  # NCBI
            tokens = [tok.lower() for tok in sent[0]]
        else:
            continue
        if tokens:
            tokenized.append(tokens)

    print(f"Entraînement de FastText sur {len(tokenized)} phrases...")
    ft_model = FastText(
        sentences=tokenized,
        vector_size=vector_size,
        window=5,
        min_count=min_count,
        workers=4,
        sg=1,
        epochs=15,
        seed=42
    )
    if model_path:
        ft_model.save(model_path)
        print(f"Modèle sauvegardé dans {model_path}")
    return ft_model

# ─────────────────────────────────────────────
# 3. Comptage des entités
# ─────────────────────────────────────────────
def get_entity_counts(sentences):
    counter = Counter()
    for sent in sentences:
        if isinstance(sent, list):
            for _, label in sent:
                if label != "O":
                    counter[label] += 1
        elif isinstance(sent, tuple) and len(sent) == 2:
            _, labels = sent
            for label in labels:
                if label != "O":
                    counter[label] += 1
    return counter

# ─────────────────────────────────────────────
# 4. Augmentation d'une phrase
# ─────────────────────────────────────────────
def augment_sentence(sentence, fasttext_model, replace_prob=0.25, top_k=10, use_filter=True):
    # POUR JNLPBA :
    if isinstance(sentence, list):  
        new_sent = []
        for item in sentence:
            if len(item) < 2:
                new_sent.append(item)
                continue
            word, tag = item[0], item[1]
            new_word = word
            if (
                tag == "O" and
                isinstance(word, str) and
                word.lower() in fasttext_model.wv and
                random.random() < replace_prob
            ):
                try:
                    similar = fasttext_model.wv.most_similar(word.lower(), topn=top_k)
                    candidates = [
                        w for w, _ in similar
                        if (not use_filter) or is_biologically_plausible(w)
                    ]
                    if candidates:
                        new_word = random.choice(candidates)
                except Exception:
                    pass
            new_sent.append((new_word,) + tuple(item[1:]))
        return new_sent

    # POUR NCBI :
    elif isinstance(sentence, tuple) and len(sentence) == 2:  
        tokens, labels = sentence
        new_tokens = []
        for word, tag in zip(tokens, labels):
            new_word = word
            if (
                tag == "O" and
                isinstance(word, str) and
                word.lower() in fasttext_model.wv and
                random.random() < replace_prob
            ):
                try:
                    similar = fasttext_model.wv.most_similar(word.lower(), topn=top_k)
                    candidates = [
                        w for w, _ in similar
                        if (not use_filter) or is_biologically_plausible(w)
                    ]
                    if candidates:
                        new_word = random.choice(candidates)
                except Exception:
                    pass
            new_tokens.append(new_word)
        return (new_tokens, labels)

    else:
        return sentence

# ─────────────────────────────────────────────
# 5. Équilibrage INTELLIGENT (priorité à l'ARN)
# ─────────────────────────────────────────────
def balance_ner_dataset(
    sentences,
    fasttext_model,
    target_ratio=0.8,
    max_aug_per_sentence=3,
    replace_prob=0.25,
    top_k=10,
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)

    counts = get_entity_counts(sentences)
    if not counts:
        print(" Aucune entité trouvée. Retour du dataset original.")
        return sentences.copy()

    print(f" Comptage avant augmentation :\n{dict(counts)}")

    # --- Calcul des cibles ---
    max_count = max(counts.values())
    median_count = int(np.median(list(counts.values())))
    target_counts = {}
    for ent, cnt in counts.items():
        if ent in {"B-RNA", "I-RNA"}:
            target_counts[ent] = min(int(median_count * 2.0), int(max_count * 1.0))
        elif cnt < median_count:
            target_counts[ent] = min(int(median_count * 1.2), int(max_count * target_ratio))
        else:
            target_counts[ent] = cnt

    print(f" Cibles (ARN boosté) :\n{target_counts}")

    # --- Initialisation ---
    deficit = {ent: max(0, target_counts[ent] - counts[ent]) for ent in counts}
    augmented_dataset = sentences.copy()
    aug_count_per_orig = defaultdict(int)

    # --- Séparation : phrases avec ARN vs autres ---
    rna_sentences = []
    other_rare_sentences = []

    for idx, sent in enumerate(sentences):
        entities = []
        if isinstance(sent, list):
            entities = [label for _, label in sent if label != "O"]
        elif isinstance(sent, tuple) and len(sent) == 2:
            _, labels = sent
            entities = [label for label in labels if label != "O"]
        else:
            continue

        rare_entities_here = [e for e in entities if deficit.get(e, 0) > 0]
        if rare_entities_here:
            if any(e in {"B-RNA", "I-RNA"} for e in rare_entities_here):
                rna_sentences.append((idx, sent, rare_entities_here))
            else:
                other_rare_sentences.append((idx, sent, rare_entities_here))

    random.shuffle(rna_sentences)
    random.shuffle(other_rare_sentences)

    print(f" Trouvé {len(rna_sentences)} phrases contenant de l'ARN (B-RNA/I-RNA)")

    # --- ÉTAPE 1 : Augmenter AGRESSIVEMENT les phrases ARN en priorité ---
    for idx, sent, rare_entities_here in rna_sentences:
        # Augmenter tant que le déficit ARN est élevé OU limite de sécurité atteinte
        while (
            (deficit.get("B-RNA", 0) > 100 or deficit.get("I-RNA", 0) > 200)
            and aug_count_per_orig[idx] < 20
        ):
            # Plus de diversité pour l'ARN
            aug_sent = augment_sentence(
                sent,
                fasttext_model,
                replace_prob=0.4,
                top_k=15,
                use_filter=True
            )
            augmented_dataset.append(aug_sent)
            aug_count_per_orig[idx] += 1

            # Décrémenter les entités ARN
            if "B-RNA" in rare_entities_here:
                deficit["B-RNA"] = max(0, deficit["B-RNA"] - 1)
            if "I-RNA" in rare_entities_here:
                deficit["I-RNA"] = max(0, deficit["I-RNA"] - 1)

            # Crédit partiel pour les autres entités rares co-occurrentes
            for e in rare_entities_here:
                if e not in {"B-RNA", "I-RNA"}:
                    deficit[e] = max(0, deficit[e] - 0.2)

    print(f"Augmentation ARN terminée. Déficit restant : B-RNA={deficit.get('B-RNA',0)}, I-RNA={deficit.get('I-RNA',0)}")

    # --- ÉTAPE 2 : Augmenter les autres phrases rares ---
    for idx, sent, rare_entities_here in other_rare_sentences:
        if all(d <= 0 for d in deficit.values()):
            break

        if aug_count_per_orig[idx] >= max_aug_per_sentence:
            continue

        sponsor = max(rare_entities_here, key=lambda e: deficit[e])
        aug_sent = augment_sentence(
            sent,
            fasttext_model,
            replace_prob=replace_prob,
            top_k=top_k,
            use_filter=True
        )
        augmented_dataset.append(aug_sent)
        aug_count_per_orig[idx] += 1
        deficit[sponsor] -= 1

        for e in rare_entities_here:
            if e != sponsor:
                deficit[e] = max(0, deficit[e] - 0.3)

    # --- Sous-échantillonnage léger des classes sur-représentées ---
    final_counts = get_entity_counts(augmented_dataset)
    overrepresented = {
        e for e, c in final_counts.items()
        if c > target_counts.get(e, c) * 1.5
    }

    if overrepresented:
        print(f"⚠️ Entités sur-représentées ({overrepresented}), sous-échantillonnage léger...")
        downsampled = []
        for sent in augmented_dataset:
            if isinstance(sent, list):
                sent_entities = [label for _, label in sent if label != "O"]
            elif isinstance(sent, tuple) and len(sent) == 2:
                _, labels = sent
                sent_entities = [label for label in labels if label != "O"]
            else:
                sent_entities = []

            if sent_entities and all(e in overrepresented for e in sent_entities):
                if random.random() < 0.6:
                    continue
            downsampled.append(sent)
        augmented_dataset = downsampled

    # --- Statistiques finales ---
    final_counts = get_entity_counts(augmented_dataset)
    print(f" Comptage final après équilibrage :\n{dict(final_counts)}")
    print(f" Phrases totales : {len(sentences)} → {len(augmented_dataset)} (+{len(augmented_dataset)-len(sentences)})")

    return augmented_dataset

# ─────────────────────────────────────────────
# 6. Partitionnement et équilibrage
# ─────────────────────────────────────────────
def split_and_balance(sentences, fasttext_model, target_ratio=0.8, seed=42):
    random.seed(seed)
    shuffled = sentences.copy()
    random.shuffle(shuffled)

    total = len(shuffled)
    train_size = int(0.7 * total)
    dev_size = int(0.15 * total)

    train = shuffled[:train_size]
    dev = shuffled[train_size:train_size + dev_size]
    test = shuffled[train_size + dev_size:]

    print(f"\n Équilibrage de la partition TRAIN ({len(train)} phrases)...")
    train_balanced = balance_ner_dataset(
        train,
        fasttext_model,
        target_ratio=target_ratio,
        max_aug_per_sentence=3,
        replace_prob=0.25,
        top_k=10,
        seed=seed
    )
    return train_balanced, dev, test

# ─────────────────────────────────────────────
# 7. Visualisation
# ─────────────────────────────────────────────
def plot_class_distribution(title, original_counts, augmented_counts, save_path=None):
    labels = sorted(set(original_counts.keys()) | set(augmented_counts.keys()))
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, [original_counts.get(l, 0) for l in labels], width, label='Original', color='#1f77b4')
    bars2 = ax.bar(x + width/2, [augmented_counts.get(l, 0) for l in labels], width, label='Augmenté', color='#ff7f0e')

    ax.set_xlabel('Type d\'entité')
    ax.set_ylabel('Effectifs')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Annotation des barres
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{int(h)}', xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()