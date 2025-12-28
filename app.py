# streamlit_app.py
import streamlit as st
import torch
import pickle
import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import plotly.express as px
import plotly.graph_objects as go
import time
import json

# Import de votre modèle (ajustez le chemin selon votre structure)
import sys
sys.path.append('..')  # Pour importer depuis le dossier parent
from streamlit_utils import load_all_components

from models.models import CombinatorialNER  # Ajustez selon votre structure

# ============================================
# CONFIGURATION
# ============================================

st.set_page_config(
    page_title="BioNER - Biomedical NER",
    page_icon="🧬",
    layout="wide"
)

# ============================================
# CSS STYLING
# ============================================

st.markdown("""
<style>
    .main-header {
        color: #1E90FF;
        text-align: center;
        padding: 20px;
    }
    .entity-badge {
        display: inline-block;
        padding: 2px 8px;
        margin: 1px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.9em;
    }
    .results-box {
        background-color: gray;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #1E90FF;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# COULEURS DES ENTITÉS
# ============================================

ENTITY_COLORS = {
    'B-DNA': '#FF6B6B', 'I-DNA': '#FF8E8E',
    'B-RNA': '#4ECDC4', 'I-RNA': '#7FDFD9',
    'B-protein': '#45B7D1', 'I-protein': '#7ACFE5',
    'B-cell_type': '#96CEB4', 'I-cell_type': '#B8E0CD',
    'B-cell_line': "#6D664F", 'I-cell_line': "#C39A12",
    'O': 'transparent'
}

ENTITY_NAMES = {
    'B-DNA': 'DNA', 'I-DNA': 'DNA',
    'B-RNA': 'RNA', 'I-RNA': 'RNA',
    'B-protein': 'Protein', 'I-protein': 'Protein',
    'B-cell_type': 'Cell Type', 'I-cell_type': 'Cell Type',
    'B-cell_line': 'Cell Line', 'I-cell_line': 'Cell Line',
    'O': 'Other'
}

# ============================================
# CLASSES UTILITAIRES
# ============================================

class StreamlitNERPredictor:
    def __init__(self, components: Dict):
        """Initialise le prédicteur avec tous les composants chargés"""
        self.vocab = components['vocab']
        self.char_vocab = components['char_vocab']
        self.tag_to_idx = components['tag_to_idx']
        self.idx_to_tag = components['idx_to_tag']
        self.pretrained_embeddings = components['pretrained_embeddings']
        self.checkpoint = components['checkpoint']
        self.device = components['device']
        
        # Détecter le dataset (JNLPBA ou NCBI)
        checkpoint_path = components.get('checkpoint_path', '')
        if 'jnlpba' in checkpoint_path.lower():
            self.dataset_name = 'JNLPBA'
            lstm_hidden_dim = 256
        else:
            self.dataset_name = 'NCBI'
            lstm_hidden_dim = 128
        
        # IMPORTANT: Récupérer les paramètres exacts du checkpoint
        checkpoint = self.checkpoint
        epoch = checkpoint.get('epoch', 0)
        best_f1 = checkpoint.get('best_f1', 0.0)
        
        print(f"📦 Checkpoint chargé: epoch {epoch}, best_f1 {best_f1:.4f}")
        
        # Créer le modèle avec les mêmes paramètres qu'à l'entraînement
        self.model = CombinatorialNER(
            vocab_size=len(self.vocab),
            char_vocab_size=len(self.char_vocab),
            tag_to_idx=self.tag_to_idx,
            dataset=self.dataset_name,
            use_char_cnn=True,      # Récupérer du checkpoint si possible
            use_char_lstm=True,     # Récupérer du checkpoint si possible
            use_attention=True,     # Récupérer du checkpoint si possible
            use_fc_fusion=False,    # Récupérer du checkpoint si possible
            pretrained_embeddings=self.pretrained_embeddings,
            word_embed_dim=200,
            lstm_hidden_dim=lstm_hidden_dim,
            dropout=0.5,
            use_lstm=True
        ).to(self.device)
        
        # Charger les poids
        try:
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            print("✅ Poids du modèle chargés avec succès")
            
            # Vérifier les paramètres chargés
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"📊 Paramètres totaux: {total_params:,}")
            
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement: {e}")
            # Essayer de charger seulement les couches correspondantes
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() 
                             if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict, strict=False)
            print(f"✅ Chargement partiel réussi: {len(pretrained_dict)}/{len(checkpoint)} paramètres")
        
        self.model.eval()
        print(f"✅ Modèle {self.dataset_name} prêt sur {self.device}")
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenisation simple du texte"""
        # Tokenisation adaptée au texte biomédical
        tokens = re.findall(r'\b\w+(?:-\w+)*\b|[^\w\s]', text)
        return tokens
    
    def preprocess_tokens(self, tokens: List[str], max_seq_len: int = 100, max_char_len: int = 20):
        """Préparation des tokens pour le modèle"""
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
        
        seq_len = len(tokens)
        
        # IDs des mots
        word_ids = []
        for token in tokens:
            if token.isdigit():
                token_id = self.vocab.get('<NUM>', self.vocab.get('<UNK>', 1))
            else:
                token_lower = token.lower()
                token_id = self.vocab.get(token_lower, self.vocab.get('<UNK>', 1))
            word_ids.append(token_id)
        
        # Padding pour les mots
        pad_word = self.vocab.get('<PAD>', 0)
        word_ids += [pad_word] * (max_seq_len - seq_len)
        
        # Séquences de caractères
        char_seqs = []
        unk_char = self.char_vocab.get('<UNK>', 1)
        pad_char = self.char_vocab.get('<PAD>', 0)
        
        for token in tokens:
            chars = [self.char_vocab.get(c, unk_char) for c in token[:max_char_len]]
            chars += [pad_char] * (max_char_len - len(chars))
            char_seqs.append(chars)
        
        # Padding pour les caractères
        char_seqs += [[pad_char] * max_char_len] * (max_seq_len - seq_len)
        
        return tokens, word_ids, char_seqs, seq_len
    
    def predict(self, text: str):
        """Prédiction principale - CORRIGÉ POUR CRF"""
        # Tokenisation
        tokens = self.tokenize_text(text)
        
        if not tokens:
            return []
        
        # Préparation
        tokens, word_ids, char_seqs, seq_len = self.preprocess_tokens(tokens)
        
        # Conversion en tensors
        word_tensor = torch.tensor([word_ids], dtype=torch.long).to(self.device)
        char_tensor = torch.tensor([char_seqs], dtype=torch.long).to(self.device)
        
        # Créer le masque (True pour les tokens réels, False pour padding)
        mask = torch.ones((1, 100), dtype=torch.bool).to(self.device)  # max_seq_len = 100
        mask[:, seq_len:] = False  # Masquer le padding
        
        # Prédiction avec CRF
        with torch.no_grad():
            try:
                # Utiliser la méthode forward du modèle qui retourne les chemins CRF décodés
                predictions = self.model(word_tensor, char_tensor, mask=mask)
                
                # Le modèle retourne une liste de listes (chemins CRF)
                if isinstance(predictions, list) and len(predictions) > 0:
                    predicted_ids = predictions[0][:seq_len]
                else:
                    # Fallback: utiliser l'émission seule si CRF échoue
                    print("⚠️ Utilisation du fallback (sans CRF)")
                    emissions = self.get_emissions(word_tensor, char_tensor, mask)
                    predicted_ids = torch.argmax(emissions, dim=2)[0][:seq_len].cpu().numpy()
            
            except Exception as e:
                print(f"⚠️ Erreur CRF: {e}, utilisation du fallback")
                # Fallback: prédiction sans CRF
                emissions = self.get_emissions(word_tensor, char_tensor, mask)
                predicted_ids = torch.argmax(emissions, dim=2)[0][:seq_len].cpu().numpy()
        
        # Conversion en tags
        pred_tags = [self.idx_to_tag.get(idx, 'O') for idx in predicted_ids]
        
        return list(zip(tokens, pred_tags))
    
    def get_emissions(self, word_tensor, char_tensor, mask):
        """Récupère les émissions brutes (sans CRF) pour le fallback"""
        # Refaire un forward pass manuel pour obtenir les émissions
        word_emb = self.model.word_embedding(word_tensor)
        
        char_embs = []
        if hasattr(self.model, 'use_char_cnn') and self.model.use_char_cnn:
            char_embs.append(self.model.char_cnn(char_tensor))
        if hasattr(self.model, 'use_char_lstm') and self.model.use_char_lstm:
            char_embs.append(self.model.char_lstm(char_tensor))
        
        if char_embs:
            combined = torch.cat([word_emb] + char_embs, dim=-1)
        else:
            combined = word_emb
        
        if hasattr(self.model, 'use_fc_fusion') and self.model.use_fc_fusion:
            combined = self.model.fusion(combined)
        
        if hasattr(self.model, 'context_lstm') and self.model.context_lstm is not None:
            lstm_out, _ = self.model.context_lstm(combined)
            if hasattr(self.model, 'attention_layer') and self.model.attention_layer is not None:
                lstm_out = self.model.attention_layer(lstm_out, mask)
        else:
            lstm_out = combined
        
        emissions = self.model.emission(lstm_out)
        
        return emissions
    
    def extract_entities(self, predictions: List[Tuple[str, str]]):
        """Extraction des entités des prédictions"""
        entities = []
        current_entity = None
        entity_tokens = []
        entity_type = None
        
        for token, tag in predictions:
            if tag.startswith('B-'):
                # Sauvegarder l'entité précédente
                if current_entity:
                    entities.append({
                        'text': ' '.join(entity_tokens),
                        'type': entity_type[2:],
                        'tag': entity_type,
                        'tokens': entity_tokens.copy()
                    })
                
                # Nouvelle entité
                current_entity = tag[2:]
                entity_type = tag
                entity_tokens = [token]
                
            elif tag.startswith('I-'):
                if current_entity == tag[2:]:
                    entity_tokens.append(token)
                else:
                    # I- sans B- précédent
                    if current_entity:
                        entities.append({
                            'text': ' '.join(entity_tokens),
                            'type': entity_type[2:],
                            'tag': entity_type,
                            'tokens': entity_tokens.copy()
                        })
                    
                    current_entity = tag[2:]
                    entity_type = 'B-' + tag[2:]  # Convertir en B-
                    entity_tokens = [token]
            
            else:  # 'O'
                if current_entity:
                    entities.append({
                        'text': ' '.join(entity_tokens),
                        'type': entity_type[2:],
                        'tag': entity_type,
                        'tokens': entity_tokens.copy()
                    })
                    current_entity = None
                    entity_tokens = []
        
        # Dernière entité
        if current_entity:
            entities.append({
                'text': ' '.join(entity_tokens),
                'type': entity_type[2:],
                'tag': entity_type,
                'tokens': entity_tokens.copy()
            })
        
        return entities
# ============================================
# FONCTIONS UTILITAIRES
# ============================================

@st.cache_resource
def load_all_for_streamlit():
    """Charge tous les composants pour Streamlit (cached)"""
    try:
        # Chemins (à adapter)
        model_path = "./checkpoints/JNLPBA/WE_char_bilstm_cnn_attention/best_model.pt"
        vocab_dir = "./vocab/jnlpba"
        word2vec_path = "./word2Vecembeddings/jnlpba_word2vec"
        
        # Vérifier les fichiers
        if not os.path.exists(model_path):
            st.error(f"❌ Modèle non trouvé: {model_path}")
            return None
        
        if not os.path.exists(vocab_dir):
            st.error(f"❌ Vocabulaire non trouvé: {vocab_dir}")
            return None
        
        # Charger les composants
        components = load_all_components(model_path, vocab_dir, word2vec_path)
        components['checkpoint_path'] = model_path
        
        # Créer le prédicteur
        predictor = StreamlitNERPredictor(components)
        
        return predictor
        
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def highlight_text(text: str, predictions: List[Tuple[str, str]]):
    """Surligne le texte avec les entités"""
    highlighted = ""
    for token, tag in predictions:
        if tag != 'O':
            color = ENTITY_COLORS.get(tag, '#CCCCCC')
            entity_name = ENTITY_NAMES.get(tag, tag[2:])
            highlighted += f'<span class="entity-badge" style="background-color: {color};" title="{entity_name}">{token}</span> '
        else:
            highlighted += f'{token} '
    
    return highlighted

def create_entity_legend():
    """Crée la légende des entités"""
    st.markdown("### 🎨 Types d'Entités")
    
    cols = st.columns(4)
    entity_items = []
    
    for tag, color in ENTITY_COLORS.items():
        if tag != 'O' and tag.startswith('B-'):
            entity_name = ENTITY_NAMES.get(tag, tag[2:])
            entity_items.append((entity_name, color))
    
    items_per_col = len(entity_items) // 4 + 1
    
    for i, col in enumerate(cols):
        start_idx = i * items_per_col
        end_idx = min((i + 1) * items_per_col, len(entity_items))
        
        with col:
            for entity_name, color in entity_items[start_idx:end_idx]:
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <div style="width: 15px; height: 15px; background-color: {color}; margin-right: 8px; border-radius: 3px;"></div>
                    <span>{entity_name}</span>
                </div>
                """, unsafe_allow_html=True)

# ============================================
# APPLICATION STREAMLIT
# ============================================

def main():
    st.markdown('<h1 class="main-header">🧬 Biomedical Named Entity Recognition</h1>', unsafe_allow_html=True)
    st.markdown("Extract biomedical entities from text using deep learning")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        
        # Charger le modèle
        if 'predictor' not in st.session_state:
            with st.spinner("Chargement du modèle..."):
                predictor = load_all_for_streamlit()
                if predictor:
                    st.session_state.predictor = predictor
                    st.success("✅ Modèle chargé!")
                else:
                    st.error("❌ Échec du chargement")
                    return
        
        predictor = st.session_state.predictor
        
        st.markdown("---")
        st.markdown("### 📊 Informations")
        st.markdown(f"""
        - **Dataset:** {predictor.dataset_name}
        - **Vocabulaire:** {len(predictor.vocab)} mots
        - **Entités:** {len(predictor.tag_to_idx) - 1} types
        - **Device:** {predictor.device}
        """)
    
    # Légende des entités
    create_entity_legend()
    
    st.markdown("---")
    
    # Zone de texte
    st.markdown("### 📝 Entrez votre texte biomédical")
    
    # Exemples
    # Exemples
    examples = {
    "Génétique": (
        "Mutations in the TP53 gene are frequently observed in human cancers and lead to loss of p53 protein "
        "tumor suppressor activity. Overexpression of MDM2 results in increased degradation of p53, while "
        "alterations in BRCA1 and BRCA2 genes impair DNA double-strand break repair through homologous recombination. "
        "Recent studies also indicate that ATM and ATR kinases phosphorylate p53 in response to DNA damage."
    )
    }
    
    examples["Immunologie"] = (
    "Activation of T lymphocytes requires signaling through the T cell receptor complex and costimulatory "
    "molecules such as CD28. IL-2 gene transcription is regulated by NF-kappa B, AP-1, and NFAT transcription factors. "
    "Inhibition of JAK3 signaling suppresses STAT5 phosphorylation and reduces IL-2 mRNA expression in activated T cells."
    )
    
    examples["Cellulaire"] = (
    "HeLa cells and HEK293 cell lines are widely used to study transcriptional regulation and protein-protein interactions. "
    "Jurkat T cells exhibit strong activation of MAPK and ERK signaling pathways following stimulation with phorbol esters. "
    "Primary fibroblasts show increased expression of collagen genes during wound healing."
    )

    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🧬 Exemple Génétique", use_container_width=True):
            st.session_state.example_text = examples["Génétique"]
    with col2:
        if st.button("🩸 Exemple Immunologie", use_container_width=True):
            st.session_state.example_text = examples["Immunologie"]
    with col3:
        if st.button("🔬 Exemple Cellulaire", use_container_width=True):
            st.session_state.example_text = examples["Cellulaire"]
    
    # Zone de texte
    text_input = st.text_area(
        "**Texte à analyser:**",
        value=st.session_state.get('example_text', ''),
        height=200,
        placeholder="Collez votre texte biomédical ici..."
    )
    
    # Bouton de prédiction
    if st.button("🔍 Analyser le texte", type="primary", use_container_width=True):
        if not text_input.strip():
            st.error("❌ Veuillez entrer du texte.")
        else:
            with st.spinner("Analyse en cours..."):
                start_time = time.time()
                
                try:
                    # Prédiction
                    predictions = predictor.predict(text_input)
                    entities = predictor.extract_entities(predictions)
                    
                    processing_time = time.time() - start_time
                    
                    # Stocker les résultats
                    st.session_state.last_results = {
                        'predictions': predictions,
                        'entities': entities,
                        'text': text_input,
                        'processing_time': processing_time,
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    st.success(f"✅ {len(entities)} entités trouvées en {processing_time:.2f} secondes!")
                    
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
    
    # Afficher les résultats
    if 'last_results' in st.session_state:
        results = st.session_state.last_results
        
        st.markdown("---")
        st.markdown("### 📊 Résultats")
        
        # Métriques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Entités trouvées", len(results['entities']))
        with col2:
            st.metric("Temps d'analyse", f"{results['processing_time']:.2f}s")
        with col3:
            unique_types = len(set([e['type'] for e in results['entities']]))
            st.metric("Types d'entités", unique_types)
        
        # Onglets
        tab1, tab2, tab3 = st.tabs(["📄 Texte annoté", "📊 Liste des entités", "📈 Statistiques"])
        
        with tab1:
            st.markdown("#### Texte avec entités surlignées")
            highlighted = highlight_text(results['text'], results['predictions'])
            st.markdown(f'<div class="results-box">{highlighted}</div>', unsafe_allow_html=True)
        
        with tab2:
            if results['entities']:
                df_data = []
                for entity in results['entities']:
                    df_data.append({
                        'Entité': entity['text'],
                        'Type': ENTITY_NAMES.get(entity['tag'], entity['type']),
                        'Tag': entity['tag'],
                        'Tokens': len(entity['tokens'])
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ Aucune entité trouvée.")
        
        with tab3:
            if results['entities']:
                # Distribution par type
                type_counts = {}
                for entity in results['entities']:
                    entity_type = ENTITY_NAMES.get(entity['tag'], entity['type'])
                    type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
                
                # Graphique
                if type_counts:
                    fig = px.bar(
                        x=list(type_counts.keys()),
                        y=list(type_counts.values()),
                        title="Distribution des types d'entités",
                        labels={'x': 'Type', 'y': 'Nombre'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Longueur moyenne des entités
                avg_length = np.mean([len(e['tokens']) for e in results['entities']])
                st.metric("Longueur moyenne", f"{avg_length:.1f} tokens")
        
        # Export
        st.markdown("---")
        st.markdown("### 💾 Exporter les résultats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export JSON
            export_data = {
                'text': results['text'],
                'entities': results['entities'],
                'timestamp': results['timestamp'],
                'processing_time': results['processing_time']
            }
            
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="📥 Télécharger JSON",
                data=json_str,
                file_name="bio_ner_results.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Export CSV
            if results['entities']:
                df_data = []
                for entity in results['entities']:
                    df_data.append({
                        'entity': entity['text'],
                        'type': ENTITY_NAMES.get(entity['tag'], entity['type']),
                        'tag': entity['tag']
                    })
                
                df = pd.DataFrame(df_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Télécharger CSV",
                    data=csv,
                    file_name="bio_ner_entities.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ============================================
# SCRIPT PRINCIPAL
# ============================================

if __name__ == "__main__":
    main()