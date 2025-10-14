import open_clip
import os
import json
import numpy as np
import requests
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from PIL import Image
import pillow_avif
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import hashlib
import time

class ImageRetriever:
    def __init__(self, model_name, checkpoint, device='cpu'):
        self.clip_embd = OpenCLIPEmbeddings(model_name=model_name, checkpoint=checkpoint, device=device)
        self.embeddings = []
        self.image_paths = []

    def embed_image(self, image_path):
        img_feat = np.array(self.clip_embd.embed_image([image_path]))
        self.embeddings.append(img_feat)
        self.image_paths.append(image_path)
        return img_feat
    
    def embed_image_direct(self, image_path, embedding):
        self.embeddings.append(embedding)
        self.image_paths.append(image_path)
        return embedding

    def retrieve_similarity(self, query_image_path):
        query_embedding = np.array(self.clip_embd.embed_image([query_image_path])).astype('float32')
        similarities = [np.dot(query_embedding, img_emb.T).item() for img_emb in self.embeddings]
        return similarities

class TextRetriever:
    def __init__(self, model_name, device='cpu'):
        self.embeddings_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': device}, encode_kwargs={'normalize_embeddings': True})
        self.embeddings = []
        self.texts = []

    def embed_text(self, text):
        text_feat = np.array(self.embeddings_model.embed_query(text))
        self.embeddings.append(text_feat)
        self.texts.append(text)
        return text_feat
    
    def embed_text_direct(self, text, embedding):
        self.embeddings.append(embedding)
        self.texts.append(text)
        return embedding

    def retrieve_similarity(self, query_text):
        query_embedding = np.array(self.embeddings_model.embed_query(query_text)).astype('float32')
        similarities = [np.dot(query_embedding, txt_emb.T).item() for txt_emb in self.embeddings]
        return similarities

class EnsembleRetriever:
    def __init__(self, image_model_name="ViT-B-32", image_checkpoint="laion2b_s34b_b79k",
                 text_model_name="BAAI/bge-large-en", ensemble_ratio=0.5, cache_folder="cache", device='cpu'):
        self.image_retriever = ImageRetriever(model_name=image_model_name, checkpoint=image_checkpoint, device=device)
        self.text_retriever = TextRetriever(model_name=text_model_name, device=device)
        self.ensemble_ratio = ensemble_ratio
        self.embeddings = []  # Store tuples of (image_embedding, text_embedding, image_path, text, metadata)
        self.cache_folder = cache_folder

        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

    def _get_cache_key(self, image_path, text, metadata):
        key = json.dumps({"image_path": image_path, "text": text, "metadata": metadata}, sort_keys=True)
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return os.path.join(self.cache_folder, f"{key_hash}.json")

    def embed_pair(self, image_path, text, metadata=None):
        cache_key = self._get_cache_key(image_path, text, metadata)

        # Check if the embedding exists in cache
        if os.path.exists(cache_key):
            # print(f"cache hit for {cache_key}")
            with open(cache_key, "r") as f:
                cached_data = json.load(f)
                image_embedding = np.array(cached_data["image_embedding"], dtype=np.float32)
                self.image_retriever.embed_image_direct(image_path, image_embedding)
                text_embedding = np.array(cached_data["text_embedding"], dtype=np.float32)
                self.text_retriever.embed_text_direct(text, text_embedding)
                self.embeddings.append((image_embedding, text_embedding, image_path, text, metadata))
                return
        # print(f"cache miss for {cache_key}")
        # Compute embeddings if not cached
        image_embedding = self.image_retriever.embed_image(image_path).reshape(-1)
        text_embedding = self.text_retriever.embed_text(text).reshape(-1)
        self.embeddings.append((image_embedding, text_embedding, image_path, text, metadata))

        # Save embeddings to cache
        with open(cache_key, "w") as f:
            json.dump({
                "image_embedding": image_embedding.tolist(),
                "text_embedding": text_embedding.tolist()
            }, f)

    def retrieve_top_k_with_similarity(self, query_image_path, query_text, k=1):
        if not self.embeddings:
            raise ValueError("No pairs indexed. Please embed image-text pairs first.")

        # Retrieve individual similarities
        image_similarities = self.image_retriever.retrieve_similarity(query_image_path)
        text_similarities = self.text_retriever.retrieve_similarity(query_text)

        # Combine similarities using weighted sum
        combined_scores = [
            (self.ensemble_ratio * img_sim + (1 - self.ensemble_ratio) * txt_sim, pair[2], pair[3], pair[4])
            for img_sim, txt_sim, pair in zip(image_similarities, text_similarities, self.embeddings)
        ]

        # Sort by combined scores
        combined_scores.sort(reverse=True, key=lambda x: x[0])

        # Return top-k results
        return combined_scores[:k]

