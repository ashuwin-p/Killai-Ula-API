import os
import pandas as pd
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

# Relative imports for the package structure
from .bm25l import RobustBM25L
from .dense_retriever import DenseRetriever
from .tourism_graph import TourismGraph

class AdvancedTourismEngine:
    def __init__(self, file_path: str, indices_dir: str, db_path: str):
        """
        Args:
            file_path: Path to .csv/.xlsx data
            indices_dir: Path to folder containing bm25.pkl, etc.
            db_path: Path to ChromaDB folder
        """
        self.df = self._load_data(file_path)
        self.dense = DenseRetriever(db_path=db_path)
        self.graph = TourismGraph()
        self.corpus, self.doc_ids = self._build_corpus()

        # Load Indices
        print(f"[Engine] Loading indices from {indices_dir}...")
        try:
            self.bm25 = RobustBM25L.load(os.path.join(indices_dir, "bm25.pkl"))
            self.graph.load(os.path.join(indices_dir, "tourism_graph.json"))
            # Dense is loaded by init logic of Chroma, but we ensure connection
            print("[Engine] Indices loaded successfully.")
        except Exception as e:
            print(f"[Engine] Error loading indices: {e}. Ensure 'tourism_indices' folder is uploaded.")
            # Fallback or re-build logic could go here if you wanted write-access

        self.districts, self.categories, self.themes = self._extract_metadata()
        
        # Optional Reranker (CPU friendly model)
        try:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except:
            self.reranker = None

    def _load_data(self, file_path):
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        return pd.read_excel(file_path)

    def _build_corpus(self):
        # (Same logic as before, just kept concise here)
        corpus, doc_ids = [], []
        cols = ['location_name', 'district', 'primary_category', 'description', 'summary'] # + others
        for idx, row in self.df.iterrows():
            parts = [str(row[c]) for c in cols if c in self.df.columns and pd.notna(row[c])]
            corpus.append(" ".join(parts))
            doc_ids.append(str(idx))
        return corpus, doc_ids

    def _extract_metadata(self):
        # simplified for brevity, assume columns exist
        districts = set(self.df["district"].dropna().astype(str).str.lower()) if "district" in self.df else set()
        return districts, set(), set()

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # Fetch larger pool for reranking
        pool_size = max(10, k * 2)
        
        # 1. Get Candidates
        candidates = self._get_candidates(query, pool_size)
        
        # 2. Score & Fuse
        scored_ids = self._score_and_fuse(candidates, query, pool_size)
        
        # 3. Rerank
        if self.reranker:
            pairs = [[query, self.corpus[int(uid)]] for uid in scored_ids]
            rerank_scores = self.reranker.predict(pairs)
            scored_ids = [x for _, x in sorted(zip(rerank_scores, scored_ids), reverse=True)]

        # 4. Final K and Context
        final_ids = scored_ids[:k]
        return self._add_graph_context(final_ids)

    def _get_candidates(self, query, k):
        vec_res = self.dense.query(query, k)
        bm25_scores = self.bm25.get_scores(query)
        bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
        return {"vector_ids": vec_res.get("ids", [[]])[0], "bm25_indices": bm25_indices}

    def _score_and_fuse(self, candidates, query, k):
        scores = {}
        # RRF (Reciprocal Rank Fusion)
        for rank, uid in enumerate(candidates["vector_ids"]):
            scores[uid] = scores.get(uid, 0) + 1.0 / (1 + rank)
        for rank, idx in enumerate(candidates["bm25_indices"]):
            uid = self.doc_ids[idx]
            scores[uid] = scores.get(uid, 0) + 0.5 / (1 + rank)
        return sorted(scores, key=scores.get, reverse=True)[:k]

    def _add_graph_context(self, uids):
        results = []
        for uid in uids:
            row = self.df.iloc[int(uid)]
            # Get graph neighbors
            neighs = self.graph.neighbors(uid, limit=2)
            neigh_names = [self.df.iloc[int(n)]["location_name"] for n in neighs]
            
            content = (
                f"LOCATION: {row.get('location_name', 'N/A')}\n"
                f"DISTRICT: {row.get('district', 'N/A')}\n"
                f"DESC: {row.get('description', 'N/A')}\n"
                f"NEARBY: {', '.join(neigh_names)}"
            )
            results.append({"name": row.get('location_name'), "content": content})
        return results