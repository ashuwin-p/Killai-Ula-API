"""
    Author: Ashuwin P
    File Description:
    Optimized Retrieval Engine. Implements Hybrid Search (BM25 + Dense),
    Metadata Boosting, Cross-Encoder Reranking, and Graph Context.
"""

import os
import pandas as pd
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

from rag_modules.bm25l import RobustBM25L
from rag_modules.dense_retriever import DenseRetriever
from rag_modules.tourism_graph import TourismGraph

class AdvancedTourismEngine:
    def __init__(self, file_path: str, indices_dir: str, db_path: str):
        """
        Args:
            file_path: Path to .csv/.xlsx data
            indices_dir: Path to folder containing bm25.pkl, etc.
            db_path: Path to ChromaDB folder
        """
        print(f"[Engine] Initializing...")
        self.df = self._load_data(file_path)
        self.df = self.df.reset_index(drop=True)

        self.dense = DenseRetriever(db_path=db_path)
        self.graph = TourismGraph()
        
        # 1. Build Corpus using ALL columns
        self.corpus, self.doc_ids = self._build_corpus()

        # 2. Load Indices
        print(f"[Engine] Loading indices from {indices_dir}...")
        try:
            self.bm25 = RobustBM25L.load(os.path.join(indices_dir, "bm25.pkl"))
            self.graph.load(os.path.join(indices_dir, "tourism_graph.json"))
            print("[Engine] Indices loaded successfully.")
        except Exception as e:
            print(f"[Engine] Error loading indices: {e}. Attempting fallback...")
            # Fallback: Rebuild in memory if load fails
            self.bm25 = RobustBM25L(self.corpus)
            self.graph.build(self.df)

        # 3. Extract Metadata for Boosting
        self.districts, self.categories, self.themes = self._extract_metadata()
        
        # 4. Reranker
        try:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except:
            print("[Engine] Warning: Reranker failed to load. Running without it.")
            self.reranker = None

    def _load_data(self, file_path: str) -> pd.DataFrame:
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        return pd.read_excel(file_path)

    def _build_corpus(self) -> tuple:
        """
        Builds a search corpus using ALL semantic columns to ensure no data is missed.
        """
        corpus, doc_ids = [], []
        # ALL useful text columns
        cols = [
            'location_name', 'district', 'primary_category', 'secondary_categories',
            'themes', 'uniqueness_factor', 'activities', 'target_audience',
            'historical_period', 'architectural_style', 'description', 'summary',
            'best_seasons', 'how_to_reach_by_bus', 'how_to_reach_by_air', 
            'how_to_reach_by_rail', 'accommodation'
        ]
        
        # Safe check for columns existence
        available_cols = [c for c in cols if c in self.df.columns]

        for idx, row in self.df.iterrows():
            parts = [str(row[c]) for c in available_cols if pd.notna(row.get(c))]
            corpus.append(" ".join(parts))
            doc_ids.append(str(idx))
        return corpus, doc_ids

    def _extract_metadata(self) -> tuple:
        """
        Extracts unique metadata for query boosting.
        """
        districts = set()
        categories = set()
        themes = set()
        
        if "district" in self.df:
            districts = set(self.df["district"].dropna().astype(str).str.lower())
        
        if "primary_category" in self.df:
            categories = set(self.df["primary_category"].dropna().astype(str).str.lower())
            
        if "themes" in self.df:
            for t_str in self.df["themes"].dropna().astype(str):
                for t in t_str.split(','):
                    themes.add(t.strip().lower())
                    
        return districts, categories, themes

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # Fetch 2*k candidates for reranking
        pool_size = max(10, k * 2)
        
        # 1. Get Candidates
        candidates = self._get_candidates(query, pool_size)
        
        # 2. Score & Fuse (with Metadata Boost)
        scored_ids = self._score_and_fuse(candidates, query, pool_size)
        
        # 3. Rerank
        reranked_ids = self._rerank_if_available(query, scored_ids)
        
        # 4. Final K and Graph Context
        final_ids = reranked_ids[:k]
        return self._add_graph_context(final_ids)

    def _get_candidates(self, query: str, k: int) -> Dict[str, Any]:
        vector_results = self.dense.query(query, k)
        
        if hasattr(self, 'bm25') and self.bm25:
            bm25_scores = self.bm25.get_scores(query)
            bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
        else:
            bm25_indices = []
            
        return {"vector_ids": vector_results.get("ids", [[]])[0], "bm25_indices": bm25_indices}

    def _score_and_fuse(self, candidates: Dict[str, Any], query: str, k: int) -> List[str]:
        doc_scores = {}
        # RRF
        for rank, doc_id in enumerate(candidates["vector_ids"]):
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (1 + rank)
        
        for rank, idx in enumerate(candidates["bm25_indices"]):
            if idx < len(self.doc_ids):
                doc_id = self.doc_ids[idx]
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 0.5 / (1 + rank)

        # Apply metadata boosting logic
        doc_scores = self._apply_metadata_boost(doc_scores, query.lower())
        return sorted(doc_scores, key=doc_scores.get, reverse=True)[:k]

    def _apply_metadata_boost(self, doc_scores: Dict[str, float], query_lower: str) -> Dict[str, float]:
        """
        Boosts scores if query contains specific metadata keywords.
        """
        if not any(x in query_lower for x in self.districts | self.categories | self.themes):
            return doc_scores
            
        boosted = doc_scores.copy()
        for doc_id in boosted:
            try:
                row = self.df.iloc[int(doc_id)]
                if "district" in row and str(row["district"]).lower() in query_lower: 
                    boosted[doc_id] += 0.5
                if "primary_category" in row and str(row["primary_category"]).lower() in query_lower: 
                    boosted[doc_id] += 0.3
                
                if "themes" in row:
                    doc_themes = str(row.get("themes", "")).lower()
                    if any(t in doc_themes for t in self.themes if t in query_lower):
                        boosted[doc_id] += 0.4
            except:
                continue
        return boosted

    def _rerank_if_available(self, query: str, candidates: List[str]) -> List[str]:
        if not self.reranker or not candidates: return candidates
        
        # Filter out invalid IDs
        valid_candidates = [cid for cid in candidates if int(cid) < len(self.corpus)]
        if not valid_candidates: return []

        pairs = [[query, self.corpus[int(doc_id)]] for doc_id in valid_candidates]
        scores = self.reranker.predict(pairs)
        return [x for _, x in sorted(zip(scores, valid_candidates), reverse=True)]

    def _add_graph_context(self, candidate_ids: List[str]) -> List[Dict[str, Any]]:
        results = []
        for doc_id in candidate_ids:
            try:
                idx = int(doc_id)
                if idx >= len(self.df): continue
                
                row = self.df.iloc[idx]
                neighbors = self.graph.neighbors(doc_id, limit=3)
                
                # Safe neighbor name retrieval
                neighbor_names = []
                for n in neighbors:
                    if int(n) < len(self.df):
                        neighbor_names.append(self.df.iloc[int(n)]["location_name"])
                
                results.append({
                    "name": row["location_name"],
                    "content": self._format_result_content(row, neighbor_names)
                })
            except Exception as e:
                print(f"[Engine] Error formatting result {doc_id}: {e}")
                continue
        return results

    def _format_result_content(self, row: pd.Series, neighbor_names: List[str]) -> str:
        """
        Formats the context passed to the LLM.
        CRITICAL: Includes ALL columns (History, Logistics, etc.)
        """
        def val(key):
            v = row.get(key, '')
            return str(v).strip() if pd.notna(v) and str(v).strip() != '' else 'N/A'

        return (
            f"LOCATION: {row.get('location_name', 'N/A')} ({row.get('district', 'N/A')})\n"
            f"TYPE: {row.get('primary_category', 'N/A')} | {val('secondary_categories')}\n"
            f"THEMES: {val('themes')} | AUDIENCE: {val('target_audience')}\n"
            f"HIGHLIGHTS: {val('uniqueness_factor')}\n"
            f"ACTIVITIES: {val('activities')}\n"
            f"HISTORY & ARCH: {val('historical_period')} | {val('architectural_style')}\n"
            f"TIMING: {val('best_seasons')}\n"
            f"LOGISTICS: Air: {val('how_to_reach_by_air')} | Rail: {val('how_to_reach_by_rail')} | Bus: {val('how_to_reach_by_bus')}\n"
            f"STAY: {val('accommodation')}\n"
            f"DETAILS: {val('description')}\n"
            f"NEARBY: {', '.join(neighbor_names) if neighbor_names else 'None'}"
        )