"""
    Author: Ashuwin P
    File Description:
    Optimized Retrieval Engine. Implements Triple Representation RAG:
    1. Dense (Vector)
    2. Sparse (BM25)
    3. Graph (Traversal & Connectivity)
    
    UPDATES: 
    - Implemented True Graph Retrieval (Query -> Node -> Neighbors -> Candidates).
    - Added Relevance Scores to LLM Context.
"""

import os
import pandas as pd
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Tuple

from rag_modules.bm25l import RobustBM25L
from rag_modules.dense_retriever import DenseRetriever
from rag_modules.tourism_graph import TourismGraph

class AdvancedTourismEngine:
    def __init__(self, file_path: str, indices_dir: str, db_path: str):
        print(f"[Engine] Initializing...")
        self.df = self._load_data(file_path)
        self.df = self.df.reset_index(drop=True)

        self.dense = DenseRetriever(db_path=db_path)
        self.graph = TourismGraph()
        
        # 1. Build Corpus & Name Map for Graph Retrieval
        self.corpus, self.doc_ids = self._build_corpus()
        self.name_map = self._build_name_map()

        # 2. Load Indices
        print(f"[Engine] Loading indices from {indices_dir}...")
        try:
            self.bm25 = RobustBM25L.load(os.path.join(indices_dir, "bm25.pkl"))
            self.graph.load(os.path.join(indices_dir, "tourism_graph.json"))
            print("[Engine] Indices loaded successfully.")
        except Exception as e:
            print(f"[Engine] Error loading indices: {e}. Attempting fallback...")
            self.bm25 = RobustBM25L(self.corpus)
            self.graph.build(self.df)

        # 3. Extract Metadata
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
        corpus, doc_ids = [], []
        # Full column utilization
        cols = [
            'location_name', 'district', 'primary_category', 'secondary_categories',
            'themes', 'uniqueness_factor', 'activities', 'target_audience',
            'historical_period', 'architectural_style', 'description', 'summary',
            'best_seasons', 'how_to_reach_by_bus', 'how_to_reach_by_air', 
            'how_to_reach_by_rail', 'accommodation'
        ]
        available_cols = [c for c in cols if c in self.df.columns]
        for idx, row in self.df.iterrows():
            parts = [str(row[c]) for c in available_cols if pd.notna(row.get(c))]
            corpus.append(" ".join(parts))
            doc_ids.append(str(idx))
        return corpus, doc_ids

    def _build_name_map(self) -> Dict[str, str]:
        """Maps lowercase location names to Doc IDs for Graph Entry."""
        name_map = {}
        for idx, row in self.df.iterrows():
            if pd.notna(row.get('location_name')):
                name_map[str(row['location_name']).lower().strip()] = str(idx)
        return name_map

    def _extract_metadata(self) -> tuple:
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

    # -------------------------------------------------------------------------
    # RETRIEVAL LOGIC
    # -------------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # Fetch larger pool for reranking
        pool_size = max(10, k * 2)
        
        # 1. Get Candidates (Dense + Sparse)
        candidates = self._get_candidates(query, pool_size)
        
        # 2. Score & Fuse (with Metadata + Graph Injection)
        # We define scored_ids as a Dict[str, float] here
        doc_scores = self._score_and_fuse(candidates, query, pool_size)
        
        # 3. Graph RAG Retrieval (Injection)
        # If query mentions "Madurai", explicitly fetch neighbors and boost them
        doc_scores = self._graph_retrieval_injection(query, doc_scores)
        
        # Sort by score
        sorted_candidates = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:pool_size]
        
        # 4. Rerank (if available)
        final_results_with_score = self._rerank_with_score(query, sorted_candidates)
        
        # 5. Final Slice & Context Formatting
        top_k = final_results_with_score[:k]
        return self._add_graph_context(top_k)

    def _get_candidates(self, query: str, k: int) -> Dict[str, Any]:
        vector_results = self.dense.query(query, k)
        
        bm25_indices = []
        if hasattr(self, 'bm25') and self.bm25:
            bm25_scores = self.bm25.get_scores(query)
            # Get top indices
            bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
            
        return {"vector_ids": vector_results.get("ids", [[]])[0], "bm25_indices": bm25_indices}

    def _score_and_fuse(self, candidates: Dict[str, Any], query: str, k: int) -> Dict[str, float]:
        doc_scores = {}
        # RRF (Reciprocal Rank Fusion)
        for rank, doc_id in enumerate(candidates["vector_ids"]):
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (1 + rank)
        
        for rank, idx in enumerate(candidates["bm25_indices"]):
            if idx < len(self.doc_ids):
                doc_id = self.doc_ids[idx]
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 0.5 / (1 + rank)

        # Apply metadata boosting
        doc_scores = self._apply_metadata_boost(doc_scores, query.lower())
        return doc_scores

    def _graph_retrieval_injection(self, query: str, doc_scores: Dict[str, float]) -> Dict[str, float]:
        """
        True Graph RAG:
        1. Identify nodes mentioned in the query.
        2. Traverse edges to find neighbors.
        3. Inject neighbors into the candidate pool or boost them if already present.
        """
        query_lower = query.lower()
        matched_ids = []
        
        # Simple entity matching (names are usually distinct enough in tourism)
        for name, doc_id in self.name_map.items():
            if name in query_lower:
                matched_ids.append(doc_id)
        
        if not matched_ids:
            return doc_scores

        # Traverse Graph
        for start_node in matched_ids:
            # Also boost the start node itself heavily (it's in the query!)
            doc_scores[start_node] = doc_scores.get(start_node, 0) + 2.0
            
            neighbors = self.graph.neighbors(start_node)
            for neighbor_id in neighbors:
                if neighbor_id in doc_scores:
                    # Boost existing candidate (Validation)
                    doc_scores[neighbor_id] += 1.0 
                else:
                    # Inject new candidate (Discovery)
                    # Give it a reasonable base score so it competes with retrieved docs
                    doc_scores[neighbor_id] = 0.8 
        
        return doc_scores

    def _apply_metadata_boost(self, doc_scores: Dict[str, float], query_lower: str) -> Dict[str, float]:
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

    def _rerank_with_score(self, query: str, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Returns list of (doc_id, score) sorted by score."""
        if not self.reranker or not candidates:
            return candidates # Already sorted from previous step
        
        # Filter invalid IDs
        valid_candidates = [c for c in candidates if int(c[0]) < len(self.corpus)]
        if not valid_candidates: return []

        pairs = [[query, self.corpus[int(doc_id)]] for doc_id, _ in valid_candidates]
        scores = self.reranker.predict(pairs)
        
        # Re-sort based on new cross-encoder scores
        reranked = sorted(zip(valid_candidates, scores), key=lambda x: x[1], reverse=True)
        
        # Format as [(doc_id, new_score), ...]
        return [(c[0], s) for c, s in reranked]

    def _add_graph_context(self, candidates_with_score: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        results = []
        for doc_id, score in candidates_with_score:
            try:
                idx = int(doc_id)
                if idx >= len(self.df): continue
                
                row = self.df.iloc[idx]
                neighbors = self.graph.neighbors(doc_id, limit=3)
                
                neighbor_names = []
                for n in neighbors:
                    if int(n) < len(self.df):
                        neighbor_names.append(self.df.iloc[int(n)]["location_name"])
                
                results.append({
                    "name": row["location_name"],
                    "content": self._format_result_content(row, neighbor_names, score)
                })
            except Exception as e:
                print(f"[Engine] Error formatting result {doc_id}: {e}")
                continue
        return results

    def _format_result_content(self, row: pd.Series, neighbor_names: List[str], score: float) -> str:
        """
        Includes Score, History, Logistics, and all details.
        """
        def val(key):
            v = row.get(key, '')
            return str(v).strip() if pd.notna(v) and str(v).strip() != '' else 'N/A'

        return (
            f"[RELEVANCE SCORE: {score:.2f}]\n"
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