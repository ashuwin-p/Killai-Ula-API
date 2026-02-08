"""
Optimized Graph Builder.
Fixes brittle CSV parsing for 'nearby_destinations'.
"""

import json
import networkx as nx
import pandas as pd
import ast

class TourismGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def _safe_parse_list(self, value):
        """
        Robustly parses a string that might be a list '["A", "B"]' 
        or a comma-separated string "A, B, C".
        """
        if pd.isna(value) or value == "":
            return []
        
        value = str(value).strip()
        
        # Try evaluating as python list literal
        try:
             if value.startswith("[") and value.endswith("]"):
                 return ast.literal_eval(value)
        except:
            pass
            
        # Fallback: Simple comma separated splitting
        # Clean up quotes if present (e.g. "'Chennai', 'Madurai'")
        cleaned = [x.strip().strip("'").strip('"') for x in value.split(',')]
        return [c for c in cleaned if c]

    def build(self, df):
        self.graph = nx.Graph()
        
        # Create a lookup map for case-insensitive name matching
        # Name -> [List of Indices]
        name_map = {}
        for idx, row in df.iterrows():
            name_lower = str(row["location_name"]).lower().strip()
            if name_lower not in name_map:
                name_map[name_lower] = []
            name_map[name_lower].append(str(idx))

        for idx, row in df.iterrows():
            node_id = str(idx)
            self.graph.add_node(node_id, name=row["location_name"])

            dests = self._safe_parse_list(row.get("nearby_destinations", ""))
            
            for dest in dests:
                dest_lower = dest.lower().strip()
                if dest_lower in name_map:
                    # Connect to all nodes with that name
                    for target_id in name_map[dest_lower]:
                        if node_id != target_id:
                            self.graph.add_edge(node_id, target_id)

    def neighbors(self, node_id, limit=3):
        if node_id not in self.graph:
            return []
        return list(self.graph.neighbors(node_id))[:limit]

    def save(self, path: str = "./tourism_graph.json"):
        data = {
            "nodes": {n: self.graph.nodes[n] for n in self.graph.nodes},
            "edges": list(self.graph.edges())
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str = "./tourism_graph.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        g = nx.Graph()
        for n, attr in data.get("nodes", {}).items():
            g.add_node(n, **attr)
        for u, v in data.get("edges", []):
            g.add_edge(u, v)
        self.graph = g