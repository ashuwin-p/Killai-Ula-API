import time
from typing import Dict, Any


class AdvancedBot:
    def __init__(self, engine, llm_client):
        self.engine = engine
        self.llm = llm_client

    def process_query(self, query: str, k: int = 5) -> Dict[str, Any]:
        start_t = time.time()

        # 1. Retrieve
        # Pass k dynamically to the engine
        context_data = self.engine.retrieve(query, k=k)
        retrieval_time = time.time() - start_t

        if not context_data:
            return {
                "response": "No relevant locations found.",
                "context": [],
                "timings": {"retrieval": retrieval_time, "generation": 0},
            }

        # 2. Prompt
        context_str = "\n\n".join(
            [f"--- Doc {i+1} ---\n{c['content']}" for i, c in enumerate(context_data)]
        )

        system_prompt = (
            "You are an expert Tamil Nadu Tourism Guide. Always answer ONLY using the information "
            "found in the provided CONTEXT. If the answer is not in the context, say "
            "'I don't have that information in my database yet.'\n\n"
            "RAG RULES:\n"
            "- Use ONLY facts present in the CONTEXT. No external knowledge, no assumptions.\n"
            "- If multiple context items appear, combine them logically without making up links.\n"
            "- Quote relevant fields verbatim when referring to: Logistics, Best Time to Visit, "
            "Nearby Places, History, Timings\n"
            "- If the user asks about how to reach, explicitly include 'Logistics'.\n"
            "- If the user asks for what else to visit, use 'Nearby Connected Places'.\n"
            "- If information is missing or incomplete in context, state it transparently.\n\n"
            "GUARDRAILS:\n"
            "- No hallucinations: do not invent facts, numbers, distances, or names.\n"
            "- No unsafe suggestions: avoid risky travel advice, medical advice, or claims "
            "about legal/government procedures.\n"
            "- Keep tone friendly, professional, and tourism-oriented.\n\n"
            "OUTPUT STYLE:\n"
            "- Write concise, clear paragraphs.\n"
            "- Keep answers helpful and welcoming.\n"
            "- Use bullet points only when needed (e.g., itineraries).\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"USER QUERY: {query}"
        )

        # 3. Generate
        start_gen = time.time()
        answer = self.llm.generate(system_prompt)
        gen_time = time.time() - start_gen

        return {
            "response": answer,
            "context": [
                c["name"] for c in context_data
            ],  # Return source names for debugging
            "timings": {
                "retrieval": round(retrieval_time, 2),
                "generation": round(gen_time, 2),
            },
        }
