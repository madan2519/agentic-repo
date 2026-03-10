# reflective_agent.py
import json
import os
from typing import Dict, Any

# (Optional) load .env during local dev
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

from langchain_openai import ChatOpenAI                         # OpenAI provider  [docs]
from langchain_core.prompts import ChatPromptTemplate           # Prompt builder   [docs]
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser  # [docs]

# -------------------------
# Model configuration
# -------------------------
MODEL_ID = os.getenv("LLM_MODEL", "gpt-4o-mini")  # any OpenAI Chat Completions model you have
# Low temperature for stable, deterministic behavior across planning/eval loops
llm_generative = ChatOpenAI(model=MODEL_ID, temperature=0.2)
llm_evaluator = ChatOpenAI(model=MODEL_ID, temperature=0)

# -------------------------
# 1) DRAFTER (Generate)
# -------------------------
draft_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a precise assistant. Produce a clear, correct, concise answer. "
         "Cite facts only if you are confident. If uncertain, say so briefly."),
        ("human", "{question}")
    ]
)
draft_chain = draft_prompt | llm_generative | StrOutputParser()

# -------------------------
# 2) EVALUATOR (Critique → JSON score)
# -------------------------
# The evaluator must *always* return strict JSON with the specified schema.
judge_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a strict evaluator. Assess the ASSISTANT_ANSWER against the USER_QUESTION. "
         "Return STRICT JSON ONLY with this schema and NO extra text:\n"
         "{\n"
         '  "decision": "pass" | "fail",\n'
         '  "score": float,             // 0.0 to 1.0\n'
         '  "feedback": "string"        // concrete, actionable rewrite guidance\n'
         "}\n"
         "Rubric: correctness (factual, on-topic), completeness (addresses all parts), "
         "safety (no harmful/unsafe guidance), and clarity/conciseness."),
        ("human",
         "USER_QUESTION:\n{question}\n\n"
         "ASSISTANT_ANSWER:\n{answer}\n\n"
         "Return JSON now.")
    ]
)
judge_chain = judge_prompt | llm_evaluator | JsonOutputParser()

# -------------------------
# 3) REVISER (Improve using feedback)
# -------------------------
revise_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a careful rewriter. Improve the answer based on evaluator FEEDBACK while "
         "preserving factual accuracy and staying concise. If there is uncertainty, state it explicitly."),
        ("human",
         "USER_QUESTION:\n{question}\n\n"
         "PREVIOUS_ANSWER:\n{answer}\n\n"
         "FEEDBACK:\n{feedback}\n\n"
         "Rewrite the answer now.")
    ]
)
revise_chain = revise_prompt | llm_generative | StrOutputParser()

# -------------------------
# Reflection Orchestrator
# -------------------------
def reflective_answer(question: str, *, pass_threshold: float = 0.75, max_retries: int = 2) -> Dict[str, Any]:
    """
    Draft -> Evaluate -> (optionally) Revise loop.
    Returns a dict with final_answer, attempts (list of traces).
    """
    attempts = []

    # 0) Initial draft
    draft = draft_chain.invoke({"question": question})

    for attempt in range(max_retries + 1):
        # Evaluate
        try:
            verdict = judge_chain.invoke({"question": question, "answer": draft})
            # verdict expected: {"decision": "pass"/"fail", "score": float, "feedback": str}
            decision = str(verdict.get("decision", "fail")).lower()
            score = float(verdict.get("score", 0.0))
            feedback = verdict.get("feedback", "").strip()
        except Exception as e:
            # Malformed JSON (rare). Treat as fail with generic feedback.
            decision, score, feedback = "fail", 0.0, f"Evaluator returned malformed output: {e}"

        attempts.append(
            {
                "attempt": attempt + 1,
                "draft": draft,
                "decision": decision,
                "score": score,
                "feedback": feedback,
            }
        )

        # Check pass criteria
        if decision == "pass" and score >= pass_threshold:
            return {
                "final_answer": draft,
                "attempts": attempts,
                "passed": True,
                "score": score,
            }

        # If we still have retries, revise; else break with last draft
        if attempt < max_retries:
            draft = revise_chain.invoke(
                {"question": question, "answer": draft, "feedback": feedback or "Improve clarity and correctness."}
            )
        else:
            break

    # Failed to pass within retry budget; return best-effort draft
    # Optionally, include a short note that improvements were attempted.
    return {
        "final_answer": draft,
        "attempts": attempts,
        "passed": False,
        "score": attempts[-1]["score"] if attempts else 0.0,
        "note": "Returned best available answer after reflective improvements.",
    }

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    print("✅ Reflective Agent (LangChain) ready.")
    print("Model:", MODEL_ID)
    print("Ask a question (or 'exit'):\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not q or q.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        try:
            result = reflective_answer(q, pass_threshold=0.8, max_retries=2)
            print("\nAssistant:", result["final_answer"])
            print("\n— Reflection Trace —")
            for a in result["attempts"]:
                print(f"[Attempt {a['attempt']}] score={a['score']:.2f}, decision={a['decision']}")
                if a["feedback"]:
                    print("Feedback:", a["feedback"])
                print()
            if not result.get("passed", False):
                print(result.get("note", ""))
            print()
        except Exception as e:
            print("Error:", e)
            print("Tip: Ensure OPENAI_API_KEY is set and the model name is valid.\n")