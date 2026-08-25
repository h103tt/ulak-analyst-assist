# ULAK Test Analysis Agent

Backend agent for the ULAK Quality Test Analyst. Talks to a local Ollama
model, retrieves context from the internal standards knowledge base (and
any files the user uploads in a session), and answers with structured,
citation-checked test analysis.

## RAG pipeline: handling ambiguous / multi-step questions

A single embedding-similarity search over the knowledge base breaks down on
questions that are vague, phrased differently than the source text, or
require pulling from more than one standard/section. To handle that, the
retrieval pipeline in `vector_embed.py`, `agent.py`, and `refine.py`
combines several techniques:

- **Query expansion** (`vector_embed.py` -> `build_expanded_retriever`)
  Uses LangChain's `MultiQueryRetriever`, backed by the local LLM, to
  generate a few alternate phrasings of the user's query. Each phrasing is
  retrieved and reranked separately, then the results are merged and
  de-duplicated. This improves recall when the user's wording doesn't match
  the standard's wording.

- **Query reformulation** (`agent.py` system prompt)
  The agent is instructed to rewrite elliptical follow-up questions (e.g.
  "what about the timing requirement?", "and for the other standard?")
  into a complete, standalone query -- using the conversation so far --
  *before* calling a retriever tool, rather than passing the raw follow-up
  text as the search query.

- **Multi-hop retrieval** (`agent.py` system prompt + agent tool loop)
  When a question spans more than one standard or compares two
  requirements, the agent issues a separate, focused search call per
  standard/topic instead of one combined query, chaining tool calls across
  turns of its reasoning loop as needed.

- **Re-ranking** (`vector_embed.py` -> `build_reranking_retriever`)
  Retrieves a wider MMR candidate set (`k * 4`) from Chroma, then reranks
  it down to the final top-k with a local cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Retrieve-then-rerank is cheaper
  and more accurate than trusting embedding similarity/MMR diversity alone.

- **Answer aggregation** (`refine.py` -> `aggregate_tool_context`)
  Concatenates and de-duplicates the tool results (possibly from multiple
  retriever calls across the turn) into a single context block, so the
  refinement pass below has one consistent source of truth to check
  against.

- **Answer refinement / response refinement** (`refine.py` -> `refine_answer`,
  wired into `bridge.py`)
  A second LLM pass over the agent's draft answer, checked against the
  aggregated retrieved context:
  - every citation (standard name, clause/section) must appear verbatim in
    the retrieved text, or it's removed/corrected -- never invented;
  - test cases with overlapping preconditions must not produce
    contradictory expected results;
  - requirements flagged as ambiguous must name the specific missing
    dimension (threshold/limits, duration/timing, error messaging, state
    persistence, recovery/unlock procedure).
  Skipped when nothing was retrieved, since there's no context to check
  citations against.

## Running

See the root [README.md](../README.md) for how to run the frontend and
this backend together.
