Feature: End-to-end RAG pipeline for ulak-analyst-assist
  As a QA engineer
  I want to verify ingestion, retrieval, and generation over the real
  standards knowledge base (Chroma collection "iso_files")
  So that the agent only answers with grounded, traceable information

  # System under test:
  #   - Ingestion : vector_embed.process_single_file / load_document (Docling + TextLoader/CSVLoader)
  #   - Storage   : Chroma collection "iso_files" (test_analysis_agent/chromadb)
  #   - Retrieval : vector_embed.kb_compression_retriever (MMR k=20 -> CrossEncoderReranker top_n=3)
  #   - Generation: agent.build_agent (ChatOllama qwen3.5:4b) via FastAPI /chat and /trace (bridge.py)

  # ===================================================================
  # 1. KB Ingestion
  # ===================================================================

  @ingestion
  Scenario Outline: Ingest a supported document format into the knowledge base
    Given a "<format>" document "<filename>" placed under the knowledge_base directory
    When the ingestion pipeline processes the file
    Then the loader returns at least one chunk
    And every chunk carries "source_file", "standard", "category", "citation_tag", "section" and "page" metadata
    And every chunk's page_content is wrapped in a "<chunk id=... source=... standard=... category=... section=... page=...>" tag
    And the chunks are written into the "iso_files" Chroma collection
    And the collection's document count increases by the number of new chunks

    Examples:
      | format | filename                        |
      | pdf    | MIL-STD-461.pdf                 |
      | pdf    | MIL-STD-1586A.pdf                |
      | pdf    | 15288-2023-2.pdf                 |
      | pdf    | 29119-1-2022.pdf                 |
      | pdf    | IEEE-Test-Doc-829-2008.pdf       |
      | pdf    | MIL-STD-882E.pdf                 |
      | pdf    | SP800-53_REV-3.PDF               |
      | md     | requirements_and_testing.md       |

  @ingestion
  Scenario: Ingest a JSON document (currently unsupported format)
    Given a ".json" file with structured requirement data
    When it is passed to vector_embed.load_document
    Then the loader falls through to the generic Docling binary path
    And the current behavior is UNDEFINED for structured JSON
    # This scenario documents a real gap: load_document() branches on
    # .csv / .txt / .md explicitly and treats everything else (including
    # .json) as a binary Docling document. See test_ingestion_formats.py::
    # test_json_ingestion_documents_current_gap for the executable proof
    # and a recommendation to add an explicit JSON branch (e.g. JSONLoader
    # with jq_schema) before this format is advertised as supported.

  @ingestion
  Scenario: Corrupted or empty PDF does not crash the ingestion pipeline
    Given a PDF file that is empty or unreadable by Docling
    When process_single_file processes it
    Then no exception propagates out of process_single_file
    And an "err" status is logged for that file
    And zero chunks are returned for that file
    And the rest of the batch continues processing (ThreadPoolExecutor isolation)

  @ingestion
  Scenario: DOCS registry stays in sync with files on disk
    Given the DOC_METADATA_LOOKUP table in vector_embed.py
    When each entry's file is looked up under knowledge_base/<category>/<filename>
    Then the file must exist on disk
    # Regression guard for the "stale registry" issue found during this
    # audit: DOCS currently references MIL-STD-810H_CHG-1.pdf, 830-1998.pdf,
    # 29148-2018.pdf, ISO-9001-2015.pdf and RTCA-DO-160G.pdf, none of which
    # are present after the recent knowledge-base cleanup commits.

  # ===================================================================
  # 2. Context Retrieval (Recall / Precision)
  # ===================================================================

  @retrieval
  Scenario Outline: Top-K retrieval surfaces the correct standard for an in-domain question
    Given the "iso_files" collection is populated
    When I query search_testing_standards with "<question>"
    Then at least one of the top 3 reranked chunks has standard "<expected_standard>"
    And the returned text is wrapped in a valid <chunk ...> tag with a non-empty "source" attribute

    Examples:
      | question                                                                 | expected_standard          |
      | What electromagnetic interference limits does MIL-STD-461 define?       | MIL-STD-461                |
      | What system life cycle processes are defined in ISO/IEC/IEEE 15288?     | 15288-2023-2                |
      | What test documentation artifacts does IEEE 829 require?                | IEEE-Test-Doc-829-2008      |
      | What does MIL-STD-882E require for hazard risk assessment?              | MIL-STD-882E                |
      | What security controls does SP 800-53 define for access control?       | SP800-53_REV-3              |
      | What test process concepts does ISO/IEC/IEEE 29119-1 define?            | 29119-1-2022                |

  @retrieval
  Scenario: Hybrid / MMR retrieval deduplicates near-identical chunks
    Given a query that matches multiple overlapping chunks from the same section
    When the MMR retriever (search_type="mmr", k=20) runs before reranking
    Then the returned candidate set favors diversity over pure similarity
    And the final reranked top_n=3 does not contain two chunks with identical page_content

  @retrieval
  Scenario: Retrieval recall and precision meet the quality bar (deepeval)
    Given a golden question with a known expected_standard and expected_category
    When ContextualRecallMetric and ContextualPrecisionMetric are computed
    Then both scores are >= the configured threshold (default 0.5)

  @retrieval
  Scenario: Query expansion improves recall for an ambiguous follow-up
    Given a conversation where the latest user message is an elliptical follow-up
      ("what about the timing requirement")
    When the agent reformulates the query per the system prompt's
      "QUERY REFORMULATION" instructions before calling search_testing_standards
    Then the tool is called with a standalone, entity-complete query
    And not with the raw elliptical follow-up text

  # ===================================================================
  # 3. Generation & Faithfulness (Groundedness)
  # ===================================================================

  @generation
  Scenario Outline: Agent answers are grounded in retrieved KB content
    Given the golden question "<question>" targeting standard "<expected_standard>"
    When I POST the question to /trace
    Then the response has kb_called = true
    And kb_returned_content = true
    And the answer text does not contain a "(Standard, Section X)" citation
      unless that exact standard/section pair appeared in a retrieved chunk
    And a GEval "Answer Correctness" judge score is >= 0.6 against the retrieved context
    And a GEval "Citation Accuracy" judge score is >= 0.6 against the retrieved context

    Examples:
      | question                                                                 | expected_standard      |
      | Summarize the scope of MIL-STD-461.                                      | MIL-STD-461            |
      | What is the purpose of ISO/IEC/IEEE 29119-1?                             | 29119-1-2022            |
      | What does IEEE 829-2008 say about test plans?                           | IEEE-Test-Doc-829-2008  |

  @generation
  Scenario: Agent refuses to answer from parametric memory when retrieval is empty
    Given a question about a real standard that is NOT present in the knowledge base
      (e.g. "What does DO-178C say about software levels?")
    When the agent calls search_testing_standards and gets no relevant hits
    Then the answer explicitly states the retrieved sections do not cover this
    And the answer contains no invented clause/section numbers

  @generation
  Scenario: Test case generation follows the mandatory ISO/IEC/IEEE 29119 workflow
    Given a well-specified, testable requirement is provided as context
    When the agent is asked to generate test cases for it
    Then the answer contains Test ID/Traceability, Test Type, Preconditions,
      Test Steps and Expected Result sections
    And the closing coverage check counts (Total/Testable/Ambiguous/With test cases) are present
    And "Testable requirements" equals "Requirements with test cases"

  @generation
  Scenario: Ambiguous requirements are flagged instead of guessed
    Given an ambiguous requirement missing a threshold, timing, or error-messaging detail
      (e.g. "The system shall respond quickly under high load.")
    When the agent evaluates it
    Then the requirement is marked "NOT TESTABLE AS WRITTEN"
    And no test case is generated for it
    And the missing dimension is named explicitly

  # ===================================================================
  # 4. Edge / Unhappy Paths
  # ===================================================================

  @edge
  Scenario: Empty query is rejected before reaching the agent
    Given an empty or whitespace-only "message" field
    When I POST to /trace
    Then the response status is 400
    And the response body contains an "error" field

  @edge
  Scenario: Empty messages array on /chat is rejected
    Given an empty "messages" list
    When I POST to /chat
    Then the response status is 400

  @edge
  Scenario: Out-of-domain query does not crash the pipeline
    Given a query entirely unrelated to any standard in the KB
      (e.g. "What is the best recipe for lasagna?")
    When I POST it to /trace
    Then the response status is 200
    And the answer either declines to answer from the standards KB
      or clearly states no relevant section was found

  @edge
  Scenario: Oversized conversation history is trimmed, not rejected
    Given a conversation history whose token count exceeds HISTORY_TOKEN_BUDGET (24000)
    When trim_history_middleware runs before the model call
    Then the trimmed message list starts on a human turn
    And its token count is <= HISTORY_TOKEN_BUDGET
    And the request still succeeds end-to-end

  @edge
  Scenario: Oversized single attachment context is truncated on a word boundary
    Given attached file context text longer than CONTEXT_CHAR_BUDGET (60000 chars)
    When bridge.truncate_context runs
    Then the result is <= CONTEXT_CHAR_BUDGET + len("\n...[context truncated]")
    And the cut does not split a word (cuts on the last space before the limit)

  @edge
  Scenario: Agent is not ready yet
    Given the base agent has not finished loading (app_state["base_agent"] is None)
    When I POST to /chat or /trace
    Then the response status is 503

  @edge
  Scenario: Downstream LLM/Chroma failure surfaces as a controlled error
    Given the agent raises during invoke() (e.g. Ollama connection refused)
    When I POST to /trace
    Then the response status is 500 and includes the error message
    When I POST to /chat
    Then the stream still completes with a "type: finish" event
    And the streamed text contains "Agent error"
