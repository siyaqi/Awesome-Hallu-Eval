# Awesome-Hallu-Eval
This is a curated list of evaluators designed to assess model hallucination. Here, you can easily find the right tools you need to evaluate and analyze hallucination behavior in language models.

## Dataset & Benchmark
### 🟦 Traditional Task

| Category         | Dataset             | Task                    | Size            | Label Type                                             | Links                                                                                            |
| ---------------- | ------------------- | ----------------------- | --------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| Traditional Task | DialogueNLI         | Dialogue                | 343k pairs      | Entailment/contradiction/neutral                       | [GitHub](https://wellecks.com/dialogue_nli/)                                                     |
|                  | CoGenSumm           | Summarization           | 100 articles    | Sentence correct/incorrect                             | [Dataset Link](https://tudatalib.ulb.tu-darmstadt.de/items/9a3612a3-4fba-400f-8b23-bf1e917d894f) |
|                  | XSumFaith           | Summarization           | 500 articles    | Span intrinsic/extrinsic hallucination                 | [GitHub](https://github.com/google-research-datasets/xsum_hallucination_annotations)             |
|                  | QAGS                | Summarization           | 474 articles    | Consistent/inconsistent                                | [GitHub](https://github.com/W4ngatang/qags)                                                      |
|                  | Polytope            | Summarization           | 1.5k summaries  | Intrinsic/extrinsic hallucination                      | [GitHub](https://github.com/hddbang/PolyTope)                                                    |
|                  | FRANK               | Summarization           | 2.25k summaries | Various error types (entity, discourse, grammar, etc.) | [GitHub](https://github.com/artidoro/frank)                                                      |
|                  | Falsesum            | Summarization           | 2.97k articles  | Consistent/inconsistent                                | [GitHub](https://github.com/joshuabambrick/Falsesum)                                             |
|                  | FactEval            | Dialogue summarization  | 150 dialogues   | Consistent/inconsistent                                | [GitHub](https://github.com/BinWang28/FacEval)                                                   |
|                  | Text Simplification | Text simplification     | 1.56k pairs     | Insertion/deletion/substitution                        | [GitHub](https://github.com/AshOlogn/Evaluating-Factuality-in-Text-Simplification)               |
|                  | NonFactS            | Augmented summarization | 400k samples    | Non-factual summaries                                  | [GitHub](https://github.com/ASoleimaniB/NonFactS)                                                |
|                  | RefMatters          | Dialogue summarization  | 4k pairs        | FRANK errors                                           | [GitHub](https://github.com/kite99520/DialSummFactCorr)                                          |
|                  | DiaHalu             | Dialogue generation     | 1.0k samples    | Dialogue-level factuality/faithfulness                 | [GitHub](https://github.com/ECNU-ICALK/DiaHalu)                                                  |
|                  | TofuEval            | Dialogue summarization  | 1.5k pairs      | Consistent/inconsistent                                | [GitHub](https://github.com/amazon-science/tofueval)                                             |
|                  | RAGTruth            | RAG systems             | 2.97k samples   | Evident/subtle conflict/baseless                       | [GitHub](https://github.com/ParticleMedia/RAGTruth)                                              |
|                  | SummaCoz            | Summarization           | 6.07k summaries | Explanation                                            | [HF Dataset](https://huggingface.co/datasets/nkwbtb/SummaCoz)                                    |
|                  | FaithBench          | Summarization           | 750 samples     | Questionable/benign/unwanted                           | [GitHub](https://github.com/vectara/FaithBench)                                                  |


### 🟩 General Factuality
| Category           | Dataset    | Task                        | Size           | Label Type                                               | Links                                                         |
| ------------------ | ---------- | --------------------------- | -------------- | -------------------------------------------------------- | ------------------------------------------------------------- |
| General Factuality | Q2         | Dialogue QA                 | 750 samples    | Consistent/inconsistent                                  | [GitHub](https://github.com/orhonovich/q-squared/tree/main)   |
|                    | TruthfulQA | Truthfulness QA             | 817 pairs      | QA truthfulness                                          | [GitHub](https://github.com/sylinrl/TruthfulQA)               |
|                    | FACTOR     | Multi-choice                | 4.27k samples  | FRANK errors                                             | [GitHub](https://github.com/AI21Labs/factor)                  |
|                    | HaluEval   | QA/Summarization/Dialog/etc | 35K samples    | Hallucinations yes/no                                    | [GitHub](https://github.com/RUCAIBox/HaluEval)                |
|                    | PHD        | Passage-level QA            | 300 entities   | factual/non-factual/unverifiable                         | [GitHub](https://github.com/maybenotime/PHD)                  |
|                    | FAVA       | General queries             | 200 queries    | Entity/relation/contradictory/invented/subjective errors | [Project Page](https://fine-grained-hallucination.github.io/) |
|                    | THaMES     | General QA                  | 2.1k samples   | Correct/hallucinated                                     | [GitHub](https://github.com/holistic-ai/THaMES)               |
|                    | HELM       | LLM continue generation     | 1.2k passages  | Hallucination/non-hallucination                          | [GitHub](https://github.com/oneal2000/MIND/tree/main)         |
|                    | HalluLens  | LLM generation              | 130k instances | Intrinsic/extrinsic/factuality                           | [GitHub](https://github.com/facebookresearch/HalluLens)       |




| Category                | Dataset             | Task                               | Size             | Label Type                               | Links                                                              |
| ----------------------- | ------------------- | ---------------------------------- | ---------------- | ---------------------------------------- | ------------------------------------------------------------------ |
| Evaluate the Evaluators | Wizard of Wikipedia | Knowledge-based dialogue eval      | 22.3k dialogues  | Knowledge selection, response generation | [Project Page](https://parl.ai/projects/wizard_of_wikipedia/)      |
|                         | TopicalChat         | Knowledge-based dialogue eval      | 10.79k dialogues | Knowledge source                         | [GitHub](https://github.com/alexa/Topical-Chat)                    |
|                         | SummEval            | Summarization metric eval          | 1.6k summaries   | Consistent/inconsistent                  | [GitHub](https://github.com/Yale-LILY/SummEval)                    |
|                         | BEAMetrics          | Multi-task metric eval             | Not specified    | Coherence                                | [GitHub](https://github.com/ThomasScialom/BEAMetrics)              |
|                         | CI‑ToD              | Task-oriented dialogue             | 3.19k dialogues  | Consistent/inconsistent                  | [GitHub](https://github.com/yizhen20133868/CI-ToD)                 |
|                         | SummaC              | Summarization metric eval          | Not specified    | Consistent/inconsistent                  | [GitHub](https://github.com/tingofurro/summac)                     |
|                         | BEGIN               | Knowledge-based dialogue           | 12k turns        | Fully/not attributable/generic           | [GitHub](https://github.com/google/BEGIN-dataset)                  |
|                         | FaithDial           | Dialogue eval                      | 5.65k dialogues  | BEGIN, VRM                               | [HF Dataset](https://huggingface.co/datasets/McGill-NLP/FaithDial) |
|                         | DialSumMeval        | Dialogue summarization metric eval | 1.5k summaries   | Consistent/inconsistent                  | [GitHub](https://github.com/kite99520/DialSummEval)                |
|                         | TRUE                | Cross-task metric eval             | \~200k samples   | Consistent/inconsistent                  | [GitHub](https://github.com/google-research/true)                  |
|                         | AGGREFACT           | Summarization metric eval          | 59.7k samples    | Consistent/inconsistent                  | [HF Dataset](https://huggingface.co/datasets/lytang/LLM-AggreFact) |
|                         | FELM                | Multi-task metric eval             | 847 samples      | Factuality positive/negative             | [GitHub](https://github.com/hkust-nlp/felm)                        |











