# Awesome-Hallu-Eval: Can we catch the Elephant?
This is a curated list of evaluators designed to assess model hallucination. Here, you can easily find the right tools you need to evaluate and analyze hallucination behavior in language models.

Hallucinations can no longer remain the elephant in the room, they must be actively hunted down and captured.


## Dataset & Benchmark
### 🟦 Traditional Task
| Dataset             | Task                    | Size            | Label Type                                             | Links                                                                                            |
| ------------------- | ----------------------- | --------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| DialogueNLI         | Dialogue                | 343k pairs      | Entailment/contradiction/neutral                       | [GitHub](https://wellecks.com/dialogue_nli/)                                                     |
| CoGenSumm           | Summarization           | 100 articles    | Sentence correct/incorrect                             | [Dataset Link](https://tudatalib.ulb.tu-darmstadt.de/items/9a3612a3-4fba-400f-8b23-bf1e917d894f) |
| XSumFaith           | Summarization           | 500 articles    | Span intrinsic/extrinsic hallucination                 | [GitHub](https://github.com/google-research-datasets/xsum_hallucination_annotations)             |
| QAGS                | Summarization           | 474 articles    | Consistent/inconsistent                                | [GitHub](https://github.com/W4ngatang/qags)                                                      |
| Polytope            | Summarization           | 1.5k summaries  | Intrinsic/extrinsic hallucination                      | [GitHub](https://github.com/hddbang/PolyTope)                                                    |
| FRANK               | Summarization           | 2.25k summaries | Various error types (entity, discourse, grammar, etc.) | [GitHub](https://github.com/artidoro/frank)                                                      |
| Falsesum            | Summarization           | 2.97k articles  | Consistent/inconsistent                                | [GitHub](https://github.com/joshuabambrick/Falsesum)                                             |
| FactEval            | Dialogue summarization  | 150 dialogues   | Consistent/inconsistent                                | [GitHub](https://github.com/BinWang28/FacEval)                                                   |
| Text Simplification | Text simplification     | 1.56k pairs     | Insertion/deletion/substitution                        | [GitHub](https://github.com/AshOlogn/Evaluating-Factuality-in-Text-Simplification)               |
| NonFactS            | Augmented summarization | 400k samples    | Non-factual summaries                                  | [GitHub](https://github.com/ASoleimaniB/NonFactS)                                                |
| RefMatters          | Dialogue summarization  | 4k pairs        | FRANK errors                                           | [GitHub](https://github.com/kite99520/DialSummFactCorr)                                          |
| DiaHalu             | Dialogue generation     | 1.0k samples    | Dialogue-level factuality/faithfulness                 | [GitHub](https://github.com/ECNU-ICALK/DiaHalu)                                                  |
| TofuEval            | Dialogue summarization  | 1.5k pairs      | Consistent/inconsistent                                | [GitHub](https://github.com/amazon-science/tofueval)                                             |
| RAGTruth            | RAG systems             | 2.97k samples   | Evident/subtle conflict/baseless                       | [GitHub](https://github.com/ParticleMedia/RAGTruth)                                              |
| SummaCoz            | Summarization           | 6.07k summaries | Explanation                                            | [HF Dataset](https://huggingface.co/datasets/nkwbtb/SummaCoz)                                    |
| FaithBench          | Summarization           | 750 samples     | Questionable/benign/unwanted                           | [GitHub](https://github.com/vectara/FaithBench)                                                  |

### 🟩 General Factuality
| Dataset    | Task                        | Size           | Label Type                                               | Links                                                         |
| ---------- | --------------------------- | -------------- | -------------------------------------------------------- | ------------------------------------------------------------- |
| Q2         | Dialogue QA                 | 750 samples    | Consistent/inconsistent                                  | [GitHub](https://github.com/orhonovich/q-squared/tree/main)   |
| TruthfulQA | Truthfulness QA             | 817 pairs      | QA truthfulness                                          | [GitHub](https://github.com/sylinrl/TruthfulQA)               |
| FACTOR     | Multi-choice                | 4.27k samples  | FRANK errors                                             | [GitHub](https://github.com/AI21Labs/factor)                  |
| HaluEval   | QA/Summarization/Dialog/etc | 35K samples    | Hallucinations yes/no                                    | [GitHub](https://github.com/RUCAIBox/HaluEval)                |
| PHD        | Passage-level QA            | 300 entities   | factual/non-factual/unverifiable                         | [GitHub](https://github.com/maybenotime/PHD)                  |
| FAVA       | General queries             | 200 queries    | Entity/relation/contradictory/invented/subjective errors | [Project Page](https://fine-grained-hallucination.github.io/) |
| THaMES     | General QA                  | 2.1k samples   | Correct/hallucinated                                     | [GitHub](https://github.com/holistic-ai/THaMES)               |
| HELM       | LLM continue generation     | 1.2k passages  | Hallucination/non-hallucination                          | [GitHub](https://github.com/oneal2000/MIND/tree/main)         |
| HalluLens  | LLM generation              | 130k instances | Intrinsic/extrinsic/factuality                           | [GitHub](https://github.com/facebookresearch/HalluLens)       |

### 🟨 Evaluate the Evaluators
| Dataset             | Task                               | Size             | Label Type                               | Links                                                              |
| ------------------- | ---------------------------------- | ---------------- | ---------------------------------------- | ------------------------------------------------------------------ |
| Wizard of Wikipedia | Knowledge-based dialogue eval      | 22.3k dialogues  | Knowledge selection, response generation | [Project Page](https://parl.ai/projects/wizard_of_wikipedia/)      |
| TopicalChat         | Knowledge-based dialogue eval      | 10.79k dialogues | Knowledge source                         | [GitHub](https://github.com/alexa/Topical-Chat)                    |
| SummEval            | Summarization metric eval          | 1.6k summaries   | Consistent/inconsistent                  | [GitHub](https://github.com/Yale-LILY/SummEval)                    |
| BEAMetrics          | Multi-task metric eval             | Not specified    | Coherence                                | [GitHub](https://github.com/ThomasScialom/BEAMetrics)              |
| CI‑ToD              | Task-oriented dialogue             | 3.19k dialogues  | Consistent/inconsistent                  | [GitHub](https://github.com/yizhen20133868/CI-ToD)                 |
| SummaC              | Summarization metric eval          | Not specified    | Consistent/inconsistent                  | [GitHub](https://github.com/tingofurro/summac)                     |
| BEGIN               | Knowledge-based dialogue           | 12k turns        | Fully/not attributable/generic           | [GitHub](https://github.com/google/BEGIN-dataset)                  |
| FaithDial           | Dialogue eval                      | 5.65k dialogues  | BEGIN, VRM                               | [HF Dataset](https://huggingface.co/datasets/McGill-NLP/FaithDial) |
| DialSumMeval        | Dialogue summarization metric eval | 1.5k summaries   | Consistent/inconsistent                  | [GitHub](https://github.com/kite99520/DialSummEval)                |
| TRUE                | Cross-task metric eval             | ~200k samples    | Consistent/inconsistent                  | [GitHub](https://github.com/google-research/true)                  |
| AGGREFACT           | Summarization metric eval          | 59.7k samples    | Consistent/inconsistent                  | [HF Dataset](https://huggingface.co/datasets/lytang/LLM-AggreFact) |
| FELM                | Multi-task metric eval             | 847 samples      | Factuality positive/negative             | [GitHub](https://github.com/hkust-nlp/felm)                        |



## Hallucination Evaluation Methods



### After LLM Era

**SCALE** - "SCALE: Evaluating Factual Consistency in Long Dialogue Systems" (2024)
- **Data**: ScreenEval dataset from LLM and Human sources
- **Model**: Flan-T5
- **Method**: NLI approach
- **Evaluation Metrics**: Likelihood
- **Evaluation Perspective**: SF ✓, WF ✗

**Evaluating Factual Consistency of Summaries with Large Language Models** (2023)
- **Data**: SummEval, XSumFaith, Goyal21, CLIFF datasets
- **Model**: Flan-T5, code-davinci-002, text-davinci-003, ChatGPT, GPT-4
- **Method**: Vanilla/COT/Sent-by-Sent Prompt approaches
- **Evaluation Metrics**: Balanced Accuracy
- **Evaluation Perspective**: SF ✓, WF ✗

**GPTScore** - "GPTScore: Evaluating Factual Consistency with Large Language Models" (2023)
- **Data**: 37 datasets from 4 tasks
- **Model**: GPT-2, OPT, FLAN, GPT-3
- **Method**: Direct Assessment
- **Evaluation Metrics**: Direct Score
- **Evaluation Perspective**: SF ✓, WF ✗

**G-Eval** - "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment" (2023)
- **Data**: SummEval, Topical-Chat, QAGS datasets
- **Model**: GPT-4
- **Method**: COT and Form-filling approaches
- **Evaluation Metrics**: Weighted Scores
- **Evaluation Perspective**: SF ✓, WF ✗

**ChatGPT as a Factual Consistency Evaluator for Text Summarization** (2023)
- **Data**: 5 datasets from 3 tasks
- **Model**: ChatGPT
- **Method**: Direct Assessment and Rating
- **Evaluation Metrics**: Direct score
- **Evaluation Perspective**: SF ✓, WF ✗

**ChainPoll** - "ChainPoll: A Chain-of-Thought Approach to Hallucination Detection" (2024)
- **Data**: RealHall-closed, RealHall-open datasets from COVID-QA, DROP, Open Ass prompts, TriviaQA
- **Model**: gpt-3.5-turbo
- **Method**: Direct Assessment (2-class)
- **Evaluation Metrics**: Accuracy
- **Evaluation Perspective**: SF ✓, WF ✗

**EigenScore** - "EigenScore: Evaluating Factual Consistency via Semantic Consistency in Embedding Space" (2024)
- **Data**: CoQA, SQuAD, TriviaQA, Natural Questions
- **Model**: LLaMA, OPT
- **Method**: Semantic Consistency/Diversity in Dense Embedding Space
- **Evaluation Metrics**: AUROC, PCC
- **Evaluation Perspective**: SF ✓, WF ✗

**TruthfulQA** - "TruthfulQA: Measuring How Models Mimic Human Falsehoods" (2021)
- **Data**: TruthfulQA dataset from LLM and Human sources
- **Model**: GPT-3-175B
- **Method**: Answer Match
- **Evaluation Metrics**: Percentage, Likelihood
- **Evaluation Perspective**: SF ✗, WF ✓

**HaluEval** - "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models" (2023)
- **Data**: Task-specific and General datasets from Alpaca, Task datasets, ChatGPT
- **Model**: ChatGPT
- **Method**: Direct Assessment
- **Evaluation Metrics**: Accuracy
- **Evaluation Perspective**: SF ✓, WF ✓

**FACTOR** - "FACTOR: A Benchmark for Evaluating Factual Consistency in Text Generation" (2023)
- **Data**: Wiki-/News-/Expert-FACTOR datasets from Wikipedia, RefinedWeb, ExpertQA
- **Model**: Not specified
- **Method**: FRANK Error Classification
- **Evaluation Metrics**: Likelihood
- **Evaluation Perspective**: SF ✗, WF ✓

**FELM** - "FELM: Benchmarking Factuality Evaluation of Large Language Models" (2023)
- **Data**: FELM dataset from TruthfulQA, Quora, MMLU, GSM8K, ChatGPT, Human
- **Model**: Vicuna, ChatGPT, GPT4
- **Method**: Direct Assessment
- **Evaluation Metrics**: F1, Balanced Accuracy
- **Evaluation Perspective**: SF ✓, WF ✓

**FreshQA** - "FreshQA: A Benchmark for Evaluating Factual Consistency in Real-Time Information" (2023)
- **Data**: Never/Slow Fast-changing, false-premise datasets from Human sources
- **Model**: Not specified
- **Method**: Answer Match
- **Evaluation Metrics**: Accuracy
- **Evaluation Perspective**: SF ✗, WF ✓

**RealTimeQA** - "RealTimeQA: What's More Important in Real-Time Question Answering?" (2023)
- **Data**: RealTimeQA dataset from CNN, THE WEEK, USA Today
- **Model**: GPT-3, T5
- **Method**: Answer Match
- **Evaluation Metrics**: Accuracy, EM, F1
- **Evaluation Perspective**: SF ✗, WF ✓

**ERBench** - "ERBench: A Benchmark for Entity-Relation Factual Consistency" (2023)
- **Data**: ERBench Database from 5 datasets on Kaggle
- **Model**: Not specified
- **Method**: Direct Assessment, String Matching
- **Evaluation Metrics**: Ans/Rat/Ans-Rat Accuracy, Hallucination Rate
- **Evaluation Perspective**: SF ✗, WF ✓

**FactScore** - "FactScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation" (2023)
- **Data**: Biographies in Wikipedia
- **Model**: InstructGPT, ChatGPT, PerplexityAI
- **Method**: Binary Classification
- **Evaluation Metrics**: Precision
- **Evaluation Perspective**: SF ✗, WF ✓

**BAMBOO** - "BAMBOO: A Comprehensive Benchmark for Evaluating Hallucination Detection" (2023)
- **Data**: SenHallu, AbsHallu datasets from 10 datasets across 5 tasks
- **Model**: ChatGPT
- **Method**: Answer Match
- **Evaluation Metrics**: Precision, Recall, F1
- **Evaluation Perspective**: SF ✓, WF ✗

**MedHalt** - "MedHalt: Medical Hallucination Detection and Evaluation" (2023)
- **Data**: MedHalt dataset from MedMCQA, Medqa USMILE, Medqa (Taiwan), Headqa, PubMed
- **Model**: ChatGPT
- **Method**: Answer Match
- **Evaluation Metrics**: Pointwise Score, Accuracy
- **Evaluation Perspective**: SF ✗, WF ✓

**ChineseFactEval** - "ChineseFactEval: A Benchmark for Evaluating Factual Consistency in Chinese Text Generation" (2023)
- **Data**: ChineseFactEval dataset
- **Model**: Not specified
- **Method**: FacTool, Human annotator
- **Evaluation Metrics**: Direct Score
- **Evaluation Perspective**: SF ✗, WF ✓

**UHGEval** - "UHGEval: A Comprehensive Evaluation Framework for Chinese Hallucination Detection" (2023)
- **Data**: UHGEval dataset from Chinese News Websites
- **Model**: GPT-4
- **Method**: Answer Match, Similarity
- **Evaluation Metrics**: Accuracy, Similarity Score
- **Evaluation Perspective**: SF ✗, WF ✓

**HalluQA** - "HalluQA: A Benchmark for Hallucination Detection in Question Answering" (2023)
- **Data**: HalluQA dataset from Human sources
- **Model**: GLM-130B, ChatGPT, GPT-4
- **Method**: Direct Assessment
- **Evaluation Metrics**: Non-hallucination Rate
- **Evaluation Perspective**: SF ✗, WF ✓

**FacTool** - "FacTool: Factuality Detection in Generative AI - A Tool Augmented Framework for Multi-Task and Multi-Domain Scenarios" (2023)
- **Data**: RoSE, FactPrompts, HumanEval, GSM-Hard, Self-instruct datasets
- **Model**: ChatGPT
- **Method**: Claim Extraction, Query Generation, Tool Querying, Evidence Collection, Agreement Verification
- **Evaluation Metrics**: Precision, Recall, F1
- **Evaluation Perspective**: SF ✓, WF ✓

**UFO** - "UFO: A Unified Framework for Open-domain Fact Verification" (2023)
- **Data**: NQ, HotpotQA, TruthfulQA, CNN/DM, Multi-News, MS MARCO
- **Model**: gpt-3.5-turbo-1106
- **Method**: Fact Unit Extraction, Fact Source Verification, Fact Consistency Discrimination
- **Evaluation Metrics**: Average Sub-scores
- **Evaluation Perspective**: SF ✓, WF ✓

**CONNER** - "CONNER: Consistency-based Evaluation for Open-domain Question Answering" (2023)
- **Data**: NQ, WoW datasets
- **Model**: NLI-RoBERTa-large, ColBERTv2
- **Method**: 3-way NLI
- **Evaluation Metrics**: Accuracy
- **Evaluation Perspective**: SF ✗, WF ✓

**SelfCheckGPT** - "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models" (2023)
- **Data**: SelfCheckGPT dataset from WikiBio
- **Model**: GPT-3
- **Method**: NLI, Ngram, QA, BERTScore, Prompt approaches
- **Evaluation Metrics**: AUC-PR
- **Evaluation Perspective**: SF ✓, WF ✗

**InterrogateLLM** - "InterrogateLLM: A Framework for Evaluating Factual Consistency in Large Language Models" (2023)
- **Data**: The Movies Dataset, GCI, The Book Dataset (Kaggle)
- **Model**: GPT-3, LLaMA-2
- **Method**: Query Consistency
- **Evaluation Metrics**: AUC, Balanced Accuracy
- **Evaluation Perspective**: SF ✗, WF ✓

**SAC³** - "SAC³: Self-Aware Chain-of-Thought for Factual Consistency Evaluation" (2023)
- **Data**: HotpotQA, NQ-open datasets
- **Model**: gpt-3.5-turbo, Falcon-7b-instruct, Guanaco-33b
- **Method**: Cross-checking, QA Pair Consistency
- **Evaluation Metrics**: AUROC
- **Evaluation Perspective**: SF ✓, WF ✓

**KoLA** - "KoLA: Carefully Benchmarking World Knowledge of Large Language Models" (2023)
- **Data**: KoLA dataset from Wikipedia, Updated News and Novels
- **Model**: Not specified
- **Method**: Self-contrast, Answer Match
- **Evaluation Metrics**: Similarity
- **Evaluation Perspective**: SF ✗, WF ✓

**RV** - "RV: Reference Verification for Factual Consistency Evaluation" (2023)
- **Data**: PHD dataset from Human Annotator
- **Model**: ChatGPT
- **Method**: Construct Query, Access Databases, Entity-Answer Match
- **Evaluation Metrics**: Precision, Recall, F1
- **Evaluation Perspective**: SF ✓, WF ✗

**SummEdits** - "SummEdits: A Benchmark for Evaluating Factual Consistency in Summarization" (2023)
- **Data**: SummEdits dataset from 9 datasets in Summarization task
- **Model**: gpt-3.5-turbo
- **Method**: Seed summary verification, Summary edits, Annotation
- **Evaluation Metrics**: Balanced Accuracy
- **Evaluation Perspective**: SF ✓, WF ✗

**LLM-Check** - "LLM-Check: Leveraging Large Language Models for Factual Consistency Evaluation" (2023)
- **Data**: FAVA-Annotation, RAGTruth, SelfcheckGPT datasets
- **Model**: Llama-2, Llama-3, GPT4, Mistral-7b
- **Method**: Analyze internal attention kernel maps, hidden activations and output prediction probabilities
- **Evaluation Metrics**: AUROC, FPR, Accuracy
- **Evaluation Perspective**: SF ✗, WF ✓

**PHR** - "PHR: Posterior Hallucination Rate for Evaluating Factual Consistency" (2023)
- **Data**: Synthetic data
- **Model**: Llama-2, Gemma-2
- **Method**: Posterior Hallucination Rate (Bayesian)
- **Evaluation Metrics**: Hallucination Rate
- **Evaluation Perspective**: SF ✓, WF ✗

**HalluMeasure** - "HalluMeasure: A Framework for Measuring Hallucination in Text Generation" (2023)
- **Data**: TechNewsSumm dataset from CNN/DM, SummEval
- **Model**: Claude
- **Method**: COT, Reasoning
- **Evaluation Metrics**: Precision, Recall, F1
- **Evaluation Perspective**: SF ✓, WF ✗

**EGH** - "EGH: Embedding-based Gradient Hallucination Detection" (2023)
- **Data**: HADES, HaluEval, SelfcheckGPT datasets
- **Model**: LLaMa2, OPT, GPT-based models
- **Method**: Taylor expansion on embedding difference
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1, AUC, G-Mean, BSS
- **Evaluation Perspective**: SF ✓, WF ✓

**STARE** - "STARE: Sentence-level Translation Accuracy and Reliability Evaluation" (2023)
- **Data**: LfaN-Hall, HalOmi datasets
- **Model**: COMET-QE, LASER, XNLI and LaBSE
- **Method**: Aggregate hallucination scores
- **Evaluation Metrics**: AUROC, FPR
- **Evaluation Perspective**: SF ✓, WF ✗

**HaluAgent** - "HaluAgent: A Multi-Agent Framework for Hallucination Detection" (2023)
- **Data**: HaluEval-QA, WebQA, Ape210K, HumanEval, WordCnt datasets
- **Model**: Baichuan2-Chat, GPT-4
- **Method**: Sentence Segmentation, Tool Selection and Verification, Reflection
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1
- **Evaluation Perspective**: SF ✓, WF ✓

**RefChecker** - "RefChecker: Reference-based Factual Consistency Evaluation" (2023)
- **Data**: KnowHalBench dataset from Natural Questions, MS MARCO, databricks-dolly15k
- **Model**: Mistral-7B, GPT-4, NLI
- **Method**: Extractor and Checker
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1
- **Evaluation Perspective**: SF ✓, WF ✓

**HDM-2** - "HDM-2: Hierarchical Detection Model for Hallucination Evaluation" (2023)
- **Data**: HDMBENCH dataset from RAGTruth, enterprise support tickets, MS Marco, SQuAD, Red Pajama v2
- **Model**: Qwen-2.5-3B-Instruct
- **Method**: Classification
- **Evaluation Metrics**: Precision, Recall, F1
- **Evaluation Perspective**: SF ✓, WF ✓

**Lookback Lens** - "Lookback Lens: Attention-based Hallucination Detection in Text Generation" (2024)
- **Data**: CNN/DM, XSum, Natural Questions, MT-Bench datasets
- **Model**: LLaMA-2-7B-Chat, GPT-based models
- **Method**: Attention Map analysis
- **Evaluation Metrics**: AUROC, EM
- **Evaluation Perspective**: SF ✓, WF ✓

**KnowHalu** - "KnowHalu: A Knowledge-Aware Framework for Hallucination Detection" (2024)
- **Data**: HaluEval, HotpotQA, CNN/DM datasets
- **Model**: Starling-7B, GPT-3.5
- **Method**: Identify non-fabrication, multi-form fact-checking
- **Evaluation Metrics**: TPR, TNR, Average Accuracy
- **Evaluation Perspective**: SF ✓, WF ✓

**AXCEL** - "AXCEL: Adaptive Cross-domain Evaluation for Language Models" (2024)
- **Data**: SummEval, QAGS datasets
- **Model**: Llama-3-8B, Claude-Haiku, Claude-Sonnet
- **Method**: Direct Assessment
- **Evaluation Metrics**: Precision, Recall, F1, AUC
- **Evaluation Perspective**: SF ✓, WF ✓

**Drowzee** - "Drowzee: A Framework for Detecting Hallucinations in Question Answering" (2024)
- **Data**: Drowzee dataset
- **Model**: GPT-3.5-turbo, GPT-4, Llama2-7B, 70B, Mistral-7B-v0.2, 8x7B
- **Method**: Direct Assessment
- **Evaluation Metrics**: FCH Ratio
- **Evaluation Perspective**: SF ✗, WF ✓

**MIND** - "MIND: Model-based Inference for Neural Detection of Hallucinations" (2024)
- **Data**: HELM dataset
- **Model**: MLP
- **Method**: Embedding MLP classification
- **Evaluation Metrics**: AUC, Pearson correlation
- **Evaluation Perspective**: SF ✗, WF ✓

**BTProp** - "BTProp: Bayesian Tree Propagation for Hallucination Detection" (2024)
- **Data**: Wikibio-GPT3, FELM-Science, FactCheckGPT datasets
- **Model**: gpt-3.5-turbo, Llama3-8B Instruct
- **Method**: Hidden Markov tree
- **Evaluation Metrics**: AUROC, AUC-PR, F1, Accuracy
- **Evaluation Perspective**: SF ✗, WF ✓

**FAVA** - "FAVA: Fine-grained Analysis of Visual Attention for Hallucination Detection" (2024)
- **Data**: FAVABENCH dataset from Open prompts
- **Model**: Llama2-Chat 7B
- **Method**: Hallucination tags generation
- **Evaluation Metrics**: F1
- **Evaluation Perspective**: SF ✗, WF ✓

**Semantic Entropy** - "Semantic Entropy: A Framework for Detecting Hallucinations in Language Models" (2024)
- **Data**: BioASQ, TriviaQA, NQ Open, SQuAD datasets
- **Model**: LLaMA 2 Chat-7B, 13B, 70B, Falcon Instruct-7B, 40B, Mistral Instruct-7B
- **Method**: Semantic Entropy
- **Evaluation Metrics**: AUROC, AURAC
- **Evaluation Perspective**: SF ✗, WF ✓

**SEPs** - "SEPs: Semantic Entropy Probes for Hallucination Detection" (2024)
- **Data**: BioASQ, TriviaQA, NQ Open, SQuAD datasets
- **Model**: Llama-2-7B, 70B, Mistral-7B, Phi-3-3.8B
- **Method**: Semantic Entropy Probes
- **Evaluation Metrics**: AUROC
- **Evaluation Perspective**: SF ✗, WF ✓

**HaloScope** - "HaloScope: A Comprehensive Framework for Hallucination Detection" (2024)
- **Data**: TruthfulQA, TriviaQA, CoQA, TydiQA-GP datasets
- **Model**: LLaMA-2-chat-7B, 13B, OPT6.7B, 13B
- **Method**: Unsupervised learning
- **Evaluation Metrics**: AUROC, BLUERT, ROUGE
- **Evaluation Perspective**: SF ✗, WF ✓

**LRP4RAG** - "LRP4RAG: Layer-wise Relevance Propagation for RAG-based Hallucination Detection" (2024)
- **Data**: RAGTruth dataset
- **Model**: Llama-2-7B/13B-chat
- **Method**: Internal state classification
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1
- **Evaluation Perspective**: SF ✓, WF ✓

**Halu-J** - "Halu-J: A Japanese Hallucination Detection Framework" (2024)
- **Data**: ME-FEVER dataset from FEVER
- **Model**: GPT-4, Mistral-7B-Instruct
- **Method**: Reasoning
- **Evaluation Metrics**: Accuracy
- **Evaluation Perspective**: SF ✗, WF ✓

**NonFactS** - "NonFactS: A Dataset for Non-Factual Summarization Detection" (2023)
- **Data**: NonFactS dataset from CNN/DM
- **Model**: BART-base, RoBERTa, ALBERT
- **Method**: NLI
- **Evaluation Metrics**: Balanced Accuracy
- **Evaluation Perspective**: SF ✓, WF ✗

**MFMA** - "MFMA: Multi-Faceted Model Analysis for Hallucination Detection" (2023)
- **Data**: CNN/DM, XSum datasets
- **Model**: BART-base, T5-small, Electra-base-discriminator
- **Method**: Classification
- **Evaluation Metrics**: F1, Balanced Accuracy
- **Evaluation Perspective**: SF ✓, WF ✗

**HADEMIF** - "HADEMIF: Hidden State Calibration for Hallucination Detection" (2024)
- **Data**: Response data
- **Model**: Llama2-7B
- **Method**: Hidden state calibration
- **Evaluation Metrics**: Expected Calibration Error, Brier Score
- **Evaluation Perspective**: SF ✗, WF ✓

**REDEEP** - "REDEEP: Retrieval-Augmented Deep Evaluation for Hallucination Detection" (2024)
- **Data**: RAGTruth, Dolly (AC) datasets
- **Model**: Llama2-7B/13B/70B, Llama3-8B
- **Method**: External Context Score, Parametric Knowledge Score
- **Evaluation Metrics**: AUC, PCC, Accuracy, Recall, F1
- **Evaluation Perspective**: SF ✗, WF ✓

**LMvLM** - "LMvLM: Language Model versus Language Model for Hallucination Detection" (2024)
- **Data**: LAMA, TriviaQA, NQ, PopQA datasets
- **Model**: ChatGPT, text-davinci-003, Llama-7B
- **Method**: LMs multi-turn judge
- **Evaluation Metrics**: Precision, Recall, F1
- **Evaluation Perspective**: SF ✗, WF ✓

**OnionEval** - "OnionEval: A Layered Evaluation Framework for Hallucination Detection" (2024)
- **Data**: OnionEval dataset
- **Model**: SLLMs (Llama, Qwen, Gemma)
- **Method**: Layered Evaluation
- **Evaluation Metrics**: Accuracy, Context-influence Score
- **Evaluation Perspective**: SF ✓, WF ✗




### Before LLM Era

**Fact_acc** - "Evaluating the Factual Consistency of Abstractive Text Summarization" (2020)
- **Data**: WikiFact dataset from Wikipedia and Wikidata KB
- **Model**: Transformer
- **Method**: Triplet extraction approach
- **Evaluation Metrics**: Precision, Recall, F1
- **Evaluation Perspective**: SF ✓, WF ✗

**FactCC** - "FactCC: Evaluating the Factual Consistency of Abstractive Text Summarization" (2020)
- **Data**: CNN/DM, XSumFaith datasets
- **Model**: BERT
- **Method**: NLI (2-class classification)
- **Evaluation Metrics**: Likelihood
- **Evaluation Perspective**: SF ✓, WF ✗

**DAE (Dependency Arc Entailment)** - "DAE: Dependency Arc Entailment for Factual Consistency Evaluation" (2021)
- **Data**: PARANMT50M dataset
- **Model**: ELECTRA
- **Method**: NLI (2-class classification)
- **Evaluation Metrics**: Likelihood
- **Evaluation Perspective**: SF ✓, WF ✗

**MaskEval** - "MaskEval: Weighted Masking for Summarization Evaluation" (2021)
- **Data**: CNN/DM, WikiLarge, ASSET datasets
- **Model**: T5
- **Method**: Word weighting approach
- **Evaluation Metrics**: Weighted match score
- **Evaluation Perspective**: SF ✓, WF ✗

**Looking for a Needle in a Haystack: A Comprehensive Study of Hallucinations in Neural Machine Translation** (2023)
- **Data**: WMT2018 DE-EN dataset
- **Model**: Transformer
- **Method**: Uncertainty measure
- **Evaluation Metrics**: Average similarity
- **Evaluation Perspective**: SF ✓, WF ✗

**Detecting Hallucinations in Neural Machine Translation with Model Uncertainty** (2023)
- **Data**: Haystack dataset
- **Model**: Transformer
- **Method**: Source contribution analysis
- **Evaluation Metrics**: Percentage
- **Evaluation Perspective**: SF ✓, WF ✗

**FEQA** - "FEQA: A Question Answering Evaluation Framework for Faithfulness Assessment in Abstractive Summarization" (2020)
- **Data**: CNN/DM, XSum datasets
- **Model**: BART (QG), BERT (QA)
- **Method**: Question generation and answering (QG-QA)
- **Evaluation Metrics**: Average F1
- **Evaluation Perspective**: SF ✓, WF ✗

**QAGS** - "QAGS: Question Generation and Answering for Summarization Evaluation" (2020)
- **Data**: CNN/DM, XSum datasets
- **Model**: BART (QG), BERT (QA)
- **Method**: QG-QA with entity and noun phrase focus
- **Evaluation Metrics**: Average similarity
- **Evaluation Perspective**: SF ✓, WF ✗

**QuestEval** - "QuestEval: Summarization Asks for Fact-based Evaluation" (2021)
- **Data**: CNN/DM, XSum datasets
- **Model**: T5 (QG, QA)
- **Method**: QG-QA with entity and noun focus
- **Evaluation Metrics**: Precision, Recall, F1
- **Evaluation Perspective**: SF ✓, WF ✗

**QAFactEval** - "QAFactEval: Improved QA-Based Factual Consistency Evaluation for Summarization" (2022)
- **Data**: SummaC dataset
- **Model**: BART (QG), ELECTRA (QA)
- **Method**: QG-QA and NLI combination
- **Evaluation Metrics**: LERC
- **Evaluation Perspective**: SF ✓, WF ✗

**MQAG** - "MQAG: Multiple-choice Question Generation for Summarization Evaluation" (2022)
- **Data**: QAGS, XSumFaith, Podcast, Assessment, SummEval datasets
- **Model**: T5 (QG), Longformer (QA)
- **Method**: Multi-choice question answering
- **Evaluation Metrics**: Choice statistical distance
- **Evaluation Perspective**: SF ✓, WF ✗

**CoCo** - "CoCo: Counterfactual Contrast for Evaluating and Improving Factual Consistency" (2022)
- **Data**: QAGS, SummEval datasets
- **Model**: BART
- **Method**: Counterfactual estimation
- **Evaluation Metrics**: Average likelihood difference
- **Evaluation Perspective**: SF ✓, WF ✗

**FactGraph** - "FactGraph: Evaluating Factual Consistency via Graph-based Fact Verification" (2021)
- **Data**: CNN/DM, XSum datasets
- **Model**: ELECTRA
- **Method**: Classification approach
- **Evaluation Metrics**: Balanced accuracy, F1
- **Evaluation Perspective**: SF ✓, WF ✗

**FactKB** - "FactKB: Knowledge Base-based Factual Consistency Evaluation" (2021)
- **Data**: CNN/DM, XSum datasets
- **Model**: RoBERTa
- **Method**: Classification approach
- **Evaluation Metrics**: Balanced accuracy, F1
- **Evaluation Perspective**: SF ✓, WF ✗

**ExtEval** - "ExtEval: Extrinsic Evaluation of Summarization Models" (2021)
- **Data**: CNN/DM dataset
- **Model**: SpanBERT, RoBERTa
- **Method**: Direct prediction and statistical analysis
- **Evaluation Metrics**: Summation of sub-scores
- **Evaluation Perspective**: SF ✓, WF ✗

**Q²** - "Q²: Evaluating Factual Consistency in Knowledge-Grounded Dialogue via Question Generation and Question Answering" (2021)
- **Data**: Wizard of Wikipedia (WOW) dataset
- **Model**: T5 (QG), Albert-Xlarge (QA), RoBERTa (NLI)
- **Method**: QG-QA and NLI combination
- **Evaluation Metrics**: Likelihood
- **Evaluation Perspective**: SF ✗, WF ✓

**FactPush** - "FactPush: A Factual Consistency Evaluation Framework for Dialogue Systems" (2022)
- **Data**: TRUE dataset
- **Model**: DeBERTa
- **Method**: NLI approach
- **Evaluation Metrics**: AUC
- **Evaluation Perspective**: SF ✓, WF ✗

**AlignScore** - "AlignScore: Evaluating Factual Consistency with a Unified Alignment-based Framework" (2022)
- **Data**: 22 datasets from 7 tasks
- **Model**: RoBERTa
- **Method**: 3-way classification
- **Evaluation Metrics**: Likelihood
- **Evaluation Perspective**: SF ✓, WF ✗

**WeCheck** - "WeCheck: Weakly Supervised Factual Consistency Evaluation" (2022)
- **Data**: TRUE dataset
- **Model**: DeBERTaV3
- **Method**: Weakly supervised NLI
- **Evaluation Metrics**: Likelihood
- **Evaluation Perspective**: SF ✓, WF ✗









