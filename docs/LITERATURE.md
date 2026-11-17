# Literature

Only works that the implementation actually uses or compares against are listed, each with
what it contributed. Bibliographic details are given as far as they are established; where a
detail is uncertain it is marked rather than guessed.

## AI-generated text detection

**Sadasivan, V. S., Kumar, A., Balasubramanian, S., Wang, W., & Feizi, S. (2023). Can AI-Generated
Text be Reliably Detected?** *Findings of ACL 2023* (arXiv:2303.11156).
The theoretical argument that detection is fragile: as model output approaches the human
distribution, any detector's advantage shrinks. This is the reason Veritas reports intervals,
an uncertain band, and robustness rather than a single confident label.

**Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., & Finn, C. (2023). DetectGPT: Zero-Shot
Machine-Generated Text Detection using Probability Curvature.** *ICML 2023* (arXiv:2301.11305).
Introduced the curvature statistic: machine text sits in a local maximum of the model's own
log-likelihood, so random perturbations tend to *increase* its probability. Implemented as the
`fast_detectgpt` component (GPU only).

**Bao, G., Zhao, Y., Teng, Z., Yang, L., & Zhang, Y. (2023). Fast-DetectGPT: Efficient Zero-Shot
Detection of Machine-Generated Text via Conditional Probability Curvature.** arXiv:2310.05130.
Replaces DetectGPT's random perturbations with a single sampling pass, making the curvature
statistic cheap enough to run. This is the variant used in the `deep` mode.

**Hans, A., Schwarzschild, A., Cherepanova, V., Kazemi, H., Saha, A., Goldblum, M., Geiping, J.,
& Goldstein, T. (2024). Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated
Text.** *ICML 2024* (arXiv:2401.12070).
The ratio of two related models' perplexities separates human from machine text better than
either alone. Implemented as the `binoculars` component in `standard` mode.

**Guo, B., Zhang, X., Wang, Z., Jiang, M., Nie, J., Ding, Y., Yue, J., & Wu, Y. (2023). How Close is
ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection.** arXiv:2301.07597.
Introduces **HC3**, the human–ChatGPT paired corpus used as the primary training and
evaluation data here (five domains: Reddit ELI5, finance, medicine, open QA, wiki).

**Wang, Y., Mansurov, J., Ivanov, P., Su, J., Shelmanov, A., Tsvigun, A., et al. (2024). M4:
Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection.**
*EACL 2024* (arXiv:2305.14902).
The reference point for cross-generator evaluation. Veritas implements
leave-one-generator-out (`ml/eval_cross_generator.py`) and reports honestly that with a single
generator family in the data the experiment is not informative.

**Kumarage, T., et al. (2023). Stylometric Detection of AI-Generated Text in Twitter Timelines.**
arXiv:2303.03697.
Motivates the stylometric features (sentence-length variance, punctuation rhythm,
function-word rates) that make the CPU-only components competitive.

**Solaiman, I., Brundage, M., Clark, J., Askell, A., Herbert-Voss, A., Wu, J., et al. (2019). Release
Strategies and the Social Impacts of Synthetic Text.** arXiv:1908.09203.
The early framing of detection as a *mitigation*, not a solution — cited in the report's
ethics section.

## Plagiarism and similarity

**Schleimer, S., Wilkerson, D. S., & Aiken, A. (2003). Winnowing: Local Algorithms for Document
Fingerprinting.** *SIGMOD 2003*, 76–85.
The fingerprinting scheme used for both text and source code: k-gram hashes, minimum in each
window, positions retained so matches can be highlighted precisely.

**Broder, A. Z. (1997). On the Resemblance and Containment of Documents.** *Compression and
Complexity of Sequences 1997*, 21–29.
MinHash and the asymmetric *containment* measure (|A ∩ B| / |A|) that makes "how much of this
document came from that source" well defined.

**Charikar, M. S. (2002). Similarity Estimation Techniques from Rounding Algorithms.** *STOC 2002*.
SimHash; used inside the LSH index for near-duplicate retrieval.

**Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval.**
Cambridge University Press, ch. 3 and 8.
The inverted-index and TF-IDF background the lexical stage relies on.

**Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese
BERT-Networks.** *EMNLP 2019* (arXiv:1908.10084).
Sentence embeddings for the paraphrase stage. Optional: when `sentence-transformers` is
unavailable the engine reports paraphrase retrieval as unavailable instead of silently
lowering recall.

**Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs.**
*IEEE Transactions on Big Data* (arXiv:1702.08734).
FAISS; the dense index behind paraphrase candidate retrieval.

**Alzahrani, S. M., Salim, N., & Abraham, A. (2012). Understanding Plagiarism Linguistic Patterns,
Textual Features, and Detection Methods.** *IEEE Transactions on Systems, Man, and Cybernetics
Part C*, 42(2), 133–149.
Taxonomy of plagiarism types (verbatim, light edit, reorder, paraphrase) that the synthetic
benchmark in `ml/make_plag_benchmark.py` is built around.

**Clough, P. (2003). Old and New Challenges in Automatic Plagiarism Detection.** *Proceedings of
the National Plagiarism Advisory Service Conference*.
The classic statement of why "similarity percentage" alone is misleading, and why span-level
evidence matters.

## Calibration and evaluation

**Platt, J. C. (1999). Probabilistic Outputs for Support Vector Machines and Comparisons to
Regularized Likelihood Methods.** In *Advances in Large Margin Classifiers*, MIT Press.
The sigmoid fit used to turn stacker decisions into probabilities.

**Niculescu-Mizil, A., & Caruana, R. (2005). Predicting Good Probabilities with Supervised
Learning.** *ICML 2005*.
Why raw classifier scores are not probabilities, and why calibration must be fitted on data
the classifier did not train on.

**Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural
Networks.** *ICML 2017* (arXiv:1706.04599).
Expected calibration error, reported per mode in `metrics.json`.

**Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap.** Chapman & Hall.
The bootstrap confidence intervals around ROC-AUC and around the bias difference.

**Saito, T., & Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC
Plot When Evaluating Binary Classifiers on Imbalanced Datasets.** *PLOS ONE*, 10(3).
Why TPR at a fixed false-positive rate is the headline metric here rather than accuracy.

**Liang, W., Yuksekgonul, M., Mao, Y., Wu, E., & Zou, J. (2023). GPT Detectors Are Biased Against
Non-Native English Writers.** *Patterns*, 4(7) (arXiv:2304.02819).
The source of the bias check. Veritas reproduces the *design* (compare human false-positive
rates between ordinary text and simplified text) with a clearly labelled proxy, because it has
no licensed ESL corpus.

## Systems and tooling

**Rabiner, L. R. (1989). A Tutorial on Hidden Markov Models and Selected Applications in Speech
Recognition.** *Proceedings of the IEEE*, 77(2), 257–286.
The two-state HMM (Viterbi decoding) behind mixed-authorship segmentation.

**McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection
for Dimension Reduction.** arXiv:1802.03426.
The 3D projection on the Evaluation page; falls back to PCA when `umap-learn` is absent.

**Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K.
(2021). Datasheets for Datasets.** *Communications of the ACM*, 64(12).
The model for `docs/REPORT.md` §Datasets: what is in the data, what is not, and how it was
collected.

## What is deliberately *not* used

* No commercial detection or plagiarism API (Turnitin, GPTZero, Copyleaks, OpenAI) is called.
  The single third-party model referenced as a baseline is an open checkpoint evaluated
  offline; when it cannot be downloaded, no number is reported.
* No claim of parity with commercial services, and no citation that was not read when writing
  this document.
