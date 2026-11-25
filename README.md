# TweetLens

TweetLens examines how short English tweets express sentiment and which text spans carry that sentiment. It combines reproducible Python experiments with an interactive English and Traditional Chinese interface.

Developed by Junyan Deng in collaboration with OpenAI Codex.

[Open the interactive demonstration](https://mindy0932.github.io/TweetLens/)

## Research questions

- How well can sentiment classifiers distinguish negative, neutral and positive tweets?
- How does a frozen RoBERTa representation compare with TF-IDF baselines?
- How does predicting sentiment first affect the extraction of annotated sentiment spans?

## Data and evaluation

The Kaggle Tweet Sentiment Extraction training data contain 27,481 records. Removing one missing record and 68 duplicate text-label records leaves 27,412 samples. Normalized texts are grouped before splitting, producing 21,931 training samples and 5,481 validation samples with no normalized-text overlap. The random seed is 42.

| Classifier | Validation accuracy | Macro-F1 |
| --- | ---: | ---: |
| Majority-class baseline | 39.48% | 0.189 |
| TF-IDF + logistic regression | 68.97% | 0.691 |
| TF-IDF + linear SVM | 67.47% | 0.677 |
| Frozen RoBERTa + logistic regression head | 65.88% | 0.660 |

RoBERTa uses pretrained FacebookAI/roberta-base weights, masked mean pooling and a maximum length of 64 tokens. Its encoder remains fixed while the logistic regression head is trained. Full-parameter fine-tuning has a separate entry point and has no reported result in this project.

The original lexical span extractor scores 0.618 mean Jaccard with true sentiment labels, compared with 0.583 for returning the full text. Using predicted sentiment reduces the complete pipeline to 0.523. These results show the effect of upstream classification errors.

These are local validation results. Further tuning should use a separate final test set. The dataset does not support claims about the whole platform or engagement trends.

## Interactive application

The interface includes topic projection, sentiment composition, switchable model comparisons, confusion-matrix inspection, sample filtering and CSV export. Chart clicks reveal previews and scroll to the result. Dedicated detail buttons preserve a return path to the chart.

New-text RoBERTa inference runs in a browser Web Worker with ONNX Runtime Web. First use downloads about 125 MB of model weights plus the runtime. Chunks are checked with SHA-256 and cached. The browser export uses the fitted classification head. Quantization and runtime arithmetic can produce different probabilities from the Python experiment. Saved validation comparisons and replayed predictions refer to the Python results.

## Code entry points

| File | Purpose |
| --- | --- |
| train.py | Clean data, create grouped splits, train baselines and evaluate span extraction |
| src/modeling.py | Lexical span extractor and word-set Jaccard metric |
| roberta_frozen.py | Train and evaluate the frozen-RoBERTa classifier |
| train_roberta.py | Optional full-parameter training entry point |
| export_browser_roberta.py | Convert the fitted classifier to chunked ONNX browser assets |
| browser/worker.js | Browser tokenization, verified model loading, caching and inference |
| reports/chart_interactions.html | Chart previews, filters and result navigation |
| reports/locale.html | Language switching for static and dynamic interface text |
| build_extended.py | Rebuild the bilingual showcase using measured results |
| artifacts/ | Split manifest, metrics, fitted heads and sample-level predictions |

## Run the browser application

From the repository root, run the following command and open the localhost URL.

    python -m http.server 8000

    http://localhost:8000/

The static page includes the browser runtime and model chunks. A Python inference API is also available through serve_showcase.py for local development.

## Reproduce the experiments

Use Python 3.12. Install the requirements, obtain the data from the official Kaggle competition and place train.csv, test.csv and sample_submission.csv under data/.

    python -m pip install -r requirements.txt -r requirements-roberta.txt
    python train.py
    python roberta_frozen.py
    python build_extended.py

For an ONNX export, install onnx and onnxruntime, then run export_browser_roberta.py. On Windows, its --temp-directory option accepts an ASCII-only temporary path when required by the native ONNX tooling.

To rebuild the worker, install the dependencies in browser/ from pnpm-lock.yaml and run node build.cjs from that directory.

## Sources and licenses

- Data come from [Kaggle Tweet Sentiment Extraction](https://www.kaggle.com/competitions/tweet-sentiment-extraction). Raw CSV files are excluded from Git.
- Pretrained weights come from [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base).
- New project code is MIT-licensed. Bundled third-party notices are under assets/licenses/.
