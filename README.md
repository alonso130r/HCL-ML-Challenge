# Human-Computer Lab Challenge

### Primary track

This demo uses the **text + audio** track. Video was excluded after achieving
0.3515 weighted F1 as a standalone signal while adding decoding and
face-processing complexity that did not fit the local, sub-6B-parameter scope.

The interface records audio in the browser, transcribes it locally with a
small FunASR model, and places the transcript in the composer for editing
before sending. The original audio is retained for local emotion analysis.

## Local transcription model

The demo uses local FunASR transcription with `ct-punc`; place a compatible
model in `models/funasr-small` or pass `--transcription-model` to override the
default cached `paraformer-zh` model.

## Run the demo

Requirements: Python 3 with `venv`, Git LFS, CMake, and a C++ compiler. The first run also requires a network connection to retrieve dependencies and public model assets.

Run:

```bash
./run_app.sh
```

The script initializes the `llama.cpp` submodule when required, retrieves Git LFS weights, creates `.venv`, installs Python dependencies, downloads the base chat and emotion models, builds `llama-server`, and starts the app. It accepts the same options as the Python app, for example `./run_app.sh --port 8001`.

### My definition of "real-time"

For this prototype, real time means that a user can submit one recorded turn,
review its transcript, and begin receiving a streamed reply without waiting for
the complete response. Startup, memory, and warm-response measurements are
reported below.

### Model architecture and design process

I had a lot more fun working through model selection and training for this
specific dataset than I expected.

The architecture of the final system takes text and audio inputs to a MELD emotion classifier, which returns structured interface metadata: the fused MELD emotion category and confidence, text-only and audio-only predictions, the audio-mixture weight, and audio-induced logit change. The fused emotion label is also supplied to the response LLM, a Qwen instruction-tuned model, so that it can adjust its tone. The user receives a short streamed response grounded in the submitted text and audio-derived state.

The transcript is editable before it reaches the text branch, while the audio
branch always analyzes the original recording. This keeps speech-recognition
correction separate from acoustic-affect analysis.

#### Research framing before implementation

Before building the system, I reviewed recent MELD work to understand where
meaningful gains were coming from. The common pattern was not a slightly larger
fusion head. Stronger systems relied on better supervision and representation
learning, including contrastive objectives, pseudo-labels, knowledge
distillation, explicit context or emotion-cause structure, and sometimes
active-speaker processing. Examples include
[CEPT](https://aclanthology.org/2024.lrec-main.263/),
[DCLF](https://aclanthology.org/2025.coling-main.272/),
[ECERC](https://aclanthology.org/2025.acl-long.102/), and
[Xiao et al.](https://aclanthology.org/2026.findings-eacl.212/).

In other words, adding audio or video to an already strong text model was not
enough by itself to produce the larger reported gains. Those gains appeared
when multimodal inputs were paired with additional supervision or structure.
This project therefore treats a small, demonstrably useful audio contribution
as a more defensible goal than assuming fusion alone will close the gap.

That led to three decisions before modeling: use contextual text as the
reliable anchor, treat audio as an additional signal that must prove its value,
and avoid a video pipeline whose cost could not be justified by the available
time or local-inference budget. The evaluation therefore emphasizes
counterfactual audio and context controls, rather than reporting headline F1
alone.

#### Full-frame dual audio attention

The model retains all valid `emotion2vec` frames, rather than collapsing audio
to one summary vector. It uses a BERT-conditioned query and a learned
audio-only query, then fuses both attended representations with the acoustic
features before correcting the frozen text logits. The first 32-frame,
single-query version achieved 0.6304 weighted F1 and a +0.0015
shuffled-audio margin. Full-frame attention raised those values to 0.6344 and
+0.0053; independent audio warm-up and dual attention reached 0.6354 weighted
F1, 0.4817 macro F1, +0.0102 zero-audio margin, and +0.0064 shuffled-audio
margin. The audio frame pooler, projection, and classifier are warmed up before
joint fusion so that the stronger text branch cannot immediately dominate the
optimization.

The 197-dimensional acoustic vector includes eGeMAPS and prosody features with
speaker-relative normalization computed only from preceding turns. Current
pitch and loudness are therefore compared with that speaker's available prior
baseline without using future dialogue information.

#### Evidence and validation

Contextual BERT set a strong text baseline at 0.6291 weighted F1. Audio was
weaker alone but informative, while video was both weaker and operationally
costlier, so the submission uses text and audio only. A fused model is accepted
only if it loses performance when matched audio is shuffled or zeroed, and when
dialogue state is reset. This rejects models that raise F1 by adding a larger
head while ignoring the intended modality.

#### Causal context and selected result

The recurrent correction is constructed to be zero without historical state:

\[
r_t = f(x_t, d_{t-1}, s_{t-1}) - f(x_t, 0, 0)
\]

This prevents recurrence from acting as another feed-forward classifier on the
current turn. A two-turn text window gave the best balance between direct text
context, recurrent history, and measurable audio contribution. Across five
seeds, the selected context-two architecture produced:

| Metric | Mean | Standard deviation |
| --- | ---: | ---: |
| Weighted F1 | 0.6339 | 0.0025 |
| Macro F1 | 0.4751 | 0.0035 |
| Matched-audio margin | +0.0077 | 0.0013 |
| Zero-audio margin | +0.0058 | 0.0018 |
| Reset-state margin | +0.0030 | 0.0019 |

The selected checkpoint was seed 43 at epoch 12, chosen using development
evidence only: 0.6386 development weighted F1, 0.5232 development macro F1,
+0.0101 state margin, and +0.0200 audio margin. Seed 44 had the highest test
score, but selecting it would have leaked test-set information.

These five-seed results describe the stabilized context-two research model. The
interactive demo loads the fixed-epoch frame-attention checkpoint at
`models/frame-attention/`; it is an
interactive prototype, not a final validated deployment checkpoint.

### Interpreting the metrics

The most important negative result was checkpoint reconstruction. Six
dialogue-level cross-validation runs met the text, state, zero-audio, and
shuffled-audio eligibility controls, with mean weighted F1 of 0.6368. Training
one final model for the median selected epoch count reduced weighted F1 to
0.6299 and produced negative state and shuffled-audio margins, even as training
loss continued to fall. The experiment shows that a median training duration
does not reconstruct a valid checkpoint when correction gates keep changing.

The checkpoint used by the demo therefore remains an interactive prototype,
not a final validated deployment model. Its stronger audio influence can make
spoken affect more noticeable in use, but that interaction property is not a
claim of better aggregate MELD performance.

## Local model budget, hardware, and external components

For the default local configuration measured here, the complete inference path contains **2,454,493,458 learned parameters**, leaving **3,545,506,542 parameters** below the 6-billion limit. The count is derived from the tensors in the exact loaded artifacts:

| Component | Learned parameters |
| --- | ---: |
| Qwen3-1.7B Q8_0 GGUF | 1,720,574,976 |
| Fine-tuned BERT text classifier | 109,487,623 |
| `emotion2vec_plus_base` | 93,178,133 |
| FunASR Paraformer ASR | 247,339,018 |
| FunASR `ct-punc` | 281,365,842 |
| Frame-attention fusion checkpoint | 2,547,866 |
| **Total** | **2,454,493,458** |

Supplying a different transcription model with `--transcription-model` changes this total and must be counted separately.

Development and local testing were performed on a MacBook Pro with an Apple M4 Pro, 14 CPU cores, and 24 GB unified memory. The launcher selects Metal on macOS, CUDA when available on supported NVIDIA systems, and CPU otherwise.

Measured on that MacBook Pro with the Metal path, built binaries, installed dependencies, and model files already cached:

| Measurement | Observed result |
| --- | ---: |
| Clean process start to local HTTP readiness | 37 s |
| Python interface resident memory after readiness | 2.06 GiB |
| `llama-server` resident memory after readiness | 1.98 GiB |
| Combined resident memory after readiness | 4.04 GiB |
| First streamed token, first short text-only request | 69 ms |
| First streamed token, subsequent short text-only requests | 13 ms |
| Complete short text-only response, subsequent requests | 93 ms |

The response measurement used three sequential requests to the local `/api/chat` endpoint, each producing a 34-character reply. It measures the warm text-to-response path, not browser recording, transcription, or full audio-emotion inference. The 37-second startup measurement excludes the one-time virtual-environment creation, dependency installation, model downloads, and `llama.cpp` build.

Pretrained components are Qwen3-1.7B, `bert-base-uncased`,
`emotion2vec_plus_base`, FunASR `paraformer-zh`, and FunASR `ct-punc`.
MELD supplies training and evaluation data, and `llama.cpp` serves local Qwen
inference. The submission owns the integration, training, evaluation, and
reported results; external licenses and terms remain applicable.

## Evaluation workflow

1. Run `./run_app.sh` and open `http://127.0.0.1:8000`.
2. Record a short message in the browser.
3. Review or edit the locally generated transcript.
4. Send the message and observe the streamed response.

## Privacy

Audio recording, transcription, emotion inference, and language-model generation run on the local machine. The browser does not use a hosted speech-recognition service or a hosted LLM API.

## Reproducibility

The interactive demo downloads its required public model assets on its first run and retrieves the fine-tuned demo weights with Git LFS. Reproducing model training requires the MELD archives, which are not tracked in Git. The training and benchmarking scripts are retained under `research/experiments/` and `research/benchmarking/`.

### Reproducing the experiments

The MELD raw archive must be available at `data/MELD/MELD.Raw.tar.gz`. The scripts use their checked-in defaults and write results beside the named output directories unless an explicit output path is provided.

| Goal | Script | Result or purpose |
| --- | --- | --- |
| Train the contextual text baseline | `research/experiments/train_text.py` | Fine-tunes the BERT text classifier and writes to `research/experiments/training-output-text/`. |
| Reproduce the five-seed context-two study | `research/experiments/train_recurrent_dialogue_stabilized.py --context-window 2 --runs 5` | Produces the repeated-seed text-audio recurrent evaluation reported above. |
| Run leakage-safe architecture evaluation | `research/benchmarking/run_final_architecture_cv.py --retrain` | Retrains text models inside dialogue-level outer folds and reports out-of-fold metrics. |

The interactive submission is reproduced with `./run_app.sh`. Advanced
frame-attention and distillation experiments are in
`research/experiments/train_recurrent_dialogue_frame_attention.py`,
`research/benchmarking/run_final_frame_attention.py`, and
`research/benchmarking/distill_frame_attention_ensemble.py`.

## Limitations

- The emotion model was trained on MELD, an acted television-dialogue dataset, and may not generalize to everyday conversation. RLHF/RLVF could help with this, but a gathering/creating a suitable dataset would not have been possible
in the given timeframe.
- The classifier is not a clinical, diagnostic, or general-purpose emotion-assessment system.
- CPU inference is supported but may be slower than CUDA or Metal.
