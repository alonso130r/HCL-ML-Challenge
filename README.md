# Human-Computer Lab Challenge

### Primary track

This demo will be **text + audio**. Why? Low-latency video processing is
 extremely difficult to accomplish under the time and hardware constraints
 (2-3 days of work, <6B params, local inference). Staying with the
philosophy that a simpler, more useful product is better than a complex,
half-working product also points towards audio as extracting emotion from
video is a lot more complex than audio and more likely to be less effective.

The interface records audio in the browser, transcribes it locally with a
small FunASR model, and places the transcript in the composer for editing
before sending. The original audio is retained for local emotion analysis.

## Local transcription model

Place a small FunASR-compatible model in `models/funasr-small`, or pass a
different directory with `--transcription-model`. If that directory does not
exist, startup automatically downloads and caches FunASR's `paraformer-zh`
model. No browser speech service is used.
The `ct-punc` FunASR model is also loaded automatically so transcripts include
basic punctuation.

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
the complete response. The system is designed for short conversational turns
and keeps inference models resident after startup. No numeric latency target is
claimed because formal timing measurements were not recorded.

### Model architecture and design process

I had a lot more fun working through model selection/training for this specific dataset then I expected.

Firstly, models that are only used for the demo itself: FunASR with ct-punc is used to locally transcribe
the user's inputs in the interface so they don't have to retype them in the text input box. Even with this,
the whole demo is still well below the 6B parameter limit.

The architecture of the final system takes text and audio inputs to a MELD emotion classifier, which returns structured interface metadata: the fused MELD emotion category and confidence, text-only and audio-only predictions, the audio-mixture weight, and audio-induced logit change. The fused emotion label is also supplied to the response LLM, a Qwen instruction-tuned model, so that it can adjust its tone. The user receives a short streamed response grounded in the submitted text and audio-derived state.

#### Deployed interactive pipeline

The running interface uses the following exact configuration:

| Stage | Implementation | Why it is included |
| --- | --- | --- |
| Transcription | FunASR `paraformer-zh` with `ct-punc` | Converts the browser recording to an editable local transcript. |
| Text branch | Fine-tuned `bert-base-uncased`; current transcript plus two preceding transcripts; 768-dimensional pooled embedding and seven-class logits | Contextual text was the strongest standalone MELD signal. |
| Audio branch | `emotion2vec_plus_base` complete 768-dimensional frame sequence plus 197 acoustic features from eGeMAPS, prosody, and speaker-relative normalization | Retains temporal acoustic evidence that summary-only audio features discarded. |
| Fusion | Text projection 256, audio projection 128, dialogue GRU state 128, speaker GRU state 64, text-conditioned and audio-only masked frame attention, and bounded residual gates | Combines modalities without replacing the strong frozen text prediction. |
| Response | Qwen3-1.7B Q8_0 through `llama.cpp`; maximum 160 tokens | Streams a concise local response whose tone is conditioned on the fused emotion label. |

The live checkpoint is `benchmarking/results/final-frame-attention-light/best-model/best_frame_attention.pt`. It permits stronger audio influence than the conservative research model, which makes spoken affect more noticeable during interaction. Its evaluation caveat is described below.

#### Full-frame dual audio attention

The main architectural contribution is not simply concatenating one audio
vector with text. Every utterance retains its complete `emotion2vec` frame
sequence, from 2 to 299 frames in the MELD experiments, with a median of 123.
The batch mask prevents padding from receiving attention weight.

Two attention queries pool this sequence into turn-level audio evidence:

1. A query projected from the contextual BERT embedding asks which acoustic
   frames are relevant to the text interpretation.
2. A learned audio-only query asks which frames are salient from the acoustic
   signal itself, even when they disagree with the text branch.

The model fuses both attended representations with the 197-dimensional
acoustic vector before producing an audio correction to the frozen text logits.
This design makes it possible for tone, energy, prosody, or a short emotional
event in the recording to influence the result without allowing noisy audio to
replace the text classifier.

This mechanism emerged from a concrete failure. The initial attention model
uniformly capped each utterance at 32 frames and used only the text-conditioned
query. It reached 0.6304 weighted F1 and a +0.0015 shuffled-audio margin.
Keeping all frames raised weighted F1 to 0.6344 and the shuffled-audio margin
to +0.0053. Independent audio warm-up and dual attention then produced a staged
checkpoint with 0.6354 weighted F1, 0.4817 macro F1, +0.0102 zero-audio margin,
and +0.0064 shuffled-audio margin. Those controls are important: they show that
the improvement was associated with matched acoustic evidence rather than a
larger fusion head.

#### 1. Text-only model

The text model started as a bert-base-uncased encoder which was fine-tuned alongside a classifier layer
mapping to MELD emotions. It uses the current utterance + a sliding context window.

| Model | Test macro F1 | Test weighted F1 |
| --- | ---: | ---: |
| Initial cached MLP | 0.4047 | 0.5663 |
| Context-enhanced cached model | 0.4233 | 0.5820 |
| Fine-tuned contextual BERT | 0.4775 | 0.6291 |

#### 2. Testing audio and video independently

Audio and video were tested as standalone signals before adding them to the text model.

| Modality experiment | Test macro F1 | Test weighted F1 |
| --- | ---: | ---: |
| Attentive audio model | 0.2019 | 0.4090 |
| WavLM audio model | 0.1008 | 0.3216 |
| Best audio diagnostic probe | 0.2824 | 0.4488 |
| Video model | 0.1597 | 0.3515 |

Audio contained useful but weaker emotion information than text. Video added substantial decoding and face-processing complexity while producing the weakest results, so it was excluded from the final demo.

#### 3. Counterfactual validation: proving that audio is used

The first text-audio fusion model reached a weighted F1 of 0.6419, but shuffling the audio produced 0.6442. This showed that the model was improving numerically without using the correct acoustic signal.

From that point, evaluation included counterfactual controls:

- matched versus shuffled audio;
- matched versus zeroed audio;
- normal dialogue state versus reset state.

A model was not considered successful only because its headline F1 improved.
For a matched-audio score \(F_{matched}\), the evaluation records:

\[
\Delta_{audio} = F_{matched} - F_{shuffled}, \qquad
\Delta_{zero} = F_{matched} - F_{zero-audio}
\]

and evaluates state dependence using \(F_{matched} - F_{reset-state}\).
Positive margins mean the model loses performance when the correct audio or
prior dialogue state is removed. This prevents a strong text branch from
masking an unused audio branch behind an apparently better headline F1.

#### 4. Causal recurrent dialogue context

The next design added causal recurrent state:

- a dialogue-level GRU to retain conversational context;
- a speaker-level GRU to retain speaker-specific history;
- bounded residual corrections to the frozen text classifier;
- audio features from emotion2vec frames, eGeMAPS, and prosody summaries.

The model predicts each turn from prior state before updating that state with
the current turn. Its context correction is defined as a strict residual:

\[
r_t = f(x_t, d_{t-1}, s_{t-1}) - f(x_t, 0, 0)
\]

where \(d_{t-1}\) and \(s_{t-1}\) are the preceding dialogue and speaker
states. The correction is exactly zero when historical state is removed. A
reset-state ranking objective then requires the true label to be better
supported by genuine history than by zero state. This avoids the common failure
mode in which a recurrent module behaves like another feed-forward classifier
that simply reuses the current utterance.

#### 5. Selecting the context-two text-audio model

Experiments with one, two, and three previous text turns showed that two previous turns gave the best balance:

- text retains enough immediate conversational context;
- recurrent state contributes longer dialogue and speaker history;
- audio remains measurably useful.

The selected architecture was evaluated across five random seeds:

| Metric | Mean | Standard deviation |
| --- | ---: | ---: |
| Weighted F1 | 0.6339 | 0.0025 |
| Macro F1 | 0.4751 | 0.0035 |
| Matched-audio margin | +0.0077 | 0.0013 |
| Zero-audio margin | +0.0058 | 0.0018 |
| Reset-state margin | +0.0030 | 0.0019 |

The final system therefore uses text as the strongest signal, while preserving a tested contribution from the recorded audio and prior dialogue state.

The five-seed results above describe the stabilized context-two recurrent architecture study. The interactive demo currently loads the fixed-epoch frame-attention checkpoint stored in `benchmarking/results/final-frame-attention-light/best-model/`. That checkpoint is included as an interactive prototype, not claimed as a final validated deployment checkpoint: the final reconstruction did not preserve the validated state and matched-audio margins.

### Interpreting the metrics

The aggregate MELD metrics are useful for comparing architectures, but they do not fully describe the interactive experience. The fixed-epoch frame-attention checkpoint gives the audio branch a stronger influence than the conservative research model reported above. As a result, it can feel more accurate and responsive when a person speaks with clearly audible affect, even when its aggregate MELD F1 is not higher. This is an interaction trade-off, not a claim of a better validated classifier: the demo prioritizes a perceptible, appropriately audio-sensitive response while retaining text as the primary signal.

## Challenge requirement coverage

| Requirement | Implementation |
| --- | --- |
| One primary track | Text and audio. Video is intentionally excluded. |
| Multimodal input | The browser records audio, produces an editable transcript, and sends both the final text and original audio to the local server. |
| Structured MELD state | The interface displays the fused emotion label and confidence, plus text and audio diagnostics. |
| Short response grounded in the input | The local Qwen model receives the submitted text and fused emotion label, then streams a response of at most 160 tokens. |
| Input over time | The text branch uses up to two preceding transcripts and the fusion model retains causal dialogue state while the local server is running. |
| Coherent local prototype | FunASR, the emotion model, Qwen, and `llama.cpp` run locally through one browser workflow. |

Vision and reinforcement learning are optional extensions in the challenge specification. They were intentionally left out so the submission could focus on a complete, auditable text-audio system within the timebox.

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

| External or generated component | Role in this submission |
| --- | --- |
| MELD | Training and evaluation data, including the seven emotion categories. |
| Qwen3-1.7B GGUF | Local instruction-tuned response generator, served by `llama.cpp`. |
| `bert-base-uncased` | Fine-tuned contextual text-emotion encoder. |
| `emotion2vec_plus_base` | Frozen frame-level audio representation. |
| FunASR `paraformer-zh` and `ct-punc` | Local speech transcription and punctuation. |
| `llama.cpp` | Local OpenAI-compatible inference server for Qwen. |

The submission owns the integration, training procedures, evaluation artifacts, model-selection analysis, and the reported results. External pretrained models and the MELD dataset are identified above; their original licenses and terms remain applicable.

## Evaluation workflow

1. Run `./run_app.sh` and open `http://127.0.0.1:8000`.
2. Record a short message in the browser.
3. Review or edit the locally generated transcript.
4. Send the message and observe the streamed response.

## Privacy

Audio recording, transcription, emotion inference, and language-model generation run on the local machine. The browser does not use a hosted speech-recognition service or a hosted LLM API.

## Reproducibility

The interactive demo downloads its required public model assets on its first run and retrieves the fine-tuned demo weights with Git LFS. Reproducing model training requires the MELD archives, which are not tracked in Git. The training and benchmarking scripts are retained under `initial-testing/` and `benchmarking/`.

### Reproducing the experiments

The MELD raw archive must be available at `data/MELD/MELD.Raw.tar.gz`. The scripts use their checked-in defaults and write results beside the named output directories unless an explicit output path is provided.

| Goal | Script | Result or purpose |
| --- | --- | --- |
| Train the contextual text baseline | `initial-testing/train_text.py` | Fine-tunes the BERT text classifier and writes `initial-testing/training-output-text/`. |
| Reproduce the five-seed context-two study | `initial-testing/train_recurrent_dialogue_stabilized.py --context-window 2 --runs 5` | Produces the repeated-seed text-audio recurrent evaluation reported above. |
| Train a full-frame text-audio model | `initial-testing/train_recurrent_dialogue_frame_attention.py` | Trains the recurrent frame-attention architecture using the text checkpoint and cached audio features. |
| Run final frame-attention selection | `benchmarking/run_final_frame_attention.py --retrain` | Performs the three-fold, two-seed validation and records eligibility controls. |
| Run leakage-safe architecture evaluation | `benchmarking/run_final_architecture_cv.py --retrain` | Retrains text models inside dialogue-level outer folds and reports out-of-fold metrics. |
| Distill validated frame-attention teachers | `benchmarking/distill_frame_attention_ensemble.py` | Produces and evaluates a student from the eligible frame-attention teachers. |

The last three workflows are compute-intensive. The interactive submission is reproduced with `./run_app.sh`; the listed scripts reproduce the model studies and evaluation artifacts rather than being required to launch the demo.

## Limitations

- The emotion model was trained on MELD, an acted television-dialogue dataset, and may not generalize to everyday conversation.
- The classifier is not a clinical, diagnostic, or general-purpose emotion-assessment system.
- CPU inference is supported but may be slower than CUDA or Metal.
