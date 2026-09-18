# Human-Computer Lab Challenge

### Primary track

This demo will be **text + audio**. Why? Low-latency video processing is
 extremely difficult to accomplish under the time and hardware constraints
 (2-3 days of work, <6B params, local inference). Staying with the
philosophy that a simpler, more useful product is better than a complex,
half-working product also points towards audio as extracting emotion from
video is a lot more complex than audio and more likely to be less effective.

### My definition of "real-time"

A system like this is useless unless the client can hold an actual conversation
with it. My definition of real-time is **the user receives a response in N seconds**,
so that live conversation feels usable. Additionally, LLM responses are streamed
to the client (the full response isn't needed to begin reciting it), further
reducing the perceived latency.

### Model architecture
