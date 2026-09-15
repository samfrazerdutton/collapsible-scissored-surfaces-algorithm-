# What stays open, what's the paid layer -- and the honest risk in between

The Cursor/Ghostty/Zed playbook: the tool is open for trust and distribution,
the layer that needs ongoing engineering investment per-customer is paid.
This only works if there's a real, durable line between "the default anyone
can use for free" and "the thing a team with real stakes (a fleet, a
safety case, a contract) actually needs and would pay for." Below is that
line as it stands today, plus the honest risk that the line might not hold.

## Stays open (this is the credibility/distribution engine -- it has to stay public and rigorous, or none of this works)

| Component | Why it stays open |
|---|---|
| Core C++ codec (`src/`, `include/csa/`) -- all transforms, entropy coders, the C ABI | This is what makes every claim checkable. An engineer evaluating whether to trust a compression claim needs to be able to read the actual transform, not take a vendor's word for it. |
| CLI (`scissorc squeeze`/`unsqueeze`) | The one-command adoption path. Gatekeeping this behind a paywall kills the "try it on your worst link in an afternoon" motion the whole wedge strategy depends on. |
| WASM/browser demos (`demo/`, `docs/`, `webapp/`) | Zero-friction proof. A link anyone can open with no install is the actual distribution channel -- see the HN/benchmark-post track. |
| Language bindings (Python/Rust/C#/Go) | Table stakes for adoption across the actual language mix robotics teams use (Python for tooling/prototyping, C++/Rust for the real-time path). |
| `csa_bridge` ROS2 wedge, as built (`ros2_wedge/`) -- single topic, single robot | This is the actual adoption unit per the "wedge, not platform" strategy. A team needs to be able to try this on one topic without a sales conversation. Gatekeeping the wedge itself defeats its purpose. |
| The adversarial benchmarks (`REAL_POSE_BENCHMARK.md`, `REAL_GEO_BENCHMARK.md`, `ADVERSARIAL_BENCHMARK.md`), losses included | This is the credibility asset. The moment benchmarks selectively favor the paid tier, they stop being trusted, and trust is the entire reason to publish them. |

## The paid layer -- where ongoing engineering investment is real, not just a license flag

| Component | What it actually is | Why it's not just "the open thing with a paywall" |
|---|---|---|
| Adaptive-precision tuning per deployment | A calibrated `--quality`/quant-step/resync schedule *for a specific fleet's actual data* (vibration profile, link budget, latency envelope) -- not the generic default `squeeze` picks from a single file's own precision. | This is genuinely a per-customer engineering exercise (profile their real telemetry, tune against their real link), not a static setting to open-source. It doesn't generalize the way the core codec does. |
| Fleet-scale SDK / multi-robot orchestration | Centralized calibration-profile management across hundreds of devices, per-device health/drift monitoring, coordinated rollout of tuning changes. | The single-topic wedge (open) proves it works on one robot. Fleet coordination is a different, ongoing operational problem most teams would rather buy than build. |
| Managed backhaul/observability layer | A hosted service: compressed telemetry lands in a managed backend, decompresses, bridges into Foxglove-compatible visualization, alerting, historical analytics -- the Formant-shaped layer. | This is infrastructure a company has to run and be paged for, not code a team can just `git clone`. Real operational commitment, real SLA. |
| Certified/qualified builds for safety-critical or embedded/RTOS targets | Actual porting, testing, and certification work for a specific target platform. | **Nothing in this repo has been built or tested for an embedded/RTOS target today** -- see `DESIGN.md`'s positioning-pivot section, which explicitly declined to claim this. If/when it's real, it's real paid engineering per target, not a flag to flip. |
| Priority support / correctness SLA | A guaranteed response time on bugs that affect a customer's production fleet. | Straightforward, standard open-core monetization -- the code is free, guaranteed human attention on it isn't. |

## The honest risk this whole plan rests on

Generic compression became a commodity because zstd is free *and good
enough* for almost everyone. The bet here is that domain-aware compression
for physical AI's specific data shapes (6-DOF pose, point clouds, the
kinematic structure inside them) is *not yet* commoditized the way generic
byte-stream compression is, because there's no MP3/H.264-equivalent
standard codec for it. That's plausible -- there's no incumbent -- but it
is not proven, and two things could undo it fast:

1. **A "good enough" default gets bundled somewhere upstream** (MCAP or
   ROS2 itself ships a pose-aware compression mode; Foxglove adds one).
   If the generic tool's own default becomes good enough, the same
   commoditization that already happened to byte-level compression happens
   here too, and the paid layer's leverage (per-deployment tuning) shrinks
   with it.
2. **The core technique is genuinely not that hard to replicate** once the
   idea is public -- calibrated delta-quaternion prediction and per-block
   adaptive precision are describable in a paragraph (this repo's own
   `DESIGN.md` does it). Publishing the honest benchmarks *as the
   distribution strategy* means also publishing exactly how the technique
   works. That's the right call for trust and adoption, but it's a real,
   accepted tradeoff against defensibility, not a free lunch.

The actual moat, if there is one, is more likely to be the fleet-scale
operational layer (tuning, monitoring, support) than the core algorithm
staying secret -- which is exactly why that's where this document puts the
paid tier, and exactly why the core algorithm shouldn't be the thing anyone
is counting on to stay defensible.
