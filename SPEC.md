# recolor — interactive instance-aware video colorization

Turn a black-and-white video into color by **segmenting and tracking every
object**, giving each its **own** color (so two different shirts can be two
different colors), with a human in the loop: the user can **pick colors** for any
object and **click anything we missed** to segment and color it too.

Inspired by wanting to watch *12 Angry Men* (1957) in color.

---

## Features (the vision) → how each works → how hard

| # | Feature | How we'd build it | Reality check |
|---|---------|-------------------|---------------|
| 1 | **Segment every object & track it across frames** | SAM 2 automatic mask generator (segment-everything) + SAM 2 video propagation to follow each mask through time | ✅ Core tech works. ⚠️ Tracking *literally everything* over a long film is heavy and masks drift/merge — realistically we nail the **salient** objects and approach "everything". |
| 2 | **Tell apart same-type things (shirt A red, shirt B blue)** | **Instance** segmentation: each shirt is its own tracked instance with its own color — not one "shirt" class painted uniformly | ✅ Distinct by construction. ⚠️ Hard part is keeping an instance's identity stable after it's occluded/leaves frame and returns (re-identification). |
| 3 | **User picks a color for an object** | UI: click an object → color picker → we recolor that tracked instance across **all** frames, preserving the original brightness/shading (luminance-preserving recolor) | ✅ Feasible; luminance-preserving recolor is a known technique. Needs a UI. |
| 4 | **User selects something we missed → we segment it** | SAM 2 **interactive prompt**: user clicks/draws a box on the region, SAM 2 returns a mask, we then track it forward & back through the video | ✅ This is exactly what SAM 2 is built for. Needs a UI to capture clicks. |

**Net:** every feature is real and maps to SAM 2 + a colorization model + a UI.
The honest hard parts are (a) "everything" over long video, (b) instance re-ID
after occlusion, and (c) building the interactive UI.

---

## Architecture

```
                 ┌─────────────────────────────────────────────┐
  B&W video ──▶  │ 1. SAM 2: segment + track objects → masks    │
                 │ 2. DDColor/DeOldify: base colorization        │
                 │ 3. Consistency: lock one color per tracked    │
                 │    instance across frames (anti-flicker)      │
                 │ 4. Recolor engine: apply chosen/auto color    │
                 │    into each mask, luminance-preserving       │
                 └─────────────────────────────────────────────┘
                                │
   Interactive UI  ◀────────────┘  (object list, click-to-segment,
   (video player)                   per-object color picker, re-render)
                                │
                 ffmpeg: reassemble frames + original audio ──▶ color video
```

- **Segment/track:** SAM 2 (instance masks + video tracking).
- **Colorize:** DDColor (vibrant) or DeOldify, as the per-frame color prior.
- **Consistency:** aggregate each tracked instance's color → one stable value
  (kills flicker; note: consistent ≠ historically accurate, colors are guesses).
- **Recolor:** blend color into the masked region in a luminance-preserving way
  (keep the grayscale luminance, replace chroma) so texture/shadows survive.
- **UI:** lets the user override colors and segment missed regions.

---

## Roadmap (build smallest-useful-thing first)

- **Phase 0 — does colorization even look good? (cheap, ~$1–2 GPU)**
  CLI: `extract frames → DDColor colorize → reassemble + audio`. A 15–30s clip.
  No segmentation yet. Proves the video pipeline + base color quality.

- **Phase 1 — instance-aware + consistency (the core idea)**
  Add SAM 2 to segment+track main objects; give each instance one stable color
  across frames. Compare side-by-side with Phase 0 (does per-object consistency
  actually look better / less flickery?).

- **Phase 2 — the human in the loop (the differentiator)**
  Interactive UI: object list, **click-to-segment** missed regions (SAM 2
  prompt), **per-object color picker**, re-render. This is what makes it a tool
  others would actually use.

- **Phase 3 — scale & robustness**
  Longer videos, occlusion/re-ID, speed/compute optimization, batch export.

---

## Honest risks / open questions
- Colors are **plausible guesses**, not the true 1957 colors. Per-object
  consistency makes it look *believable and stable*, not *accurate*.
- "Segment **every** object" is aspirational on long footage; we approach it.
- Instance re-identification after occlusion is the trickiest research bit.
- 2hr = ~170k frames → real compute. We validate on clips first, scale later.

## Status
Phase 0 not yet started. Decision pending: pick a test clip and run the minimal
colorize loop on the GPU.
