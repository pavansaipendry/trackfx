# trackfx — architecture (import-the-stack)

Goal: take a video → **segment everything, labeled, with parts where possible** →
let the user assign a color per object/part → render a recolored video.

Philosophy: **import proven models, don't train an omniscient one.** Every box
below is an existing model wired together; the novel work is the glue + the UI +
the per-object color/track pipeline (which we've already proven for one object).

## Pipeline

```
 VIDEO
   │
   ▼
[1] INGEST            ffmpeg → frames;  shot-detection (cuts break tracking,
                      so we process per continuous shot)
   │
   ▼
[2] RECOGNIZE         open-vocab detector → labeled boxes ("person","car","drum")
  (what's here)       models: GroundingDINO / YOLO-World / RAM (tagger)
   │
   ▼
[3] SEGMENT           SAM 2: each detected box → instance mask
  (where)             + SAM 2 "everything" mode for things the detector missed
                      → DEDUP/merge (the user's completeness check: detector ∪ auto,
                        drop overlaps by IoU)
   │
   ▼
[4] PARTS             for categories with part models:
  (finer where)         person → human-parsing (SCHP) → hair/skin/shirt/pants
                        general → Semantic-SAM / VLPart → object parts
                      → each object optionally decomposes into labeled parts
   │
   ▼
[5] TRACK             SAM 2 video predictor: propagate every object/part mask
  (across frames)     through the shot; re-detect periodically for new objects
   │
   ▼
[6] ASSIGN            UI layer-list (labeled objects + parts). User picks a color
  (user control)      per layer, OR "auto" via a learned colorizer (DDColor).
                      Click-to-add for anything missed (SAM 2 prompt on demand).
   │
   ▼
[7] RENDER            transforms layer: luminance-preserving recolor within each
                      tracked mask → composite → mux original audio (ffmpeg)
   │
   ▼
 RECOLORED VIDEO   (+ interactive web editor over all of the above)
```

## The model stack (all imported)

| Role | Model(s) | Status |
|------|----------|--------|
| Recognize objects | GroundingDINO / YOLO-World / RAM | import |
| Segment (image+video) | **SAM 2** | ✅ already wired |
| Human parts | SCHP / Graphonomy (LIP/CIHP) | import |
| General parts | Semantic-SAM / VLPart / PartGLEE | import |
| Auto-colorize (optional) | DDColor | import |
| Glue / transforms / UI | our `transforms.py`, `video_io.py`, `app.py` | ✅ built |

## Compute / deployment
- Heavy models run on a **GPU backend** (RunPod pod or serverless) with a
  **pre-built image** (all models + weights baked in) so there's no per-run
  install/download overhead (the cost lesson: overhead >> the actual inference).
- Masks are **computed once and cached**; recoloring/rendering is then cheap and
  can run locally/instantly (already proven — re-render is ~1s).

## Honest hard parts (where effort/risk concentrates)
1. **Multi-object tracking** across long video — identity drift, occlusion,
   objects entering/leaving. Per-shot processing + periodic re-detect helps.
2. **Cuts** — must split into shots; tracking can't cross a cut.
3. **Long tail** — part models only cover some categories; the rest falls back to
   unlabeled SAM regions + user clicks.
4. **Edge quality** — feather masks to avoid color halos.
5. **Compute** — scales with (#objects × frames).
6. **UI** — managing potentially dozens of labeled layers usably.

## Phased build (tractable)
- **Phase 1** ✅ single click → track → recolor → web UI  *(done)*
- **Phase 2** + open-vocab detector → auto-find & label objects; multi-object
  track; **color-by-name** ("make the car red")
- **Phase 3** + human-parsing → people split into hair/skin/clothes
- **Phase 4** + Semantic-SAM (broader parts) + click-to-fix + video upload +
  GPU backend + audio
- **Phase 5** scale/robustness: shots, occlusion, re-identification

## What this realistically delivers
"Most nameable objects + their parts (for common categories), labeled, tracked,
and colorable by name or click." **Not** literal-perfect-everything (ill-posed),
but as close as today's models allow — with the user clicking to fix the rest.
