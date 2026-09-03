# Reproduction Targets — "Multimodal Wearable Sensing of Group Activity"

Ground-truth spec extracted from the FINAL submitted thesis PDF (130 pages, submitted 17 Aug 2026,
Arda Güney, University of Twente). Source: plain-text pdfplumber extraction
(`extracted_report.txt`, 4380 lines, page markers `=== PAGE N ===`). Numbers below are quoted
verbatim from the extraction wherever possible. Physical PDF page numbers are noted for a few
anchors; document page numbers (bottom-of-page) are given in table captions where the thesis
itself prints them.

**How to use this document**: this is the spec a from-scratch reproduction pipeline (raw sensor
data → tables) must match numerically. Section 8 at the end lists every naming convention,
file/format term, and known extraction uncertainty a human should double check against the actual
PDF before trusting a number for pipeline validation.

---

## 1. Dataset & experimental design (Chapter 3, "Experimental Design and Dataset")

### Study design
- Controlled but naturalistic collaborative setting: participants complete a **structured puzzle
  assembly task in groups of three**. Task cannot be completed efficiently alone — piece
  distribution, shared tools, and final assembly requirements force communication, object
  exchange, movement between workstations, and joint decision-making.
- **Wearable-first sensing approach**: video/audio used only for ground-truth annotation, never as
  model input. Sensor setup: ear-worn (OpenEarable 2.0), wrist-worn inertial (Xsens DOT), spatial
  position tracking (OptiTrack).
- Four design goals: (1) elicit observable individual/pairwise/whole-group activities repeatably;
  (2) collect synchronized multimodal sensor streams per participant; (3) create ground-truth
  annotations at multiple social levels; (4) generate a model-ready dataset for interaction
  detection and group activity recognition.
- **9 groups**, 3 participants each, one session per group → **27 participant recordings across 9
  collaborative sessions**. Different groups (not repeated trials) used specifically to support
  group-independent evaluation. Most participants took part in a single session; the exception is
  the author, who participated in 4 groups (Section 3.5). "The primary evaluation is therefore
  group-independent rather than strictly participant-independent."

### Collaborative puzzle task
- Task environment: **three individual workstations** + **one shared central table**. Puzzle
  divided into three sub-sections, each participant initially responsible for one.
- Cooperation forced via three mechanisms:
  1. **Mixed piece distribution**: some pieces needed by one participant are physically located at
     another participant's workstation.
  2. **Shared tools**: a single target/guide image (participants must coordinate access), and a
     shared tray for transporting pieces between workstations and the central table.
  3. **Connector pieces**: needed to merge the three sub-sections, located at the central table,
     requiring joint inspection/discussion during final assembly.

### Collaboration phases (conceptual, task design only — never annotated as explicit ELAN phase boundaries; see Section 10.5 caveat)
- **Phase A — Planning and Orientation**: inspecting instructions/target image, discussing
  strategy, assigning roles, moving to workstations. Expected signals: head movement,
  speech-related, moderate spatial movement, close proximity when gathered.
- **Phase B — Individual and Sub-Piece Building**: individual assembly at workstations mixed with
  occasional interaction (requesting pieces, sharing target image, handovers). "Valuable for
  studying binary interaction detection" because it mixes interaction/non-interaction.
- **Phase C — Joint Assembly and Merging**: moving to central table, testing connector pieces,
  joint inspection/discussion, co-building/co-merging. Expected signals: spatial clustering,
  increased inter-person proximity, hand movement, group-level coordination.

### Participants and recording sessions — researcher participation / naive groups (CONFIRMED)
- Every group intended to consist of 3 recruited participants. **In four sessions — Groups 1, 7,
  8, and 9 — the third recruited participant did not attend or could not be found in time**, so
  **the author (thesis writer) joined as the third participant**, wearing the same sensors and
  following the same instructions.
- **Researcher-participant groups: G1, G7, G8, G9** (4 groups).
- **Fully naive groups: G2, G3, G5, G6, G10** (5 groups) — this exact group-ID list is used
  consistently everywhere in the thesis it appears (Table 7.11 caption, Section 7.6, Section 8.10,
  Section 10.5, Table 10.1).
- **Group 4 was recorded but excluded** from the dataset entirely: its camera stopped recording
  partway through the session, so video-based ground-truth annotation could not be completed, so
  it could not be used for supervised modelling. Group numbering therefore skips G4.
- Final dataset groups used in **all** experiments: **G1, G2, G3, G5, G6, G7, G8, G9, G10** (9
  groups total, note the numbering gap at G4).
- Session lengths: shortest ≈ **22.9 min**, longest ≈ **66.0 min**, total recorded duration ≈
  **6.36 hours**, mean session length ≈ **42.4 minutes**.

**Table 3.1 — Recorded session duration per group** (verbatim):

| Group | Duration (min) |
|---|---|
| G1 | 30.5 |
| G2 | 47.1 |
| G3 | 44.5 |
| G5 | 66.0 |
| G6 | 22.9 |
| G7 | 48.3 |
| G8 | 38.6 |
| G9 | 59.4 |
| G10 | 24.1 |

(Note: Table 10.1, computed from ELAN annotation timelines rather than raw recording durations,
gives slightly different session lengths per group — e.g. G1 30.5 matches, but G2 48.2 vs 47.1
here, G3 45.2 vs 44.5, G5 68.2 vs 66.0, G6 22.9 matches, G7 49.7 vs 48.3, G8 38.8 vs 38.6, G9 60.0
vs 59.4, G10 25.0 vs 24.1. The thesis explicitly flags this: "Session length here is derived from
the ELAN annotation timeline and therefore differs slightly from the recording durations reported
in Table 3.1.")

### Sensing setup

**OpenEarable 2.0** (ear-worn, primary platform): recorded at **≈50 Hz**. Model-ready schema per
participant includes: accelerometer, gyroscope, magnetometer, bone-conduction accelerometer,
barometric pressure, environmental temperature, skin temperature, photoplethysmography (PPG)
channels.

**Xsens DOT** (wrist-worn IMU): recorded at **≈30 Hz**. Model-ready schema per participant
includes: Euler orientation components, linear acceleration components, gyroscope components.

**OptiTrack** (spatial position tracking, ceiling-mounted optical mocap, external reference — not
body-worn): recorded 3D position at **≈240 Hz**. Data includes spatial coordinates and
tracking-availability information.

**Video/audio**: recording only, used exclusively for ELAN ground-truth annotation, never a model
input.

**Table 3.2 — Overview of sensor modalities used in the experiment** (verbatim):

| Sensor system | Placement / role | Approx. rate | Main information captured |
|---|---|---|---|
| OpenEarable 2.0 | Ear-worn | 50 Hz | Head-related motion, bone-conduction acceleration, barometer, temperature, and PPG |
| Xsens DOT | Wrist-worn | 30 Hz | Wrist orientation, acceleration, gyroscope, and hand/arm movement |
| OptiTrack | Spatial tracking | 240 Hz | 3D participant position, movement, proximity, and tracking availability |
| Video/audio | Environment | Recording only | Ground-truth annotation source; not used as model input |

### Annotation levels (7 tiers)
Participant1; Participant2; Participant3; Participant1–Participant2; Participant1–Participant3;
Participant2–Participant3; Whole Group. (ELAN tier names use underscores — see Section 4.5 below
and Section 8 naming list.)

### Dataset summary

**Table 3.3 — High-level summary of the collected dataset** (verbatim):

| Property | Value |
|---|---|
| Number of groups | 9 |
| Participants per group | 3 |
| Total participant roles | 27 |
| Task type | Collaborative puzzle assembly |
| Main sensor systems | OpenEarable 2.0, Xsens DOT, OptiTrack |
| Ground-truth source | Video/audio annotation in ELAN |
| Total recorded duration | Approximately 6.36 hours |
| Mean session duration | Approximately 42.4 minutes |
| Session duration range | Approximately 22.9–66.0 minutes |
| Annotation levels | Individual, pairwise, whole-group |
| Fine-grained labels | 149 |
| Grouped activity categories | 16 |

Raw annotation vocabulary: **149 fine-grained labels**, cleaned/grouped into **16 broader activity
categories**. Two main prediction tasks derived: (1) binary interaction detection
(interaction/non-interaction per window); (2) group activity recognition (conversation,
co-building, co-merging, among others).

### Activity distribution

**Table 3.4 — Summary of grouped activity categories derived from the fine-grained annotation
vocabulary** (verbatim; columns: Activity group, Number of fine labels, Instances, Total duration
in minutes):

| Activity group | # fine labels | Instances | Total duration (min) |
|---|---|---|---|
| Collaborative building | 4 | 108 | 83.2 |
| Traveling / approaching | 15 | 472 | 29.7 |
| Conversation / talk | 14 | 274 | 45.0 |
| Inspecting / checking | 31 | 220 | 21.7 |
| Picking up | 10 | 188 | 8.0 |
| Putting down | 9 | 105 | 3.7 |
| Moving / placing pieces | 15 | 87 | 20.9 |
| Object handover | 6 | 87 | 4.0 |
| Matching to image | 4 | 58 | 36.4 |
| Carrying / delivering | 16 | 52 | 3.1 |
| Merging | 2 | 42 | 38.9 |
| Pointing / presenting | 7 | 32 | 2.2 |
| Searching | 4 | 27 | 1.9 |
| Target / guide image | 4 | 13 | 5.8 |
| Synchronization | 4 | 13 | 0.9 |
| Other | 1 | 1 | 0.1 |

Interpretation: Building dominant in total duration/instances; conversation and co-building also
important; merging has fewer instances but relatively long duration (final assembly phase can run
several minutes once started).

### Modelling tasks derived (Section 3.11, elaborated fully in Ch. 6)
- **Task 1 — Interaction Detection** (RQ1): binary classification, window labelled interaction if
  a sufficient portion of the window contains pairwise or whole-group activity, else
  non-interaction.
- **Task 2 — Group Activity Recognition** (RQ2): classify selected whole-group labels
  (conversation, co-building, co-merging) — selected because they have sufficient support versus
  rarer fine-grained labels.

---

## 2. Data processing pipeline (Chapter 4, "Data Processing, Synchronization, and Annotation")

### Pipeline overview — 6 stages
1. **Raw data loading**: OpenEarable, Xsens DOT, OptiTrack, ELAN files loaded per group.
2. **Timestamp normalization**: device-specific timestamps converted to comparable time
   representations.
3. **Synchronization**: sensor streams and annotation timelines aligned to a shared video-time
   axis.
4. **Annotation cleaning**: raw ELAN labels standardized and mapped to consistent activity groups.
5. **Window construction**: synchronized streams segmented into fixed-duration windows, resampled
   to a common length.
6. **Label assignment**: each window receives interaction-detection and (where applicable) group
   activity-recognition labels.

### Raw data sources

**Table 4.1 — Raw data sources used in the processing pipeline** (verbatim):

| Source | Approx. rate | Main content | Role in the pipeline |
|---|---|---|---|
| OpenEarable 2.0 | 50 Hz | Ear-worn multimodal signals | Model input; head-related motion and earable signals |
| Xsens DOT | 30 Hz | Wrist inertial signals | Model input; hand and arm movement |
| OptiTrack | 240 Hz | 3D position data | Model input; spatial position and proximity |
| ELAN annotations | Event-based | Manual activity labels | Ground-truth labels for model targets |
| Video/audio | Recording-based | Visual and audio evidence | Annotation source only; not used as model input |

Example given for why alignment/resampling is needed: a 5-second segment contains ≈250
OpenEarable samples, ≈150 Xsens samples, ≈1200 OptiTrack samples.

### Shared time representation
- Central shared time axis: **`video_time_s`** — seconds relative to the start of the video
  recording. 0 = video start; negative values possible if a sensor stream started before the
  video.

**Table 4.2 — Important time-related columns used during processing** (verbatim — NOTE:
extraction looks slightly garbled/duplicated, see Section 8 flags below):

| Column | Meaning |
|---|---|
| `video_time_s` | Microsecond timestamp used in the OpenEarable stream. |
| `datetime_utc` | UTC datetime representation derived from timestamps when available. |
| `SampleTimeFine` | Device-specific Xsens time column. |
| `frame` | OptiTrack frame index. |
| `time_s` | Time in seconds relative to the start of a sensor recording. |
| `video_time_s` | Shared time in seconds relative to the video start; used for alignment with ELAN annotations. |

(NOTE: `video_time_s` appears twice in this table with two different descriptions — "microsecond
timestamp used in the OpenEarable stream" for the first row vs. "Shared time in seconds relative
to video start" for the last row. This is very likely a pdfplumber column-merge artifact — the
first row's description more plausibly belongs to a raw OpenEarable-native timestamp column, not
`video_time_s` itself. **Flagged as uncertain — verify against the actual PDF table.**)

### Synchronization pipeline — two stages
1. **Coarse alignment**: uses available absolute timestamps from sensor files/recording metadata;
   compares sensor stream start time to video recording start time to get an initial offset.
2. **Fine alignment using synchronization events**: uses a deliberate/visible synchronization
   movement identifiable both in video and in sensor signals (distinctive peaks in
   acceleration/gyroscope channels). Procedure:
   1. Identify a synchronization event in video/ELAN annotation.
   2. Locate the corresponding peak/motion pattern in the relevant sensor signal.
   3. Compute the residual time difference between video event and sensor event.
   4. Shift the sensor stream by this residual offset.
   5. Verify alignment visually by plotting sensor signals with the aligned annotation interval.

Synchronization quality varied by group (visibility of sync events differed by sensor/system);
required both automatic timestamp processing and manual verification.

### ELAN annotation structure — 7 tiers (exact tier names)
- `Participant1`
- `Participant2`
- `Participant3`
- `Participant1_Participant2`
- `Participant1_Participant3`
- `Participant2_Participant3`
- `Whole_Group`

**Table 4.3 — ELAN annotation tiers used for group activity labelling** (verbatim):

| Tier | Social level | Example interpretation |
|---|---|---|
| Participant1 | Individual | Activity performed by Participant 1 alone |
| Participant2 | Individual | Activity performed by Participant 2 alone |
| Participant3 | Individual | Activity performed by Participant 3 alone |
| Participant1_Participant2 | Pairwise | Interaction between Participants 1 and 2 |
| Participant1_Participant3 | Pairwise | Interaction between Participants 1 and 3 |
| Participant2_Participant3 | Pairwise | Interaction between Participants 2 and 3 |
| Whole_Group | Group-level | Activity involving the full group |

### Label cleaning and grouping
- Raw ELAN vocabulary: **149 fine-grained labels** → grouped into **16 broader activity
  categories**: building, travelling/approaching, conversation, inspecting, picking up, putting
  down, moving/placing pieces, object handover, matching to image, carrying/delivering, merging,
  pointing/presenting, searching, target/guide-image activity, synchronization, other.
- Full fine→grouped mapping is in **Appendix A** ("Label Mapping and Grouping Scheme", Table
  A.1) — see Section 8 below for the exact label strings extracted from that appendix.
- Rationale: very fine-grained labels are too rare/brief for reliable classification with 9
  groups; grouping balances detail vs. robustness, especially important for macro-F1 where rare
  classes strongly affect scores.

### Windowing and resampling — CRITICAL NUMBERS
- **Window length: 5 seconds** ("each window covers five seconds of data").
- **Resample length: 64 time steps** for sequence-based models — fixed length regardless of
  sensor's native sampling rate, giving X_i ∈ ℝ^(64×C) per window (C = number of channels/features).
- **Advanced/engineered feature windows use a different, LONGER window: 10 seconds** (Section
  5.10) — "for the activity-recognition experiments these advanced features are computed over
  ten-second windows, each of which inherits the dominant (span-majority) label of the underlying
  five-second recognition-core windows; the binary interaction and forecasting experiments retain
  the five-second windows described earlier." — i.e., there are TWO window regimes in the final
  pipeline: 5s base/recognition-core windows (used directly for Task 1 interaction detection and
  for forecasting), and 10s advanced-feature windows (used for the Task 2 three-class engineered
  feature experiments), with the 10s window label taken as the majority label of the underlying 5s
  windows it spans.
- Two window extraction types considered:
  - **Fixed windows**: back-to-back 5-second windows across the full recording (realistic
    deployment setting, includes uninformative periods).
  - **Labelled windows**: windows extracted from annotated intervals / windows with sufficient
    annotation coverage (cleaner examples).

### Window label assignment rules
- **Overlap-based label assignment**: for each window, pipeline computes how much of the window
  overlaps with relevant annotation intervals.
- **Interaction detection labels**: derived from pairwise + whole-group tiers. **If pairwise or
  whole-group activity is active for at least half (≥ 0.5) of the 5-second window, label =
  interaction; else non-interaction.** Formally: T_interaction / T_window ≥ 0.5. (Text: "This 50
  [%]..." — sentence is truncated/garbled in extraction right after stating the rule; the 50%
  threshold itself is clearly stated via the formula and repeated in Section 4.8.3.)
- **Group activity recognition labels**: window assigned to whichever of {conversation,
  co_building, co_merging} occupies the **largest portion of the window**. Windows without
  sufficient overlap with one of the three target classes are **excluded** from the three-class
  task (not forced into a class).
- **Ambiguous windows** (Section 4.8.3): handled "conservatively" — text is truncated in the
  extraction right after "For binary interaction detection, the 50 [%]..." (same garbling as
  above — likely restates the 50% overlap rule as the conservative handling criterion). **Flagged
  as uncertain / truncated — verify exact wording against PDF.**

### Model-ready dataset outputs — CRITICAL for repo structure

**Table 4.4 — Main model-ready outputs produced by the processing pipeline** (verbatim):

| Output type | Used for | Description |
|---|---|---|
| Raw sequence windows | Transformer-based models | Five-second windows resampled to 64 time steps with multimodal channels |
| Engineered feature windows | Classical models and feature analysis | Window-level summaries and interaction-aware features |
| Interaction labels | RQ1 | Binary labels: interaction versus non-interaction |
| Activity labels | RQ2 | Three-class labels: conversation, co_building, and co_merging |
| Group identifiers | Evaluation | Used for leave-one-group-out splitting |

Two broad representation types produced: (1) **raw multimodal windows** — resampled sensor
windows with OpenEarable+Xsens+OptiTrack channels, suited to sequence models (Transformer); (2)
**engineered feature windows** — window-level features for proximity, motion energy,
coordination, group spatial structure, suited to classical ML (Logistic Regression, Random
Forest).

### Quality control (Section 4.10)
Four checks: (1) plot synchronized streams with annotation intervals to confirm major events
occur at expected times; (2) inspect synchronization movements to verify sensor-signal peaks
align with video events; (3) check label distributions post-grouping for rare/dominant
classes/inconsistencies; (4) inspect window counts per group to ensure every recording
contributes data.

---

## 3. Feature engineering (Chapter 5, "Feature Representation")

### Two representation strategies compared
1. **Raw multimodal representation**: sensor channels with minimal semantic transformation.
2. **Engineered interaction-aware representation**: features explicitly encoding group-level
   structure/coordination.

### Raw multimodal representation (Section 5.2)
- Sequence models: each 5s window resampled to 64 time steps, C channels → preserves temporal
  structure for Transformer.
- Classical models: raw channels summarized per window using **mean, standard deviation, minimum,
  maximum** (4 stats) per channel — "compact baseline representation."

### Limitations of raw features (Section 5.3) — 4 points
1. Participant-centred, not relational (no explicit distance/relative-movement/coordination
   info).
2. Hard to interpret (same wrist acceleration ≈ building/carrying/pointing/searching/adjusting).
3. Requires more data to learn group structure implicitly (only 9 groups available).
4. Different group activities can be semantically similar in raw motion space (co_building vs
   co_merging both involve hand movement, manipulation, close proximity).

### Engineered feature categories (Section 5.4) — 4 main categories
1. Proximity and spatial features
2. Group movement features
3. Head-movement features
4. Hand-movement and coordination features

#### 3.1 Proximity and spatial features (Section 5.5, OptiTrack-derived)
Concepts: distance between each pair of participants; nearest-pair distance; middle-pair distance;
farthest-pair distance; group dispersion; group centroid position; group centroid speed; movement
toward/away from the central workspace.

#### 3.2 Group movement features (Section 5.6)
Group centroid = average participant position. Fast centroid movement → transitioning between
areas; stable+clustered → working together; stable+dispersed → working separately. Helps
distinguish: distributed individual work; pairwise interaction at one workstation; whole-group
clustering at central table; transitions between workstations and central table.

#### 3.3 Head-movement features (Section 5.7, OpenEarable-derived)
Concepts: head movement intensity; head movement variability; approximate head pitch; similarity
of head-facing direction across participants; frequency/power of head motion. Interpretation:
intensity ↑ during searching/comparing/attention-shifting; head-facing similarity → shared
attention (e.g. looking at target image). Note: raw audio not used, video/audio annotation-only.

#### 3.4 Hand-movement features (Section 5.8, Xsens-derived)
Concepts: wrist acceleration intensity; wrist gyroscope intensity; hand-motion variability; wrist
orientation variation; frequency/power of hand motion. Most useful combined with spatial/group
context since similar hand motion appears in individual building, co-building, and co-merging.

#### 3.5 Coordination features (Section 5.9)
Concepts (cross-participant comparison within a window): similarity of hand-motion intensity
between participants; similarity of head movement between participants; variability of group
movement; simultaneous movement peaks; pairwise motion synchrony. Purpose: make the
"between-person" structure explicit rather than requiring the model to infer it implicitly from
independent per-participant channels.

### 3.6 Advanced Multimodal Feature Engineering (Section 5.10) — the final, most-expanded feature stage

This is a **second, more extensive** feature-engineering stage used for the final experiments (on
top of the conceptual categories above).

- Computed over **10-second windows** for the activity-recognition experiments (inheriting the
  span-majority label from underlying 5-second recognition-core windows); binary interaction and
  forecasting experiments retain 5-second windows.
- **The advanced three-class activity-recognition dataset contains 992 windows: 548 co_building,
  311 conversation, 133 co_merging windows.**
- After expansion: **531 OptiTrack features, 506 OpenEarable columns, 693 Xsens features.**
  - **531 OptiTrack** = 59 spatial signals × 9 window statistics, decomposed as: 43
    relative/motion signals + 14 absolute-position signals + 2 tracking-quality signals →
    (43×9=387) + (14×9=126) + (2×9=18) = 531.
  - **693 Xsens** = 66 continuous wrist signals × 9 statistics (66×9 = 594) + 96 cross-person
    synchrony descriptors + 3 availability features = 594+96+3 = 693.
  - **506 OpenEarable** is NOT a single clean multiplication — mixes head-motion, posture,
    spectral, synchrony, and dominance descriptors. OpenEarable-only experiments used smaller
    selected/available sets, e.g. **≈305 features for the developed OE9/OE10 configurations.**

#### Window Statistic Expansion (Section 5.10.1)
Most base signals expanded into a common set of **9 summary statistics**: **mean, standard
deviation, minimum, maximum, range, median, interquartile range (IQR), 10th percentile, 90th
percentile.** Applied uniformly per derived base signal (e.g. a pairwise distance, an acceleration
magnitude, a jerk signal).

#### OptiTrack feature families (Section 5.10.2)

**Table 5.1 — OptiTrack feature families derived from 3D participant positions** (verbatim):

| Feature family | How it is calculated | What it represents |
|---|---|---|
| Pairwise distance | Euclidean distance between P1–P2, P1–P3, P2–P3 | Who is close to whom |
| Nearest / middle / farthest pair | Sorting the three pairwise distances | Closest pair, middle pair, most separated pair |
| Distance change rate | Change in pair distance divided by time | Approach or separation |
| Centroid | Average position of the three participants | Group centre |
| Centroid speed | Centroid displacement divided by time | Movement of the group as a whole |
| Group spread | Distance of each participant to the centroid | Compact vs. dispersed group |
| Triangle area / perimeter | Geometry of the participant triangle | Formation size and shape |
| Triangle compactness | Area divided by perimeter squared | Huddle-like vs. spread-out formation |
| Tracking quality | Valid / available position fractions | Reliability of the spatial data |

#### OpenEarable feature families (Section 5.10.3)

Group-level descriptors also computed: number of simultaneously active participants, dominance
and asymmetry of movement across participants, pairwise synchrony and lagged-correlation
features. Developed sets: **OE9** (motion features + magnetometer magnitude) and **SPECIAL_OE**
(dedicated configuration for binary interaction detection) — used to test whether careful
ear-worn feature engineering can close the gap to spatial sensing (results in Ch.7).

**Table 5.2 — OpenEarable feature families derived from ear-worn inertial and magnetic signals**
(verbatim; some formula rendering is math-garbled in extraction, flagged):

| Feature family | How it is calculated | What it represents |
|---|---|---|
| Acceleration magnitude | Norm of (accₓ, acc_y, acc_z) | Head/body movement intensity |
| Gyroscope magnitude | Norm of (gyroₓ, gyro_y, gyro_z) | Turning and rotation intensity |
| Pitch | atan2(accₓ, √(acc_y²+acc_z²)) [formula OCR-uncertain] | Looking down/up |
| Roll | atan2(acc_y, √(accₓ²+acc_z²)) [formula OCR-uncertain] | Head tilt |
| Jerk | Derivative of acceleration magnitude | Sharp head movement |
| Angular jerk | Derivative of gyroscope magnitude | Sharp turning |
| Head-down fraction | Fraction of window with pitch below a threshold | Looking down at task materials |
| Turn rate | Gyroscope bursts above percentile thresholds | Frequent head turns |
| Nod-band / entropy | Spectral properties of movement signals | Rhythmic vs. irregular head motion |
| Magnetometer magnitude | Norm of (magₓ, mag_y, mag_z) | Magnetic/orientation change |
| Heading | atan2(mag_y, magₓ), unwrapped | Heading/orientation proxy |
| Active-person count | Number of participants active at each time step | Group-level movement intensity |
| Dominance/asymmetry | Max-minus-median and std/mean across people | Whether one person dominates movement |
| Synchrony / lag | Pairwise correlations and lag correlations | Whether people move together or with delay |

**FLAG — pitch/roll formulas**: The pdfplumber extraction of the Pitch/Roll rows contains visibly
garbled math notation (stray `(cid:0)`, `q`, `p` characters from PDF glyph substitution). The
general form (`atan2` of an acceleration component over the root-sum-square of the other two) is
recoverable, but the exact axis assignment should be verified against the PDF directly before
reimplementing.

#### Xsens feature families (Section 5.10.4)

**Table 5.3 — Xsens feature families derived from wrist-worn inertial signals** (verbatim, same
math-OCR caveat as above for the norm formulas):

| Feature family | How it is calculated | What it represents |
|---|---|---|
| Acceleration norm | √(accₓ²+acc_y²+acc_z²) | Overall wrist acceleration |
| Gyroscope norm | √(gyrₓ²+gyr_y²+gyr_z²) | Wrist rotation intensity |
| Dynamic acceleration | Acceleration axis minus window median | Movement after removing static posture |
| Jerk | Derivative of dynamic acceleration | Sharp hand movement |
| Angular jerk | Derivative of gyroscope signal | Sharp wrist rotation |
| Euler rate | Derivative of Euler angles | Wrist orientation change |
| Burst fractions | Fraction of time above movement thresholds | Active hand movement |
| Spectral features | Entropy and frequency-band ratios | Rhythmic hand motion |
| Pair synchrony | Correlations between participants | Coordinated hand movement |

### Feature Categories summary (Section 5.11)

**Table 5.4 — Main categories of engineered interaction-aware features** (verbatim):

| Feature category | Main source | Intended interpretation |
|---|---|---|
| Proximity | OptiTrack | Whether participants are physically close, separated, or clustered |
| Spatial movement | OptiTrack | Whether the group moves between workstations and the central table |
| Group dispersion | OptiTrack | Whether the group is spread out or concentrated in one area |
| Head movement | OpenEarable | Attention shifts, looking behaviour, and communication-related movement |
| Hand movement | Xsens DOT | Object manipulation, piece placement, carrying, and building activity |
| Coordination | Multiple sources | Whether participants show related movement patterns within the same window |

(Thesis notes this table is a conceptual summary, not the full feature list; full feature names
"can be included in an appendix once the final feature names are fixed" — i.e. **the thesis itself
does not provide the literal final feature-name list**, only counts and family descriptions above.)

### Representation and model choice (Section 5.13)
- Classical models (Logistic Regression, linear/RBF SVC, Random Forest) operate on **window-level
  feature vectors**: each 5s window → its (selected) engineered features → classified
  independently.
- Deep sequence models (LSTM, BiLSTM, GRU, Transformer) also use engineered features but at a
  higher temporal level: **short sequences of consecutive window-level feature vectors** (e.g.
  sequences of **9 or 18 windows**), with feature selection applied inside each training fold.

### Feature scaling and preparation (Section 5.14)
- Scaling needed because features have different units/ranges (distance vs. acceleration vs.
  temperature/pressure).
- Especially relevant for Logistic Regression (weighted linear combinations); Random Forest less
  sensitive but consistent preprocessing still applied for fair comparison.
- **Scaling fitted only on training folds within the training process** (group-independent
  evaluation: scaling parameters estimated from training groups only, applied to held-out test
  group) — to avoid information leakage. (Chapter 6 names the specific scaler: **RobustScaler**,
  fit per-fold on training groups only.)

---

## 4. Modelling & evaluation methodology (Chapter 6 — no tables, but this prose is the backbone of the pipeline)

### Task definitions (Section 6.1–6.3)
- **Task 1** (RQ1): binary classification, classes `interaction` / `non_interaction`, target
  derived from presence of pairwise/whole-group annotations per window.
- **Task 2** (RQ2): uses whole-group labels `conversation`, `co_building`, `co_merging`.
  Decomposed into **5 subtasks**:
  - **Task 2a**: conversation vs. non-conversation
  - **Task 2b**: conversation vs. co-building
  - **Task 2c**: conversation vs. co-merging
  - **Task 2d**: co-merging vs. co-building (explicitly the hardest boundary — "directly measures
    the hardest boundary in this dataset")
  - **Task 2e**: three-class recognition (conversation, co-building, co-merging) — the main
    multi-class task.
  - (Note: prompt goal-sheet mentions "Task 2a-2e" as if 5 lettered variants of one enumeration;
    the thesis text literally lists 2a/2b/2c/2d as four pairwise subtasks plus 2e as the combined
    three-class subtask — confirmed as written above.)
- **Task 3** (RQ3, covered separately in Ch. 8): next-activity forecasting — anticipate the
  group's *upcoming* activity state from recent history/sensor evidence, framed throughout as
  **exploratory**, not solved classification.

### Sensor-combination ablation design (Section 6.4) — CONFIRMED LIST
Every task evaluated across **all seven combinations** of the three modalities:
1. **OE** (OpenEarable only)
2. **OPTI** (OptiTrack only)
3. **XSENS** (Xsens only)
4. **OE+OPTI**
5. **OE+XSENS**
6. **OPTI+XSENS**
7. **OE+OPTI+XSENS**

For each task × sensor combination, **three method regimes** compared:
1. Best classical model **without** elapsed time
2. Best classical model **with** elapsed time
3. Best fast deep sequence model **without** elapsed time

### Treatment of elapsed session time (Section 6.5)
- "Elapsed session time" = time since session start, a contextual/causal cue (not a raw sensor
  signal) — correlated with collaboration phase (planning early, individual building middle,
  merging late).
- Classical models reported **both without and with** elapsed time as a feature.
- Deep sequence models reported **only without** elapsed time, "so that their scores reflect
  sensor information and temporal context only."

### Models (Section 6.6)

**Classical model family** (6.6.1): Logistic Regression, linear SVC (LinearSVC), RBF-kernel SVC
(RBF-SVC), Random Forest, ExtraTrees (depending on experiment). All operate on window-level
engineered feature vectors. **Regularization: C = 1 fixed for linear models.** Feature scaling
fitted on training folds only. Best-performing classical model per cell reported with its
feature-selection setting.

**Deep sequence model family** (6.6.2): LSTM, BiLSTM, GRU, Transformer. Receive short sequences of
consecutive window-level feature vectors (not raw sensor time series): sequence length **s = 9**
for Task 2 subtasks, **s = 18** for Task 1; each window represented by its **top-k selected
engineered features (typically k = 120)**. Trained to classify the final window of the sequence
(causal — only past/present info used). Called the **"fast deep learning" setting**.

**Separately optimized Task 1 configuration** (in addition to the fast/ablation setting): a grid
of **96 runs** over:
- 4 architectures: LSTM, BiLSTM, GRU, Transformer
- 3 sequence lengths: 6, 12, 18 windows
- 4 feature counts: k ∈ {80, 120, 160, 211}
- 2 random seeds: 7 and 42
- all on the **OPTI2_RELATIVE_ONLY** feature set (relative OptiTrack features)
- (4×3×4×2 = 96, confirmed)

Strongest optimized config: **Transformer, sequence length 18, k=120, on OPTI2_RELATIVE_ONLY**,
seed 42 selected as the max of the grid (headline macro-F1 = **0.8062**; same config with seed 7
= macro-F1 **0.7939**, i.e. **0.0122** of the headline is attributable to seed choice alone). The
thesis explicitly frames 0.8062 as an **optimistic upper-end estimate** ("roughly 0.79–0.81
macro-F1" honest range), because architecture/seq-len/k/seed were all selected in hindsight on
the same LOGO score being reported.

### In-fold feature selection (Section 6.7)
- Method: **univariate feature selection with SelectKBest** (applied because the engineered
  representation is high-dimensional per Section 5.10).
- **Classical-model ablation**: SelectKBest sweeps **k ∈ {40, 80, 120, 200, all}**; the k that
  maximizes LOGO macro-F1 is chosen per task × sensor-combination cell (in-hindsight selection —
  flagged as introducing a "mild optimistic bias").
- **Fast deep-learning ablation**: k **fixed at 120** (no per-cell k search).
- **Separately optimized Task 1 experiment**: distinct grid {80,120,160,211} (see above).
- Selection always performed **inside each training fold**: selector fitted on training groups
  only, applied to held-out test group (avoids feature-selection leakage).
- **Feature scaling: RobustScaler**, likewise fit on training groups only within each fold, applied
  to held-out group.
- Thesis explicitly warns: comparisons between classical and deep regimes should be read as
  **indicative, not exactly calibrated**, because classical cells get per-cell k search while deep
  cells use a single fixed k=120.

### Evaluation protocol (Section 6.8)

**Leave-one-group-out (LOGO)** (6.8.1): each of the 9 groups held out once as test set, models
trained on the remaining groups, scores aggregated across folds → **9 folds**. Primary/only
protocol used throughout. Explicitly noted: full 9-group evaluation is group-independent but
**not strictly participant-independent** (the author appears in 4 sessions, so can reappear in
training data when one of those 4 groups is held out) — addressed via the separate 5-naive-group
analysis (Section 7.6 / Table 7.11–7.13).

Random-seed treatment differs per experiment: systematic recognition ablation uses a single fixed
seed; separately-optimized Task 1 grid evaluates 2 seeds (7, 42) and reports the selected config;
forecasting experiments (Ch. 8) average over multiple seeds where explicitly noted.

**Metrics** (6.8.2): **accuracy, macro-F1, balanced accuracy**. **macro-F1 is the main selection
metric** (dataset imbalanced — co_building >> co_merging). Balanced accuracy reported as
complementary imbalance-robust metric.

**Pooled vs. per-fold reporting — EXACT DISTINCTION** (critical for reproducing table semantics):
- **"Pooled" scores** = predictions from all 9 LOGO folds **concatenated and scored once**. This
  is the default reported value unless stated otherwise. Pooled macro-F1 tends to be the **higher**
  of the two readings because pooling lets frequent classes in large groups compensate for folds
  where a minority class is nearly absent.
- **"Fold mean ± SD"** = mean and standard deviation of the **9 per-fold (per-group) scores**,
  i.e. compute the metric separately per held-out group, then average those 9 numbers. This is
  the **more conservative** reading and describes what to expect for a genuinely new/unseen
  group.
- Example given: optimized Task 1 interaction result — **pooled macro-F1 = 0.806** but **fold-level
  mean = 0.781 ± 0.086**; all-window Task 3 persistence result — **pooled macro-F1 = 0.743** but
  **fold-level mean = 0.671 ± 0.101**.
- Throughout Ch.7 tables: "A / M / B" = accuracy / macro-F1 / balanced accuracy, given as the
  pooled value on the first line of a cell, with the smaller second line giving fold mean ± SD for
  accuracy and macro-F1 only (not balanced accuracy) across the 9 held-out groups.
- 95% confidence intervals for the fold mean use **t = 2.306** for 9 folds (8 degrees of freedom)
  and **t = 2.776** for 5 folds (4 degrees of freedom, used in the naive-cohort-only tables).

---

## 5. Results — Task 1 & Task 2 (Chapter 7) — MAIN REPRODUCTION TARGET

### 7.1 Overview / table-reading conventions
All results under LOGO with in-fold feature selection. "A / M / B" = accuracy / macro-F1 /
balanced accuracy. Final column of each ablation table names which of the 3 regimes achieved
highest macro-F1. Every result cell reports the **pooled** score on the first line and the
**per-fold mean ± SD** (accuracy / macro-F1) on the smaller second line — see Section 4 above for
the exact semantics. "differences smaller than roughly one standard deviation should not be read
as evidence that one configuration is better than another."

### 7.2 Headline Results

**Table 7.1 — Highest-level verified results per task under LOGO evaluation** (verbatim; the
Task 1 row is the max of the 96-run tuning grid selected on the same LOGO score it reports — an
optimistic estimate):

| Task | Feature set / sensor | Model | A | M | B | macro-F1 fold mean ± SD [95% CI] |
|---|---|---|---|---|---|---|
| Task 1: interaction vs. non-interaction | OPTI2_RELATIVE_ONLY | Transformer (seq=18, k=120) | 0.8064 | 0.8062 | 0.8065 | 0.781 ± 0.086 [0.716, 0.847] |
| Task 2e: 3-class recognition (7-combination table) | OE+OPTI | LSTM (seq=9, k=120) | 0.7565 | 0.6994 | 0.7001 | 0.578 ± 0.132 [0.477, 0.679] |

Context: for Task 2e, the OptiTrack-only configuration (A=0.767, M=0.695) differs from the
OE+OPTI headline by less than the per-fold SD, so "these data do not support a meaningful ranking
between the two." Two consistent cross-table patterns: (1) OptiTrack is the most reliable
modality for spatially-structured tasks; (2) deep sequence models are frequently strongest when
OptiTrack is present, classical-with-elapsed-time wins in most combinations lacking spatial
features.

### 7.3 Task 1: Interaction Detection (RQ1)

Best fast (ablation-grid, non-optimized) result: OptiTrack-only Transformer, macro-F1 0.751,
narrowly ahead of classical OptiTrack+elapsed (macro-F1 0.750). The separately-optimized
OPTI2-relative Transformer improves this to macro-F1 0.8062 (Table 7.1). Without spatial features
performance drops sharply: Xsens alone macro-F1 0.516 (no elapsed); OpenEarable alone macro-F1
0.634 (no elapsed). Elapsed time consistently lifts non-spatial combos (e.g. OE: 0.634 → 0.699).

**Table 7.2 — Task 1 (interaction vs. non-interaction): best model per regime and sensor
combination** (verbatim; cell format = pooled A/M/B on line 1, "fold mean±SD" for A/M on line 2):

| Sensor | Classical, no elapsed | Classical, with elapsed | DL, no elapsed | Best (M) |
|---|---|---|---|---|
| OE | logreg, k=200 — 0.635/0.634/0.636; fold 0.639±0.054 / 0.605±0.060 | logreg, k=200 — 0.699/0.699/0.699; fold 0.703±0.109 / 0.683±0.119 | bilstm, seq=18 — 0.545/0.544/0.545; fold 0.533±0.067 / 0.497±0.080 | Classical + elapsed |
| OPTI | linearSVC, k=200 — 0.729/0.729/0.729; fold 0.711±0.099 / 0.699±0.094 | linearSVC, k=80 — 0.750/0.750/0.750; fold 0.733±0.112 / 0.720±0.116 | transformer, seq=18 — 0.751/0.751/0.751; fold 0.732±0.110 / 0.711±0.139 | DL |
| XSENS | logreg, k=80 — 0.516/0.516/0.517; fold 0.526±0.109 / 0.464±0.103 | logreg, k=80 — 0.667/0.667/0.667; fold 0.656±0.091 / 0.616±0.106 | bilstm, seq=18 — 0.530/0.515/0.530; fold 0.552±0.085 / 0.442±0.067 | Classical + elapsed |
| OE+OPTI | linearSVC, k=80 — 0.725/0.725/0.725; fold 0.708±0.101 / 0.702±0.097 | linearSVC, k=80 — 0.740/0.740/0.740; fold 0.726±0.115 / 0.712±0.119 | bilstm, seq=18 — 0.719/0.718/0.719; fold 0.722±0.084 / 0.714±0.083 | Classical + elapsed |
| OE+XSENS | logreg, k=200 — 0.553/0.551/0.554; fold 0.563±0.077 / 0.514±0.066 | logreg, k=80 — 0.662/0.660/0.663; fold 0.667±0.095 / 0.631±0.101 | transformer, seq=18 — 0.523/0.520/0.523; fold 0.539±0.120 / 0.487±0.098 | Classical + elapsed |
| OPTI+XSENS | linearSVC, k=80 — 0.720/0.720/0.720; fold 0.705±0.096 / 0.700±0.092 | linearSVC, k=80 — 0.740/0.740/0.740; fold 0.725±0.114 / 0.713±0.118 | transformer, seq=18 — 0.663/0.663/0.663; fold 0.651±0.130 / 0.634±0.141 | Classical + elapsed |
| OE+OPTI+XSENS | linearSVC, k=80 — 0.720/0.720/0.720; fold 0.703±0.099 / 0.698±0.096 | linearSVC, k=80 — 0.736/0.736/0.736; fold 0.722±0.114 / 0.709±0.117 | gru, seq=18 — 0.634/0.632/0.634; fold 0.653±0.122 / 0.632±0.149 | Classical + elapsed |

### 7.4 Task 2: Group Activity Recognition (RQ2)

#### 7.4.1 Task 2a: Conversation vs. Non-Conversation
Easiest subtask. Best: OptiTrack Transformer, A=0.880, M=0.845. Notable: OE+elapsed reaches
M=0.781, its strongest showing in the basic ablation — consistent with ear-worn head-motion
carrying conversation-related information.

**Table 7.3 — Task 2a (conversation vs. non-conversation)**:

| Sensor | Classical, no elapsed | Classical, with elapsed | DL, no elapsed | Best (M) |
|---|---|---|---|---|
| OE | logreg, k=80 — 0.718/0.689/0.701; fold 0.708±0.161 / 0.660±0.150 | logreg, k=80 — 0.807/0.781/0.787; fold 0.765±0.192 / 0.690±0.214 | transformer, seq=9 — 0.737/0.647/0.643; fold 0.720±0.173 / 0.600±0.167 | Classical + elapsed |
| OPTI | logreg, k=80 — 0.842/0.819/0.825; fold 0.844±0.057 / 0.802±0.088 | linearSVC, k=80 — 0.861/0.843/0.855; fold 0.832±0.130 / 0.789±0.148 | transformer, seq=9 — 0.880/0.845/0.844; fold 0.873±0.089 / 0.760±0.192 | DL |
| XSENS | logreg, k=80 — 0.672/0.628/0.632; fold 0.641±0.115 / 0.568±0.133 | logreg, k=80 — 0.812/0.790/0.802; fold 0.772±0.165 / 0.712±0.190 | transformer, seq=9 — 0.739/0.650/0.646; fold 0.724±0.124 / 0.575±0.132 | Classical + elapsed |
| OE+OPTI | linearSVC, k=80 — 0.831/0.808/0.816; fold 0.830±0.056 / 0.784±0.081 | logreg, k=80 — 0.841/0.819/0.826; fold 0.815±0.125 / 0.766±0.138 | transformer, seq=9 — 0.880/0.836/0.819; fold 0.874±0.092 / 0.762±0.182 | DL |
| OE+XSENS | logreg, k=200 — 0.702/0.672/0.686; fold 0.695±0.178 / 0.656±0.165 | logreg, k=200 — 0.779/0.752/0.761; fold 0.751±0.167 / 0.689±0.191 | transformer, seq=9 — 0.737/0.633/0.625; fold 0.707±0.203 / 0.546±0.180 | Classical + elapsed |
| OPTI+XSENS | logreg, k=80 — 0.842/0.819/0.825; fold 0.844±0.057 / 0.802±0.088 | linearSVC, k=80 — 0.861/0.843/0.855; fold 0.832±0.130 / 0.789±0.148 | transformer, seq=9 — 0.876/0.833/0.821; fold 0.866±0.076 / 0.747±0.180 | Classical + elapsed |
| OE+OPTI+XSENS | linearSVC, k=80 — 0.831/0.808/0.816; fold 0.830±0.056 / 0.784±0.081 | logreg, k=80 — 0.841/0.819/0.826; fold 0.815±0.125 / 0.766±0.138 | transformer, seq=9 — 0.880/0.836/0.819; fold 0.874±0.092 / 0.762±0.182 | DL |

#### 7.4.2 Task 2b: Conversation vs. Co-Building
Deep models dominate whenever OptiTrack present. Best: BiLSTM on OPTI+XSENS, M=0.860; close
behind: OE+OPTI (0.859), full three-sensor (0.855).

**Table 7.4 — Task 2b (conversation vs. co-building)**:

| Sensor | Classical, no elapsed | Classical, with elapsed | DL, no elapsed | Best (M) |
|---|---|---|---|---|
| OE | logreg, k=200 — 0.701/0.688/0.697; fold 0.673±0.206 / 0.603±0.186 | logreg, k=80 — 0.783/0.768/0.771; fold 0.748±0.171 / 0.645±0.202 | transformer, seq=9 — 0.756/0.697/0.689; fold 0.725±0.177 / 0.611±0.178 | Classical + elapsed |
| OPTI | logreg, k=80 — 0.830/0.817/0.819; fold 0.829±0.061 / 0.752±0.140 | logreg, k=80 — 0.848/0.838/0.845; fold 0.827±0.101 / 0.742±0.166 | gru, seq=9 — 0.868/0.845/0.847; fold 0.861±0.100 / 0.739±0.205 | DL |
| XSENS | logreg, k=80 — 0.644/0.634/0.649; fold 0.621±0.185 / 0.562±0.191 | logreg, k=80 — 0.719/0.715/0.740; fold 0.694±0.229 / 0.616±0.246 | transformer, seq=9 — 0.628/0.615/0.656; fold 0.628±0.124 / 0.534±0.140 | Classical + elapsed |
| OE+OPTI | logreg, k=80 — 0.816/0.803/0.807; fold 0.809±0.061 / 0.732±0.127 | logreg, k=80 — 0.832/0.822/0.828; fold 0.813±0.104 / 0.727±0.159 | bilstm, seq=9 — 0.881/0.859/0.859; fold 0.870±0.070 / 0.773±0.141 | DL |
| OE+XSENS | linearSVC, k=80 — 0.624/0.615/0.631; fold 0.606±0.218 / 0.543±0.207 | linearSVC, k=80 — 0.735/0.717/0.721; fold 0.713±0.160 / 0.630±0.163 | transformer, seq=9 — 0.719/0.648/0.641; fold 0.664±0.278 / 0.518±0.230 | Classical + elapsed |
| OPTI+XSENS | logreg, k=80 — 0.830/0.817/0.819; fold 0.829±0.061 / 0.752±0.140 | logreg, k=80 — 0.848/0.838/0.845; fold 0.827±0.101 / 0.742±0.166 | bilstm, seq=9 — 0.881/0.860/0.863; fold 0.868±0.103 / 0.748±0.205 | DL |
| OE+OPTI+XSENS | logreg, k=80 — 0.816/0.803/0.807; fold 0.809±0.061 / 0.732±0.127 | logreg, k=80 — 0.832/0.822/0.828; fold 0.813±0.104 / 0.727±0.159 | transformer, seq=9 — 0.881/0.855/0.845; fold 0.868±0.101 / 0.720±0.222 | DL |

#### 7.4.3 Task 2c: Conversation vs. Co-Merging
Also well separated. Best: Transformer on OE+OPTI, M=0.866; all OptiTrack-containing deep models
exceed M=0.82.

**Table 7.5 — Task 2c (conversation vs. co-merging)**:

| Sensor | Classical, no elapsed | Classical, with elapsed | DL, no elapsed | Best (M) |
|---|---|---|---|---|
| OE | logreg, k=80 — 0.739/0.707/0.723; fold 0.765±0.163 / 0.674±0.169 | logreg, k=200 — 0.797/0.768/0.780; fold 0.819±0.153 / 0.700±0.156 | bilstm, seq=9 — 0.664/0.657/0.680; fold 0.707±0.187 / 0.630±0.194 | Classical + elapsed |
| OPTI | logreg, k=80 — 0.827/0.797/0.803; fold 0.804±0.153 / 0.719±0.229 | logreg, k=80 — 0.845/0.814/0.814; fold 0.847±0.100 / 0.744±0.210 | gru, seq=9 — 0.833/0.828/0.850; fold 0.787±0.194 / 0.698±0.251 | DL |
| XSENS | logreg, k=200 — 0.613/0.553/0.556; fold 0.623±0.199 / 0.503±0.195 | linearSVC, k=200 — 0.741/0.674/0.667; fold 0.749±0.202 / 0.625±0.217 | gru, seq=9 — 0.610/0.591/0.597; fold 0.574±0.263 / 0.472±0.241 | Classical + elapsed |
| OE+OPTI | logreg, k=80 — 0.822/0.791/0.795; fold 0.794±0.147 / 0.710±0.221 | logreg, k=80 — 0.860/0.835/0.838; fold 0.855±0.094 / 0.755±0.210 | transformer, seq=9 — 0.879/0.866/0.859; fold 0.898±0.063 / 0.775±0.190 | DL |
| OE+XSENS | logreg, k=200 — 0.655/0.621/0.638; fold 0.664±0.194 / 0.571±0.178 | linearSVC, k=200 — 0.701/0.660/0.670; fold 0.721±0.204 / 0.615±0.176 | transformer, seq=9 — 0.626/0.609/0.616; fold 0.645±0.229 / 0.451±0.149 | Classical + elapsed |
| OPTI+XSENS | logreg, k=80 — 0.827/0.797/0.803; fold 0.804±0.153 / 0.719±0.229 | logreg, k=80 — 0.845/0.814/0.814; fold 0.847±0.100 / 0.744±0.210 | transformer, seq=9 — 0.868/0.852/0.843; fold 0.883±0.067 / 0.794±0.155 | DL |
| OE+OPTI+XSENS | logreg, k=80 — 0.822/0.791/0.795; fold 0.794±0.147 / 0.710±0.221 | logreg, k=80 — 0.860/0.835/0.838; fold 0.855±0.094 / 0.755±0.210 | transformer, seq=9 — 0.874/0.861/0.857; fold 0.896±0.090 / 0.828±0.178 | DL |

#### 7.4.4 Task 2d: Co-Merging vs. Co-Building
**Hardest pairwise boundary.** Best: OptiTrack Transformer, M=0.738 — "roughly ten points below
the other pairwise subtasks." Discriminative info is largely spatial (clustering at central table
for merging); explains OptiTrack dominance and poor OE/Xsens performance.

**Table 7.6 — Task 2d (co-merging vs. co-building)** (caption in source has no space:
"Table7.6:"):

| Sensor | Classical, no elapsed | Classical, with elapsed | DL, no elapsed | Best (M) |
|---|---|---|---|---|
| OE | linearSVC, k=200 — 0.788/0.689/0.706; fold 0.749±0.142 / 0.605±0.111 | logreg, k=200 — 0.736/0.667/0.730; fold 0.756±0.234 / 0.606±0.209 | gru, seq=9 — 0.665/0.571/0.588; fold 0.667±0.303 / 0.452±0.189 | Classical, no elapsed |
| OPTI | logreg, k=80 — 0.730/0.653/0.704; fold 0.730±0.141 / 0.617±0.150 | logreg, k=80 — 0.723/0.640/0.685; fold 0.719±0.158 / 0.608±0.158 | transformer, seq=9 — 0.811/0.738/0.752; fold 0.766±0.126 / 0.591±0.159 | DL |
| XSENS | logreg, k=80 — 0.607/0.526/0.565; fold 0.559±0.259 / 0.428±0.178 | linearSVC, k=80 — 0.646/0.527/0.541; fold 0.581±0.284 / 0.424±0.197 | gru, seq=9 — 0.542/0.447/0.452; fold 0.499±0.315 / 0.356±0.202 | Classical + elapsed |
| OE+OPTI | logreg, k=200 — 0.731/0.636/0.668; fold 0.706±0.197 / 0.571±0.158 | linearSVC, k=80 — 0.703/0.607/0.639; fold 0.678±0.245 / 0.531±0.160 | transformer, seq=9 — 0.729/0.659/0.694; fold 0.705±0.257 / 0.554±0.215 | DL |
| OE+XSENS | logreg, k=200 — 0.668/0.586/0.631; fold 0.618±0.211 / 0.512±0.184 | linearSVC, k=200 — 0.615/0.505/0.522; fold 0.591±0.280 / 0.463±0.224 | transformer, seq=9 — 0.644/0.442/0.447; fold 0.486±0.326 / 0.381±0.248 | Classical, no elapsed |
| OPTI+XSENS | linearSVC, k=80 — 0.709/0.615/0.648; fold 0.646±0.220 / 0.517±0.170 | logreg, k=200 — 0.711/0.599/0.618; fold 0.646±0.234 / 0.504±0.193 | transformer, seq=9 — 0.816/0.718/0.709; fold 0.741±0.203 / 0.559±0.209 | DL |
| OE+OPTI+XSENS | logreg, k=200 — 0.756/0.640/0.652; fold 0.713±0.221 / 0.577±0.176 | linearSVC, k=200 — 0.737/0.617/0.629; fold 0.687±0.249 / 0.551±0.191 | transformer, seq=9 — 0.762/0.652/0.653; fold 0.694±0.225 / 0.501±0.195 | DL |

#### 7.4.5 Task 2e: Three-Class Activity Recognition
Best: LSTM on OE+OPTI, A=0.756, M=0.699; OptiTrack-only (0.767/0.695) and OPTI+XSENS
(0.765/0.693) statistically indistinguishable (all within the ±0.13 fold SD). Note on
non-determinism: a repeat run selected LSTM instead of BiLSTM as best deep model, moving pooled
macro-F1 0.690→0.699 — "an order of magnitude smaller than the fold-level standard deviation."
Adding OE to OptiTrack does not improve the 3-class result; Xsens alone weakest (M=0.399, no
elapsed). Accuracy/macro-F1 gap reflects class imbalance (minority co_merging class dominates
errors), consistent with Task 2d difficulty.

**Table 7.7 — Task 2e (three-class recognition)**:

| Sensor | Classical, no elapsed | Classical, with elapsed | DL, no elapsed | Best (M) |
|---|---|---|---|---|
| OE | logreg, k=200 — 0.552/0.518/0.556; fold 0.540±0.244 / 0.463±0.176 | logreg, k=200 — 0.640/0.607/0.654; fold 0.640±0.238 / 0.540±0.221 | bilstm, seq=9 — 0.598/0.489/0.475; fold 0.588±0.209 / 0.408±0.201 | Classical + elapsed |
| OPTI | logreg, k=80 — 0.695/0.653/0.688; fold 0.713±0.102 / 0.603±0.124 | linearSVC, k=80 — 0.727/0.628/0.630; fold 0.721±0.102 / 0.558±0.112 | lstm, seq=9 — 0.767/0.695/0.677; fold 0.750±0.113 / 0.554±0.152 | DL |
| XSENS | logreg, k=80 — 0.458/0.399/0.410; fold 0.435±0.182 / 0.333±0.127 | logreg, k=80 — 0.567/0.527/0.554; fold 0.534±0.193 / 0.439±0.166 | transformer, seq=9 — 0.552/0.463/0.470; fold 0.530±0.171 / 0.349±0.112 | Classical + elapsed |
| OE+OPTI | logreg, k=80 — 0.695/0.653/0.687; fold 0.713±0.102 / 0.602±0.122 | linearSVC, k=80 — 0.722/0.623/0.625; fold 0.714±0.099 / 0.549±0.107 | bilstm, seq=9 — 0.716/0.651/0.664; fold 0.744±0.137 / 0.578±0.132 | Classical, no elapsed |
| OE+XSENS | linearSVC, k=80 — 0.507/0.469/0.495; fold 0.503±0.198 / 0.431±0.157 | linearSVC, k=200 — 0.525/0.497/0.539; fold 0.520±0.188 / 0.430±0.157 | gru, seq=9 — 0.458/0.421/0.455; fold 0.461±0.218 / 0.356±0.093 | Classical + elapsed |
| OPTI+XSENS | logreg, k=80 — 0.695/0.653/0.688; fold 0.713±0.102 / 0.603±0.124 | linearSVC, k=80 — 0.727/0.628/0.630; fold 0.721±0.102 / 0.558±0.112 | gru, seq=9 — 0.762/0.684/0.674; fold 0.748±0.110 / 0.555±0.152 | DL |
| OE+OPTI+XSENS | logreg, k=80 — 0.695/0.653/0.687; fold 0.713±0.102 / 0.602±0.122 | linearSVC, k=80 — 0.722/0.623/0.625; fold 0.714±0.099 / 0.549±0.107 | bilstm, seq=9 — 0.716/0.651/0.664; fold 0.744±0.137 / 0.578±0.132 | Classical, no elapsed |

### 7.5 Dedicated OpenEarable Experiments

Basic ablation tables above could suggest ear-worn sensing is weak; dedicated developed-feature
experiments show this would be misleading.

#### 7.5.1 OpenEarable Conversation Detection

Isolated OE-only conversation-detection experiment shows OE is much stronger for conversation
detection than detailed activity separation. With elapsed time, OE-only LogReg reaches M=0.781;
without elapsed, RBF-SVC reaches 0.711, GRU sequence model 0.722.

**Table 7.8 — OpenEarable-only conversation vs. non-conversation results**:

| Condition | Model | Selection | A | M | B | fold mean±SD (A/M) |
|---|---|---|---|---|---|---|
| Classical OE, no elapsed | RBF-SVC (C=1) | all / 305 feat. | 0.7339 | 0.7108 | 0.7302 | 0.726±0.174 / 0.661±0.188 |
| Classical OE, with elapsed | LogReg (C=1) | k=80 / 306 feat. | 0.8075 | 0.7810 | 0.7873 | 0.765±0.192 / 0.690±0.214 |
| DL OE, no elapsed | GRU seq=3, 30s | k=200 | 0.7618 | 0.7221 | 0.7266 | 0.755±0.174 / 0.688±0.179 |

(Note: GRU row's "seq=3, 30s" implies a different windowing for this isolated experiment —
3 segments of 10s each ≈ 30s context, distinct from the seq=9/18 five-second-window convention
used elsewhere. Flag for verification.)

#### 7.5.2 Developed OE Feature Sets for Three-Class Recognition

Best purely OE-only result (**OE9**: motion features + magnetometer magnitude) reaches M=0.592, a
clear improvement over basic OE (Table 7.7, M=0.518). Adding a small number of spatial cues
improves further to M=0.640 (3 proximity features + elapsed time) — but these configs are no
longer OE-only; OptiTrack-based configs remain stronger overall.

**Table 7.9 — Developed OpenEarable three-class results (RBF-SVC, C=1, gamma=scale)**:

| Condition | Features | A | M | B | fold mean±SD (A/M) | Interpretation |
|---|---|---|---|---|---|---|
| OE9 motion + MAG magnitude | 305 | 0.6340 | 0.5920 | 0.6260 | 0.641±0.192 / 0.526±0.131 | Best OE-only |
| OE best + manual OptiTrack | 315 | 0.6620 | 0.6150 | 0.6420 | 0.678±0.173 / 0.560±0.120 | Manual spatial cues; not OE-only |
| OE best + ENG7 proximity | 308 | 0.6710 | 0.6270 | 0.6530 | 0.680±0.175 / 0.555±0.139 | Three proximity features; not OE-only |
| OE best + ENG7 proximity + elapsed | 309 | 0.6850 | 0.6400 | 0.6640 | 0.691±0.167 / 0.558±0.128 | Best OE-contextual |

#### 7.5.3 SPECIAL_OE Binary Interaction Detection

SPECIAL_OE alone reaches M=0.635 (no elapsed) / M=0.700 (with elapsed) — matching the best basic
OE result; SPECIAL_OE + relative OptiTrack reaches M=0.752 with elapsed.

**Table 7.10 — SPECIAL_OE binary interaction detection results**:

| Condition | Time | Model | Sel. | A | M | B | fold mean±SD (A/M) |
|---|---|---|---|---|---|---|---|
| SPECIAL_OE | no elapsed | LogReg (C=1) | k=200 | 0.6360 | 0.6350 | 0.6370 | 0.640±0.053 / 0.605±0.060 |
| SPECIAL_OE | with elapsed | LogReg (C=1) | k=200 | 0.7000 | 0.7000 | 0.7000 | 0.704±0.108 / 0.684±0.118 |
| SPECIAL_OE + OPTI2 relative | no elapsed | LinearSVC (C=1) | k=80 | 0.7290 | 0.7290 | 0.7290 | 0.713±0.105 / 0.707±0.102 |
| SPECIAL_OE + OPTI2 relative | with elapsed | LinearSVC (C=1) | k=80 | 0.7520 | 0.7520 | 0.7520 | 0.736±0.116 / 0.724±0.119 |

Note: the table caption says the final column notes "the unusually small spread of the OE-only
configuration without elapsed time" (fold SD 0.053/0.060, visibly tighter than most other cells
in this chapter).

Conclusion of Section 7.5: developed OE features are most convincing for binary interaction and
conversation detection; for three-class recognition, spatial OptiTrack-derived features remain
strongest for separating co-building from co-merging.

### 7.6 Sensitivity to Researcher Participation

In 4 of 9 groups (**G1, G7, G8, G9**) the author was the third participant — not fully naive.
Tested by evaluating the headline configuration of every recognition experiment on the **5 fully
naive groups only (G2, G3, G5, G6, G10)**, which contain **204.6 of the 381.4 minutes** of
annotated data (**54%**). Same pipeline/features/in-fold scaling/selection; only the group set
changes → **LOGO produces 5 folds instead of 9**.

**Table 7.11 — Full nine-group cohort vs. five fully naive groups (G2, G3, G5, G6, G10),
OptiTrack feature family.** A=accuracy, M=macro-F1, both pooled across folds. Deep column =
best of BiLSTM/Transformer at k=120, no elapsed:

| Task | Classical no-elapsed (full9 A/M) | Classical no-elapsed (naive5 A/M) | Classical +elapsed (full9 A/M) | Classical +elapsed (naive5 A/M) | Deep no-elapsed (full9 A/M) | Deep no-elapsed (naive5 A/M) |
|---|---|---|---|---|---|---|
| Task 1: interaction | 0.729/0.729 | 0.713/0.713 | 0.750/0.750 | 0.730/0.727 | 0.751/0.751 | 0.610/0.609 |
| Task 2a: conversation vs. rest | 0.842/0.819 | 0.885/0.857 | 0.861/0.843 | 0.885/0.861 | 0.880/0.845 | 0.901/0.860 |
| Task 2b: conversation vs. building | 0.830/0.817 | 0.839/0.819 | 0.848/0.838 | 0.854/0.837 | 0.875/0.852 | 0.798/0.757 |
| Task 2c: conversation vs. merging | 0.827/0.797 | 0.904/0.882 | 0.845/0.814 | 0.932/0.918 | 0.833/0.828 | 0.869/0.867 |
| Task 2d: merging vs. building | 0.730/0.653 | 0.867/0.693 | 0.723/0.640 | 0.870/0.710 | 0.811/0.738 | 0.819/0.591 |
| Task 2e: three-class | 0.695/0.653 | 0.740/0.644 | 0.727/0.628 | 0.787/0.671 | 0.767/0.695 | 0.677/0.558 |

Key takeaway: naive-only subset is NOT uniformly worse — conversation boundaries actually improve
(2a: 0.819→0.857; 2c: 0.797→0.882 classical no-elapsed); Task 1 essentially unchanged
(0.729 vs 0.713 no-elapsed; 0.750 vs 0.727 with-elapsed). Three-class is the only setting clearly
harder on naive subset **for deep models** (0.695→0.558), though its classical result is close
(0.628 vs 0.671 with-elapsed). Explanation offered: removing 4 groups removes 46% of windows and
4 of 9 training folds — co_merging class drops to 51 windows across 5 groups — more consistent
with data-quantity effect than researcher-driven leakage; classical models (fewer params, per-fold
feature selection) far less sensitive.

**Table 7.12 — Headline configuration of each of the six recognition experiments (+ Task 1
ablation config as comparator), evaluated on the five naive groups.** Fold columns = mean ± SD of
per-group macro-F1 with 95% CI (t₈=2.306 for 9 folds, t₄=2.776 for 5):

| Experiment | Config | Full9 A | Full9 M | Full9 fold mean±SD [CI] | Naive5 A | Naive5 M | Naive5 fold mean±SD [CI] | ΔM |
|---|---|---|---|---|---|---|---|---|
| Task 1: interaction (optimised) | OPTI2_REL transf. s=18 | 0.806 | 0.806 | 0.781±0.086 [0.716,0.847] | 0.706 | 0.706 | 0.689±0.081 [0.589,0.789] | −0.100 |
| Task 1: interaction (ablation) | OPTI transf. s=18 | 0.751 | 0.751 | 0.711±0.139 [0.604,0.818] | 0.627 | 0.626 | 0.579±0.169 [0.369,0.788] | −0.125 |
| Task 2a: conv. vs rest | OPTI transf. s=9 | 0.880 | 0.845 | 0.760±0.192 [0.612,0.908] | 0.883 | 0.825 | 0.759±0.136 [0.590,0.928] | −0.020 |
| Task 2b: conv. vs building | OPTI+XSENS bilstm s=9 | 0.881 | 0.860 | 0.748±0.205 [0.590,0.906] | 0.741 | 0.706 | 0.698±0.226 [0.417,0.978] | −0.154 |
| Task 2c: conv. vs merging | OE+OPTI transf. s=9 | 0.879 | 0.866 | 0.775±0.190 [0.629,0.921] | 0.672 | 0.587 | 0.571±0.347 [0.140,1.000] | −0.279 |
| Task 2d: merging vs building | OPTI transf. s=9 | 0.811 | 0.738 | 0.591±0.159 [0.469,0.713] | 0.609 | 0.403 | 0.403±0.181 [0.179,0.627] | −0.335 |
| Task 2e: three-class | OE+OPTI lstm s=9 | 0.756 | 0.699 | 0.578±0.132 [0.477,0.679] | 0.540 | 0.513 | 0.470±0.184 [0.242,0.698] | −0.186 |

**Table 7.13 — Per-group (per-fold) macro-F1 for the configurations of Table 7.12, on the five
naive groups**:

| Experiment | G2 | G3 | G5 | G6 | G10 | pooled |
|---|---|---|---|---|---|---|
| Task 1: interaction (optimised) | 0.743 | 0.627 | 0.743 | 0.578 | 0.753 | 0.706 |
| Task 1: interaction (ablation) | 0.666 | 0.829 | 0.474 | 0.407 | 0.519 | 0.626 |
| Task 2a: conversation vs rest | 0.882 | 0.922 | 0.612 | 0.664 | 0.715 | 0.825 |
| Task 2b: conversation vs building | 0.882 | 0.931 | 0.389 | 0.561 | 0.725 | 0.706 |
| Task 2c: conversation vs merging | 0.438 | 0.432 | 0.132 | 0.930 | 0.921 | 0.587 |
| Task 2d: merging vs building | 0.172 | 0.443 | 0.423 | 0.662 | 0.313 | 0.403 |
| Task 2e: three-class | 0.263 | 0.766 | 0.406 | 0.452 | 0.464 | 0.513 |

Key notes: Group 2 has **zero co_merging windows**, so its Task 2d/2c fold is single-class and
macro-F1 collapses accordingly (0.172, 0.438). Only **51 co_merging windows** remain across all 5
naive groups. Confidence intervals extremely wide (Task 2c spans [0.14, 1.00]). Ordering of task
difficulty preserved across cohorts (conversation-vs-rest easiest, co-merging-vs-co-building
hardest in both). Nine-group values in Table 7.12 agree with the ablation tables "to within 0.0004
macro-F1" (i.e., Table 7.12's full9 column is a consistency check against Tables 7.2–7.7, not an
independent number). Deep-model figures in Table 7.11 vs Table 7.12 differ by up to 0.02
macro-F1 — attributed to run-to-run non-determinism (Section 7.4.5), an order of magnitude
smaller than fold SDs.

### 7.7 Summary of Key Results (prose, verbatim structure)

Three findings:
1. Wearable/spatial sensor data supports meaningful group activity recognition under
   group-independent evaluation: interaction detection reaches **macro-F1 0.81** (optimized
   setting), conversation-related boundaries reach **macro-F1 0.84–0.87**, three-class recognition
   reaches **macro-F1 0.69**.
2. Sensor ablation identifies **OptiTrack-derived spatial features as the most reliable
   modality**; deep sequence models typically strongest when spatial features present, classical
   +elapsed strongest otherwise; hardest boundary = co-merging vs. co-building (depends on
   spatial clustering info).
3. Ear-worn sensing should not be dismissed: with developed feature sets OpenEarable becomes
   genuinely informative for conversation and interaction detection, though it cannot replace
   spatial sensing for fine-grained activity separation.

Further support: evaluating every headline config on the 5 naive groups alone (Section 7.6)
reproduces the qualitative pattern, though magnitudes shift by several points in both directions.
Cross-cutting pattern: temporal information helps in every regime where added — elapsed time
lifts window-independent classical models; sequence-aware deep models beat window-independent
classifiers whenever the representation is spatially informative. "Across all tasks, the
decisive factor is the quality of the group-aware representation rather than model complexity
alone; temporal context then amplifies what a good representation provides."

---

## 6. Results — Task 3 forecasting (Chapter 8) — SECOND REPRODUCTION TARGET

### 8.1 Motivation and task definition
RQ3: can the group's next activity state be predicted from recent activity history and/or sensor
evidence? Fundamentally different from recognition — target is future-oriented. Central
methodological theme: **evaluation honesty** — because activities persist across many consecutive
windows, a trivial "predict next = current" strategy scores highly under naive all-window
evaluation but says nothing about anticipation ability. Experiments therefore separate
**persistence-dominated all-window performance** from **transition-only performance** (next
activity differs from current). Task 3 framed throughout as **exploratory**, not solved
supervised classification. All experiments use LOGO evaluation.

### 8.2 Data representations — four representations, exact example counts

1. **Window-level history-aware data, 7-label next-window prediction.** History length 1 →
   **2,071 examples**, of which **275 are true transitions** and **1,796 are non-transitions.**
   Used to expose the persistence problem.
2. **Segment-level 6-label data**: consecutive same-label windows collapsed into one segment →
   **244 activity segments**: conversation 95, inspection 45, building 43, object_handover 28,
   moving_transport 22, merging 11.
3. **Tokenized 6-label activity sequences**: each segment = one token. Full-statistics variant:
   each token carries a **337-dimensional raw sensor-statistics vector** before in-fold selection.
4. **Expanding-prefix data** for next-segment prediction from full session history: fine 13-label
   granularity (**n = 287**) and coarse 4-label granularity (**n = 216**).

Segment distribution strongly imbalanced (95 conversation vs. 11 merging segments) → macro-F1
more informative than accuracy throughout this chapter.

### 8.3 Model families

**Table 8.1 — Model families evaluated for next-activity forecasting**:

| Method family | What it does | Why it matters |
|---|---|---|
| Repeat-current / persistence | Predicts next label = current label | Strong on all windows; invalid for transition anticipation (always fails on transitions) |
| N-gram / Markov back-off | Uses recent label history to estimate next label; no-self variant forces a different next label | Best simple transition-only window baseline; later, the strongest overall predictor |
| Sensor-history logistic regression | Uses lagged engineered sensor summaries without label history | Tests whether sensors alone contain predictive signal for upcoming transitions |
| Segment feature forecasting | Forecasts next segment feature vector, decodes to a label | Tests whether the future sensor pattern can be forecast before the next activity starts |
| Tokenized Transformer / LSTM | Represents each segment as an activity token; full-stat version adds sensor summaries to each token | Best neural sequence result for 6-label segment tokens |
| HMM / transition models | Learns transition dynamics with sensor or categorical emissions | Useful for analysing transitions; smoothing results must be separated from causal prediction |
| Expanding-prefix neural models | Predicts next segment from full past sequence (suffix back-off, LSTM, CNN1D, Transformer) | Shows coarse next-state forecasting is more feasible than fine-grained forecasting |

### 8.4 The Persistence Problem: All-Window vs. Transition-Only Evaluation

Repeat-current baseline: **0.8672 accuracy, 0.7425 macro-F1** on all windows — "entirely explained
by label persistence." Restricted to true transitions, repeat-current necessarily drops to
**zero** (always predicts no change). Best transition-only model: **no-self Markov back-off,
history length 5**, reaching **0.4704 accuracy, 0.2402 macro-F1**. Sensor-only logistic regression
(lagged summaries) reaches **0.2184 macro-F1** on transitions — "sensor history contains some
predictive signal but is far from sufficient."

**Table 8.2 — Window-level 7-label next-window prediction: all-window vs. transition-only
evaluation** (95% CI uses t₈=2.306):

| Model | Hist. | Scope | n | A | M | B | macro-F1 fold mean±SD [CI] |
|---|---|---|---|---|---|---|---|
| repeat_current_label | 1 | all windows | 2071 | 0.8672 | 0.7425 | 0.7425 | 0.639±0.104 [0.559,0.720] |
| logreg, label history only | 1 | all windows | 2071 | 0.8672 | 0.7425 | 0.7425 | 0.639±0.104 [0.559,0.720] |
| n-gram Markov back-off | 1 | all windows | 2071 | 0.8629 | 0.6925 | 0.6957 | 0.608±0.090 [0.538,0.677] |
| logreg, label+sensor history | 1 | all windows | 2071 | 0.8160 | 0.6766 | 0.7021 | 0.546±0.074 [0.490,0.603] |
| logreg, sensor history only | 5 | all windows | 2035 | 0.3622 | 0.2592 | 0.2887 | 0.181±0.075 [0.123,0.239] |
| n-gram Markov, no-self back-off | 5 | transition only | 270 | 0.4704 | 0.2402 | 0.2801 | 0.213±0.107 [0.131,0.296] |
| logreg, sensor history only | 1 | transition only | 275 | 0.2218 | 0.2184 | 0.2595 | 0.139±0.074 [0.082,0.196] |
| logreg, label+sensor history | 5 | transition only | 270 | 0.1296 | 0.1158 | 0.1159 | 0.091±0.049 [0.054,0.129] |
| repeat_current_label | 1 | transition only | 275 | 0.0000 | 0.0000 | 0.0000 | 0.000±0.000 |

### 8.5 Segment-Level Feature Forecasting

Pipeline: **24 engineered features**, collapse same-label consecutive windows into segments,
train **Ridge or Random Forest** regressors to forecast the next feature vector, apply a
**logistic-regression decoder** to map forecasted features → labels. Labels used only in the
final decoder, not as forecasting input. Feature forecasting does NOT clearly beat the label
Markov segment baseline: best Ridge forecast M=0.2742, below label-only baseline at M=0.2919.
Oracle rows show even decoding the true next feature vector is limited — difficulty is not only
the forecasting model but the discriminative power of segment-level features and small sample
size.

**Table 8.3 — Segment-level 6-label forecasting results**:

| Model | Hist. | n | A | M | B | fold mean±SD [CI] | Comment |
|---|---|---|---|---|---|---|---|
| Label majority baseline | 1 | 235 | 0.3702 | 0.0901 | 0.1667 | — | Simple majority baseline |
| Label Markov segment baseline | 1 | 235 | 0.5106 | 0.2919 | 0.3775 | — | Best simple label-only baseline |
| Forecast features, decode (Ridge) | 1 | 235 | 0.3021 | 0.2742 | 0.3074 | 0.163±0.084 [0.099,0.228] | Best real feature-forecasting model |
| Forecast features, decode (RF) | 2 | 226 | 0.3230 | 0.1679 | 0.2060 | 0.077±0.030 [0.054,0.100] | Higher accuracy, lower macro-F1 |
| Oracle: decode true next features (Ridge) | 2 | 226 | 0.3186 | 0.2753 | 0.3167 | 0.188±0.093 [0.117,0.260] | Ceiling when true next features are decoded |

(Dashes "—" indicate per-fold scores not stored for that baseline.)

### 8.6 Tokenized Transformer over Activity Segments

Each activity segment = one token. Labels-only setting: token = previous activity identity only.
Labels+sensors setting: token also carries full statistical summary per channel — **mean, SD,
min, max, range, median, IQR, 10th/25th/75th/90th percentiles, energy, RMS, entropy** — creating
244 tokens × 337-dim raw stats vector. **SelectKBest with k=40** applied inside each LOGO training
fold only (avoid leakage); neural results **averaged over three seeds**.

**Table 8.4 — Tokenized 6-label next-activity results (LOGO, three seeds where applicable)**:

| Model | Token setting | A | M | Interpretation |
|---|---|---|---|---|
| Segment Markov | activity labels | 0.481 | 0.285 | Baseline; conversation F1 high, several classes weak |
| Transformer, labels only | activity tokens only | 0.464 ± 0.018 | 0.332 ± 0.021 | Previous labels without sensor token features |
| Transformer, labels + full-stat sensors | tokens, in-fold k=40 | 0.563 ± 0.014 | 0.424 ± 0.020 | Best tokenized Transformer result |
| LSTM, labels + sensors | full-stat tokens, in-fold k=40 | 0.482 ± 0.012 | 0.340 ± 0.021 | Lower than the Transformer with the same tokens |

Full-stat labels+sensors Transformer = best neural 6-label result (M=0.424±0.020), clear
improvement over labels-only (0.332±0.021). Importantly: tokenization itself defeats the
persistence artefact since consecutive identical windows are collapsed.

### 8.7 Fine vs. Coarse Next-Segment Forecasting

Expanding-prefix experiments test whether full activity history observed so far can predict the
next segment, at two granularities.

**Table 8.5 — Expanding-prefix next-segment prediction at fine (13-label) and coarse (4-label)
granularity. W-F1 = weighted F1** (95% CI uses t₈=2.306):

| Labels | Features | Predictor | n | A | M | W-F1 | macro-F1 fold mean±SD [CI] |
|---|---|---|---|---|---|---|---|
| coarse_4 | activity only | CNN1D | 216 | 0.657 | 0.564 | 0.684 | 0.503±0.077 [0.444,0.562] |
| coarse_4 | activity only | suffix back-off | 216 | 0.708 | 0.561 | 0.703 | 0.506±0.086 [0.440,0.572] |
| coarse_4 | activity only | Transformer | 216 | 0.653 | 0.557 | 0.665 | 0.522±0.106 [0.441,0.603] |
| coarse_4 | activity only | Markov-last | 216 | 0.722 | 0.409 | 0.624 | 0.395±0.061 [0.348,0.442] |
| fine_13 | activity only | Transformer | 287 | 0.254 | 0.178 | 0.274 | 0.104±0.044 [0.070,0.138] |
| fine_13 | activity only | suffix back-off | 287 | 0.355 | 0.169 | 0.337 | 0.126±0.063 [0.077,0.175] |
| fine_13 | activity+segment features | Transformer | 287 | 0.345 | 0.173 | 0.346 | 0.121±0.059 [0.075,0.167] |

Coarse 4-label is the most promising next-segment setup (macro-F1 ≈0.56 for CNN1D and suffix
back-off). Fine 13-label remains difficult — too few segment transitions per label; adding
segment-level sensor features to prefix models did not clearly improve best macro-F1.

### 8.8 HMM and Next-State Experiments

Two important observations: (1) all-window 3-class next-window setting again
persistence-dominated (0.955 accuracy for repeat/Markov) — not a genuine transition result;
(2) **Viterbi smoothing is not causal prediction**: best 5-class smoothing score M=0.599
reconstructs the timeline using future information, while causal HMM prediction is lower
(M=0.546), and honest transition-only HMM variants much lower still (macro-F1 ≤ 0.301). HMMs
presented as analysis tools revealing structured transition dynamics, not as the winning
forecasting approach.

**Table 8.6 — HMM and next-state experiments**:

| Experiment | Model | n | A | M | Interpretation |
|---|---|---|---|---|---|
| 3-class next-window, all windows | repeat / Markov | 1773 | 0.955 | 0.955 | Persistence-dominated |
| 3-class next-window, transition only | Markov-transition | 79 | 0.886 | 0.627 | Fair opponent that forces a different state |
| 3-class next-window, transition only | Gaussian HMM | 79 | 0.304 | 0.301 | Exploratory; weaker than forced-transition Markov |
| 5-class collective state, all windows | HMM Viterbi smoothing | 732 | 0.643 | 0.599 | Best smoothing score; not causal prediction |
| 5-class collective state, all windows | HMM causal prediction | 732 | 0.585 | 0.546 | Causal HMM close to LogReg, below Viterbi |
| 5-class collective state, transition only | HMM categorical | 259 | 0.282 | 0.265 | Best transition-only HMM variant |
| 5-class collective state, transition only | HMM sensor emissions | 259 | 0.282 | 0.202 | Sensor emissions did not outperform categorical |

### 8.9 Grammar-Order Experiments: Short-Order Activity Grammar — **THE WINNING RESULT**

Follow-up round on the same 244 activity tokens: tested whether the earlier comparison had
handicapped classical temporal models by restricting their history. It had — given proper
token-level history, **a simple back-off n-gram Markov model over the last two-to-three activity
labels becomes the strongest next-activity predictor overall, beating the tokenized Transformer
with sensors.**

Three findings:
1. **The grammar is short-order and dominant.** Performance peaks at history 2–3 activities
   (h=2: 0.604 accuracy; h=3: 0.513 macro-F1), declines for h=5, h=10, full history — 244 tokens
   cannot populate longer contexts. Independently-implemented suffix-back-off Markov reproduces
   this (max order 3: 0.587/0.489); second-order HMM confirms via probabilistic filtering (0.606
   accuracy).
2. **Sensors cannot substitute for labels, and add little on top.** Sensor-only-emission
   second-order HMM reaches only 0.389/0.194 — reflects the discriminability ceiling of token
   features. Hybrid model (n-gram grammar prior × sensor-based next-label classifier, mixing
   weight λ chosen in-fold) selected **λ between 0.6 and 1.0 (mostly 0.9–1.0)** and did NOT exceed
   the pure n-gram (0.583/0.495 vs. 0.583/0.513). Latent-state HMM variants with sensor clusters
   likewise did not beat label-only counterparts.
3. **The Transformer result is honestly reframed.** Tokenized Transformer's earlier headline
   (0.563/0.424) is reinterpreted: with 244 tokens it never learned the second-order grammar that
   simple counting extracts (its labels-only variant reached just 0.332 macro-F1) — its apparent
   "sensor gain" was largely compensation for that failure. A history-length sweep confirms
   full-sequence attention adds nothing beyond the last few tokens at this data size.

**Table 8.7 — Grammar-order experiments: 6-label next-activity prediction over 244 tokens
(LOGO)** (95% CI uses t₈=2.306 where per-fold scores stored):

| Model | A | M | macro-F1 fold mean±SD [CI] | Information used |
|---|---|---|---|---|
| n-gram Markov, h=2 (back-off) | 0.604 | 0.499 | 0.426±0.087 [0.359,0.493] | last 2 labels |
| n-gram Markov, h=3 (back-off) | 0.583 | 0.513 | 0.435±0.113 [0.348,0.521] | last 3 labels |
| Second-order HMM (categorical) | 0.606 | 0.443 | 0.393±0.054 [0.351,0.434] | last 2 labels, probabilistic |
| Hybrid: n-gram × sensor→next (λ in-fold) | 0.583 | 0.495 | — | labels+sensors |
| Tokenized Transformer + full-stat sensors | 0.563 | 0.424 | — | full history + sensors |
| Latent HMM, 12 states, label-only (seed avg.) | ~0.57–0.60 | ~0.42–0.46 | — | labels, latent states |
| Second-order HMM (sensor emissions) | 0.389 | 0.194 | 0.175±0.041 [0.143,0.206] | sensors only |
| First-order segment Markov | 0.481 | 0.285 | — | last 1 label |

### 8.10 Sensitivity to Researcher Participation (Task 3)

Same 4 researcher groups (**G1, G7, G8, G9**) excluded; forecasting evaluated on 5 naive groups
(**G2, G3, G5, G6, G10**). The activity-token grammar retains its characteristic shape: peaks at
history 2 (pooled M=0.516), degrades for longer histories exactly as in the full cohort. The
persistence result is reproduced: repeat-current reaches pooled M=0.763 across all windows and
exactly 0.000 on transitions in every fold; no-self n-gram reaches pooled M=0.238 on transitions.
**Naive subset contains 111 transitions in 942 next-window prediction examples (11.8%)** — five
fewer than the 947 labelled windows of Table 8.8, because the final window of each group has no
successor — close to the 13% observed on the full cohort, so "the persistence problem is a
property of the activity structure rather than of researcher participation."

**Table 8.8 — Task 3 on the five naive groups: per-group (per-fold) macro-F1, fold mean ± SD, 95%
CI for the mean (t₄=2.776).** Upper block = activity-token back-off grammar at history lengths h
(116 tokens for the naive subset, vs. 244 for the full cohort); lower block = persistence analysis
on 5-second process labels (947 labelled windows, 111 transitions):

| Configuration | G2 | G3 | G5 | G6 | G10 | mean ± SD | 95% CI |
|---|---|---|---|---|---|---|---|
| n-gram back-off, h=1 | 0.190 | 0.278 | 0.234 | 0.121 | 0.074 | 0.179 ± 0.083 | [0.077, 0.282] |
| n-gram back-off, h=2 | 0.335 | 0.492 | 0.492 | 0.366 | 0.417 | 0.420 ± 0.072 | [0.331, 0.509] |
| n-gram back-off, h=3 | 0.335 | 0.468 | 0.440 | 0.188 | 0.306 | 0.347 ± 0.112 | [0.208, 0.487] |
| n-gram back-off, h=5 | 0.334 | 0.362 | 0.451 | 0.205 | 0.306 | 0.331 ± 0.090 | [0.220, 0.442] |
| repeat-current, all windows | 0.488 | 0.661 | 0.774 | 0.582 | 0.759 | 0.653 ± 0.121 | [0.503, 0.803] |
| no-self n-gram, transitions | 0.190 | 0.278 | 0.234 | 0.121 | 0.074 | 0.179 ± 0.083 | [0.077, 0.282] |
| repeat-current, transitions | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 | — |

(Note: "n-gram back-off, h=1" and "no-self n-gram, transitions" rows are numerically identical in
the source table — both 0.190/0.278/0.234/0.121/0.074 — this appears to be intentional, i.e. h=1
back-off IS the no-self-transition baseline at that history length, not a duplication error, but
flagged for a human to confirm against the PDF.)

### 8.11 Chapter Conclusion (verbatim summary)

Task 3 should be framed as exploratory next-activity forecasting, not a solved classification
problem. All-window results provide context; transition-only and segment-level results are "the
honest scientific evidence." **Headline Task 3 result: the back-off n-gram over activity tokens —
0.604 accuracy / 0.499 macro-F1 at h=2 (0.513 macro-F1 at h=3)** — the strongest, simplest, most
interpretable next-activity predictor, confirmed independently by a second-order HMM. Predictive
signal = a short-order activity grammar over the last 2–3 activities; longer history and
full-sequence attention add nothing at this data size. Sensors do not substitute for activity
labels and add at most marginal value on top of the grammar (hybrid λ≈0.9–1.0); the Transformer's
earlier apparent sensor contribution reflected its failure to learn the grammar, not unique
sensor information. Tokenized Transformer remains the best neural method (tokenization defeats
the persistence artefact) but is not the headline predictor. Coarse next-segment forecasting
(macro-F1 ≈0.56 at 4-label granularity) more promising than fine 13-label forecasting
(macro-F1 ≈0.18).

---

## 7. Table 10.1 — researcher-participant vs. naive cohort comparison (Chapter 10, Limitations, Section 10.5)

### Context
Section 10.5 "Researcher Participation in Four Groups" restates: in 4 of 9 groups (**Groups 1, 7,
8, 9**), the author took part as third participant because a recruited participant did not attend
or could not be found (Section 3.5). Same protocol/sensors as other participants, but familiar
with task design and research goals — could plausibly influence group behaviour (e.g., moving
through phases more efficiently, implicitly guiding the group), so these 4 groups "cannot be
treated as fully naive." Effect on results bounded two ways: (1) LOGO means each such group is
also tested fully held-out; (2) activity labels describe observable behaviour, not participant
intent. Tested directly via evaluating every headline config (incl. optimized Task 1 Transformer)
on the 5 naive groups (Section 7.6) — qualitative conclusions unchanged (spatial OptiTrack
dominant, elapsed time helps, conversation easiest / co-merging-vs-co-building hardest,
forecasting/persistence findings reproduce). Several classical scores are actually higher on the
naive subset (argues against systematic researcher-participation advantage); deep models lose
ground but also lose 46% of training data and 4 of 9 folds — more plausibly a sample-size effect.
Thesis explicitly recommends: "a replication in which all groups consist entirely of naive
participants, and which is large enough to train sequence models on that cohort alone, would
strengthen the findings."

### Descriptive comparison methodology
All 9 sessions compared directly from ELAN annotations (raw annotation boundaries, NOT the
5-second modelling windows). Activity labels normalised onto the Section 4.6 "process categories"
first (raw vocabulary has many spelling variants that would otherwise register as spurious
activity changes). **Caveat explicitly stated**: the 3 conceptual collaboration phases (Section
3.4: planning/orientation, individual/sub-piece building, joint assembly/merging) were **never
annotated as explicit phase boundaries in ELAN** — exact phase durations cannot be recovered
retrospectively, and no proxy is substituted. What IS compared: duration of the annotated
*activity categories* (not the 3 conceptual phases).

**Table 10.1 — Descriptive comparison of the researcher-participant and fully naive cohorts,
computed from the ELAN annotations after label normalisation.** Session length here is derived
from the ELAN annotation timeline (latest annotation end time) and therefore differs slightly
from the recording durations of Table 3.1. Transition counts summed over all 7 annotation tiers
(3 individuals + 3 pairs + whole-group; **6 tiers for G3** — an explicit exception noted in the
caption, presumably one tier had no annotations for that group). Per-tier rate = total/7 (or /6
for G3): **0.67 changes/min for researcher cohort, 0.47 changes/min for naive cohort** (≈1 change
every 90s vs. every 128s respectively). p-values are two-sided Mann–Whitney U tests, 4 vs. 5
groups, reported "for completeness only" — smallest attainable p at this sample size is 0.016:

| Group | Cohort | Session (min) | Transitions (total) | Transitions per min | Individual build (%) | Dominant share |
|---|---|---|---|---|---|---|
| G1 | researcher | 30.5 | 134 | 4.39 | 61.0 | 0.607 |
| G7 | researcher | 49.7 | 245 | 4.93 | 55.1 | 0.583 |
| G8 | researcher | 38.8 | 179 | 4.61 | 62.9 | 0.630 |
| G9 | researcher | 60.0 | 283 | 4.72 | 62.9 | 0.600 |
| G2 | naive | 48.2 | 216 | 4.48 | 66.7 | 0.667 |
| G3 | naive | 45.2 | 91 | 2.02 | 79.7 | 0.797 |
| G5 | naive | 68.2 | 216 | 3.17 | 51.6 | 0.516 |
| G6 | naive | 22.9 | 78 | 3.41 | 63.1 | 0.631 |
| G10 | naive | 25.0 | 74 | 2.96 | 84.3 | 0.843 |
| researcher mean | — | 44.8 | 210.3 | 4.66 | 60.5 | 0.605 |
| naive mean | — | 41.9 | 135.0 | 3.21 | 69.1 | 0.691 |
| U / p | — | 12.0 / 0.73 | 16.0 / 0.18 | 19.0 / 0.03 | — | 4.0 / 0.19 |

### Interpretation (verbatim findings)
- **No evidence of session-length difference** (44.8 vs 41.9 min, U=12, p=0.73) — but with 4/5
  groups the test has very little power; this should not be read as establishing equivalence.
- **Groups including the author changed activity more frequently**: 4.66 vs 3.21 transitions/min
  summed across all 7 tiers (= 0.67 vs 0.47 changes/min on a single tier); **U=19, p=0.03**
  (the only individually significant comparison in the table, though caveated by small N).
- Spent a **smaller share of the session in uninterrupted individual building**: 60.5% vs 69.1%,
  with correspondingly more co-building, conversation, and movement.
- This is argued NOT to be an annotation-granularity artefact: if researcher sessions were simply
  segmented more finely, every category would show shorter mean annotation durations. They do
  not: travelling (3.7 vs 3.6 s), object handover (2.4 vs 2.7 s), collaborative building (32.1 vs
  33.3 s), task conversation (9.4 vs 8.9 s) are all closely comparable between cohorts. **The only
  category that differs is individual building**: uninterrupted episodes last **27.9 s** average in
  researcher groups vs **43.7 s** in naive groups.
- Conclusion: pattern is "consistent with a genuine behavioural difference between the
  cohorts—in particular more frequent coordination and shorter uninterrupted periods of individual
  work in the researcher-participant groups." The available sample cannot determine whether this
  difference was caused specifically by the author's familiarity with the [task/study — sentence
  is cut off at the end of the extracted chunk read for this document; verify final clause against
  PDF].

---

## 8. Cross-cutting notes

### 8.1 Naming conventions / terms to map to actual Drive files, folders, columns (verbatim as found in text)

**ELAN annotation tier names** (exact, underscore-joined):
- `Participant1`, `Participant2`, `Participant3`
- `Participant1_Participant2`, `Participant1_Participant3`, `Participant2_Participant3`
- `Whole_Group`

**Time / synchronization columns** (Table 4.2):
- `video_time_s` (shared time axis, seconds relative to video start — used twice in the table with
  apparently conflicting descriptions, see flag in Section 2 above)
- `datetime_utc`
- `SampleTimeFine` (Xsens-specific)
- `frame` (OptiTrack frame index)
- `time_s` (time relative to sensor recording start)

**Modelling target label strings** (from Chapter 4/6/Appendix A):
- `interaction` / `non_interaction` (Task 1 target)
- `conversation`, `co_building`, `co_merging` (Task 2 target classes — note underscore in
  `co_building`/`co_merging` but NOT in `conversation`)
- `individual_build` (grouped label, distinguished from `co_building`)

**Appendix A fine-grained label examples grouped into cleaned categories** (Table A.1, partial —
only what was captured in the read range; more rows likely continue beyond what was extracted for
this pass):
- individual building ← `individual_build`, `building`, `assembling`, `working_on_piece`
- co-building ← `co_building`, `co_building_subpiece`, `building_together`, `collaborative_building`
- conversation ← `conversation`, `task_conversation`, `discussion`, `planning`, `strategy_talk`
- co-merging ← `co_merging`, `merging`, `combining_pieces`, `joining_subpieces`
- inspection ← `inspection`, `inspecting`, `checking_piece`, `looking_at_piece`
- target-image checking ← `matching_pieces_to_target_image`, `matching_pieces_with_target_image`,
  `checking_target_image`
- picking up ← `pick_up`, `picking_up_piece`, `take_piece`
- putting down / placing ← `put_down`, `placing_piece`, `drop_piece`
- moving or placing pieces ← `move_piece`, `moving_piece`, `repositioning`, `placing`
- object handover ← `object_handover` (only this exact label included)
- carrying / delivering ← `carrying`, `delivering`, `bringing_piece`, `transporting_piece`
- travelling / approaching ← `approaching`, (list continues beyond the read excerpt)

**Feature-set / configuration codenames** (used as literal config identifiers in results tables —
these must map to actual pipeline feature-set names in the reproduction repo):
- `OPTI2_RELATIVE_ONLY` — relative OptiTrack feature set, the winning Task 1 configuration
- `OE9` — developed OpenEarable feature set: "motion features + magnetometer magnitude"
- `OE10` — mentioned alongside OE9 as a "developed OpenEarable configuration" (≈305 features) but
  never independently defined/tabulated in the body text as read — verify in appendix or source
  code.
- `SPECIAL_OE` — dedicated OpenEarable feature set for binary interaction detection
- `ENG7` — a 3-feature "proximity" feature addition used in the OE-contextual three-class
  experiments (Table 7.9) — name implies 7 total engineered proximity features exist in this
  family, only 3 used here.
- Sensor-combination shorthand used throughout Ch.7/Ch.6: `OE`, `OPTI`, `XSENS`, `OE+OPTI`,
  `OE+XSENS`, `OPTI+XSENS`, `OE+OPTI+XSENS`

**Model / method identifiers** used in table cells: `logreg` (Logistic Regression), `linearSVC`,
`RBF-SVC`, Random Forest, ExtraTrees (classical); `lstm`, `bilstm`, `gru`, `transformer` (deep,
lowercase in table cells); `n-gram Markov back-off`, `no-self back-off`, `HMM` variants
(categorical / sensor-emissions / Viterbi-smoothing / causal-prediction), `suffix back-off`,
`CNN1D`, `Markov-last` (forecasting-specific).

**Scalers/selectors**: `RobustScaler` (feature scaling, fit per-fold on training groups only),
`SelectKBest` (univariate feature selection, various k grids — see Section 4 above).

**No literal file/script/notebook paths or filenames appear anywhere in the extracted thesis
text.** The thesis describes the pipeline conceptually (stages, formulas, feature families) but
does not name actual source files, scripts, or notebooks. A reproduction pipeline will need to
invent its own file/module naming — these codenames above (`OPTI2_RELATIVE_ONLY`, `OE9`,
`SPECIAL_OE`, `ENG7`, etc.) are the only "canonical" identifiers given in the source and should be
preserved verbatim if the actual Drive data/scripts use them (worth checking against the Drive
inventory files already in this staging folder: `drive_inventory.md`, `thesis_inventory.md`).

### 8.2 Discrepancies flagged (not silently resolved)

1. **Table 3.1 vs Table 10.1 session durations differ per group** (e.g. G2: 47.1 min in Table 3.1
   vs 48.2 min in Table 10.1; G7: 48.3 vs 49.7; G9: 59.4 vs 60.0). The thesis itself explains this
   is because Table 10.1 uses ELAN-annotation-timeline end time while Table 3.1 uses raw recording
   duration — NOT a contradiction, but a reproduction pipeline must decide which one to treat as
   ground truth for "session duration" depending on the metric being reproduced.
2. **Abstract headline numbers round the more precise in-chapter numbers**: abstract says "0.81
   macro-F1" for Task 1 (actual: 0.8062) and "0.69 macro-F1" for three-class (actual: 0.6994) —
   consistent rounding, not a real discrepancy, but flagged since the goal sheet asked to check
   for this.
3. **Table 8.8 naive-cohort grammar table**: "n-gram back-off, h=1" and "no-self n-gram,
   transitions" rows are byte-for-byte identical (0.190/0.278/0.234/0.121/0.074 across all 5
   groups). This may be intentional (h=1 back-off degenerates to the no-self-transition rule) but
   was not explicitly confirmed in the surrounding prose — flagged for human verification against
   the PDF.
4. **OE10** is named in Section 5.10 prose ("developed OE9/OE10 configurations") but no table in
   the extracted text separately defines or reports OE10 results — only OE9 appears in Table 7.9.
   Either OE10 results are in a part of the document not covered by the specific table anchors
   given, or OE10 is only used elsewhere (e.g. appendix, or dropped from the final table set).
   Flag for the pipeline builder to search the full PDF for "OE10" specifically.
5. **Table 4.2 `video_time_s` double-definition** — see Section 2 above; the two rows of Table 4.2
   sharing the exact column name `video_time_s` with different descriptions is very likely a
   pdfplumber row-merge/OCR artifact rather than the thesis intentionally listing the same column
   twice with different meanings. Needs PDF-level verification.
6. Section 4.8.1 and 4.8.3 both contain a **sentence truncated mid-word ("This 50" / "the 50")**
   immediately after stating the interaction-window overlap rule — almost certainly "This 50%
   threshold..." / "...the 50% threshold is applied conservatively..." but the extraction cuts off
   the percent sign and the rest of the sentence in both places. The formula elsewhere in the same
   section (T_interaction/T_window ≥ 0.5) removes ambiguity about the numeric threshold itself, but
   the exact wording of the "ambiguous window" handling rule (Section 4.8.3) is NOT fully
   recovered — flag for PDF verification.

### 8.3 Tables/passages where extraction looked garbled or uncertain — human should double-check against the PDF

- **Table 4.2** (time-related columns) — duplicate `video_time_s` row with conflicting
  description (see above).
- **Table 5.2** (OpenEarable feature families) — Pitch/Roll formula rows contain visible PDF
  glyph-substitution artifacts (`(cid:0)`, stray `q`/`p` characters standing in for square-root
  symbols); the general form is recoverable but exact axis order needs PDF verification.
- **Table 5.3** (Xsens feature families) — same square-root/norm formula rendering issue as Table
  5.2 (acceleration norm / gyroscope norm rows).
- **Section 4.8.1 and 4.8.3** — truncated "This 50[%]..." sentences (see discrepancy #6 above).
- **Table 7.8, GRU row ("seq=3, 30s")** — this window/sequence convention (3 segments × 10s ≈ 30s)
  does not match the seq=9/seq=18 five-second-window convention used everywhere else in Chapter 7;
  worth confirming this isolated OE-only experiment used a genuinely different windowing scheme
  rather than a transcription inconsistency.
- **Table 8.8 upper block header "(116 tokens)"** — worth confirming 116 is exactly half of the
  full-cohort 244 tokens (used to sanity-check that this is a straightforward subset-of-groups
  operation, not a re-tokenization with different rules).
- Every occurrence of `104:` `0.104` -style page-anchor artifacts (i.e. the small subscript "8" or
  "4" that appears attached to "t =" for the confidence-interval degrees-of-freedom notation,
  e.g. "t = 2.306" printed across two lines as "t\n8" in the extraction) — these were manually
  resolved to "t₈=2.306" / "t₄=2.776" in this document based on the stated fold counts (9 groups →
  8 df; 5 groups → 4 df), but the raw extraction renders them ambiguously and should be spot
  checked.

### 8.4 Not covered in this document (out of requested scope, but noted for completeness)
- Chapter 2 (Related Work / literature review) — not requested.
- Chapter 9 (Discussion) — only skimmed for the opening "Main Finding" framing (used in Section 5
  headline framing); full Section 9.2–9.9 prose not transcribed here since the goal sheet did not
  request it, but it contains useful interpretive context (e.g. Section 9.2 "What the Sensor
  Ablation Reveals About Each Modality", Section 9.6 "Difficulty of Co-Merging") that a pipeline
  author may want to read directly from the PDF/extraction (lines ~3396–3660 of
  `extracted_report.txt`) when writing up final repo documentation.
- Appendix A's full fine-label-to-group mapping table (Table A.1) continues beyond what was read
  in this pass (only the first ~12 grouped categories were captured above); the remaining rows
  (further into "travelling/approaching" and beyond) should be pulled from
  `extracted_report.txt` starting around line 4119 if a complete label dictionary is needed for
  the annotation-cleaning step of the pipeline.
- Appendix C (per-group score tables, referenced repeatedly as "Appendix C.1" / "Table C.1") was
  not located/transcribed in this pass — it would give the raw 9-fold per-group scores behind
  several pooled tables in Chapter 7 and is worth pulling separately if per-fold ground truth is
  needed for validation beyond what Table 7.13 already provides for the naive-5 subset.
