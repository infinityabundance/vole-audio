# PHASE N — transport / archive

## Mission

The contract's deliverable list (§52):

* finalized canonical archive;
* event/checkpoint transport;
* integrity;
* recovery;
* reproducible manifests.

Phase M answered *what the material costs*. Phase N answers *how an authoritative
bundle of it is stored and carried*: one explicit binary container for the frozen
objects and their session, one deterministic framing for state/events/checkpoints,
one integrity story for both, and a recovery policy that never pretends an xrun
did not happen.

The phase is bound by §30 ("write explicit encoding/decoding", "no unchecked
`usize` conversions from file lengths", "no recursive allocation bombs") and §36
("transport procedural/state information, not mandatory PCM", "do not build a
giant networking stack").

## What is in this increment

### The canonical `.volea` archive (§30, §31)

`src/format/archive.rs`. Explicit little-endian binary; **no serde/bincode layout
is normative**.

```text
Archive :=
    MAGIC            12   b"vole.archive"
    VERSION          u8    = 1
    PROFILE_LEN      u8    1..=MAX_TAG
    PROFILE          bytes e.g. b"u1/v1"
    UNIVERSE_LEN     u8    1..=MAX_TAG
    UNIVERSE         bytes e.g. b"vole.audio.u1"
    SECTION_COUNT    u32   1..=MAX_SECTIONS
    SECTION_TABLE    SECTION_COUNT × SectionRecord
    SECTIONS         contiguous payloads, in table order
    ARCHIVE_DIGEST   32    SHA-256 over everything before it

SectionRecord :=
    KIND      u8    SectionKind
    RESERVED  u8    = 0
    OFFSET    u64   absolute payload offset
    LENGTH    u64   payload length
    SHA256    32    payload digest

Manifest :=
    ENTRY_COUNT       u32
    ENTRY_COUNT × Entry          ordered by name, strictly ascending
    EVENT_COUNT       u32
    CHECKPOINT_COUNT  u32
    DEPENDENCY_COUNT  u32

Entry :=
    NAME_LEN        u8   1..=MAX_NAME
    NAME            bytes printable ASCII, no separators
    REPRESENTATION  u8   the U1 representation tag
    CHANNELS        u8
    SAMPLE_RATE_HZ  u32
    FRAMES          u64
    CONTENT_ID      32   SHA-256 of the canonical object bytes
```

Section kinds are carried **explicitly** in the table. Canonical layout the
decoder enforces: section 0 is `MANIFEST`, then `OBJECT` sections in manifest-entry
order, then `EVENT`, `CHECKPOINT` and `DEPENDENCY` sections in manifest-count
order; payload offsets are contiguous and start immediately after the table; no
trailing bytes; no unknown kind in v1. Unknown kinds are **rejected, never
skipped** — the unknown-extension policy is fail-closed.

| payload | canonical form | identity |
| ------- | -------------- | -------- |
| `OBJECT` | canonical U1 object bytes, or a full-object container | entry `CONTENT_ID` must equal SHA-256 of the bytes |
| `EVENT` | `transport::event::encode_event` (19 bytes) | section digest |
| `CHECKPOINT` | `transport::checkpoint::encode_checkpoint` | section digest |
| `DEPENDENCY` | exactly a 32-byte content id | section digest |

`encode_archive` refuses non-canonical manifests (empty, unsorted, bad names),
object/manifest count mismatches, malformed session payloads, and any entry whose
content id does not match the bytes it is about to store (typed `Integrity`).
`decode_archive` re-validates every one of those plus every digest and the archive
digest, and re-validates the session payloads semantically.

**Dependencies are declarations.** A `DEPENDENCY` section names a required content
id and is integrity-bound, but carries **no byte accounting** and makes no
amortization claim (§31): an external dependency's bytes are never counted as
zero. The archive court records this as an explicit limitation.

### Deterministic transport (§36, §37)

`src/transport/`. The frame model carries `OBJECT`, `EVENT`, `STATE`, `CHECKPOINT`,
`DEPENDENCY`, `CLOCK` and `INTEGRITY` frames: **procedural/state information, never
mandatory PCM**. The literal fallback rides as an `OBJECT` payload.

```text
Frame :=
    KIND         u8    FrameKind
    FLAGS        u8    bit0 = LAST; other bits must be 0
    EPOCH        u32
    SEQUENCE     u64   monotonic within an epoch
    MEDIA_FRAME  i64   timeline anchor (-1 = not tied to a frame)
    PAYLOAD_LEN  u32   0..=MAX_FRAME_PAYLOAD_BYTES
    PAYLOAD      bytes
    DIGEST       32    SHA-256 over KIND..PAYLOAD
```

`TransportReceiver` is the ordered, bounded state machine. Every decision is an
`Outcome` a receipt can record:

| input | decision |
| ----- | -------- |
| sequence already consumed | `Duplicate` (dropped, never replayed) |
| sequence jump | `SequenceGap` (missing count recorded, accepted forward) |
| older epoch | `StaleEpoch` (dropped) |
| newer epoch from a non-CLOCK frame | `StaleEpoch` — only a `CLOCK` frame may re-anchor |
| newer epoch from a `CLOCK` frame | `Clock { resync: true }` (explicit epoch advance) |
| event before the cursor | `Event { late: true }` |
| checkpoint disagreeing with the clock | `Checkpoint { resync: true }`, then re-anchor |
| dependency declaration | tracked until an `OBJECT` satisfies it |

Resource bounds are explicit: `MAX_FRAME_PAYLOAD_BYTES`, `MAX_FRAMES_PER_STREAM`,
`MAX_PENDING_EVENTS`, `MAX_CHECKPOINT_STATE_BYTES`. Every length is converted with
`try_from`, never cast (`encode_frame`/`encode_checkpoint` included). Pending
events have an explicit `consume_events` release path; a checkpoint resets the
buffer too.

`verify_integrity_frame` proves a stream's closing `INTEGRITY` frame attests the
body that precedes it, and that it is the final frame.

### Recovery (§37)

`src/universe/clock.rs` (frozen in Phase B) plus `transport::clock::MediaClock`.
An xrun is never a silent reset; each policy records exactly what it did:

| policy | timeline | observation | epoch |
| ------ | -------- | ----------- | ----- |
| `PreserveTimeline` | continues; missed frames skipped, none fabricated | continuous | unchanged |
| `Discontinuity` | media frame unchanged; explicit silence inserted | discontinuity | unchanged |
| `RestartEpoch` | restarts at frame 0 | discontinuity | +1 |

The `RecoveryOutcome` carries `skipped_frames` / `inserted_frames` /
`preserves_timeline` / `inserts_discontinuity` / `restarts_epoch`, so a receipt can
state which of the three the architecture chose.

### Reproducible manifests (§52)

`src/format/manifest.rs` renders the archive's catalogue as deterministic,
newline-delimited text (objects sorted by name; session entries in frozen section
order):

```text
vole.audio.manifest.v1
profile <tag>
universe <tag>
entry <name> <representation> <channels> <rate> <frames> <content_id_hex>
events <count>
event <index> <payload_sha256_hex>
checkpoints <count>
checkpoint <index> <payload_sha256_hex>
dependencies <count>
dependency <content_id_hex>
```

Its SHA-256 is the reproducible manifest digest; the archive digest covers the
bytes. Two archives are equal iff both agree.

## Courts

`court archive` builds one `.volea` bundle over the **frozen corpus**: each
object's full-object container as an integrity-bound `OBJECT` payload, plus a
deterministic session (one timed event per object in frozen order, one end
checkpoint, one declared external dependency). It proves deterministic encoding,
byte-exact round trip, decode → re-encode canonicality, and a **resealed** hostile
battery (truncation, bit flip, reserved byte, unknown kind, swapped kinds, offset
gap, content-id forgery, manifest count lie, session kind reorder, malformed event
payload, malformed checkpoint, trailing bytes, garbage, allocation bomb) in which
structural mutations recompute the section and archive digests so they reach the
content validators rather than failing the outer digest.

`court transport` builds one frame stream over the frozen corpus (dependencies,
object containers, timed events, a checkpoint, a clock epoch, a closing integrity
attestation), verifies the round trip and attestation, drives the receiver through
every classification path, runs the hostile framing battery (unknown kind/flags,
oversized payload bomb, corruption, truncation, forged attestation, duplicate,
stale epoch, gap, late event) and records all three xrun policies on identical
clocks.

`court phase-n` is the aggregate: it re-runs both and is `SUPPORTED` only when both
are.

## Seal ledger

| seal | release | subject | contents |
| ---- | ------- | ------- | -------- |
| 1 | v0.23.0 | `70cc330a…` | canonical archive + event/checkpoint/dependency sections, deterministic transport + receiver, clock recovery, reproducible manifests, `court archive` (`81db84de…`), `court transport` (`782a46b8…`), `court phase-n` (`92df23d2…`). 26-row `seal verify` matrix; 468 passed / 12 ignored all-features, 458 / 12 default. Device artifacts byte-identical (`d13d22c3…` PTX, `5c30a4bc…` AMDGPU). |

Archive seal evidence: 115 objects, 115 events, 1 checkpoint, 1 dependency;
31,089,591 payload bytes in a 31,113,564 B archive; manifest digest `826f98d6…`,
archive digest `919096c8…`; hostile battery 10 integrity + 10 structural
rejections. Transport seal evidence: 348 frames, 115 resolved objects, 115 events
(0 late), 1 checkpoint resync, 10 hostile rejections, all three recovery policies.

## Non-claims

* This is **not** a network stack. It is deterministic framing plus an ordered,
  bounded receiver; there is no transport security, congestion, retransmission or
  session negotiation claim.
* The archive does not reinterpret `OBJECT` payloads; their identity is their
  SHA-256.
* Declared `DEPENDENCY` content ids carry no byte accounting and no amortization
  claim.
* Wall-clock timing is not part of any Phase-N static result hash.
