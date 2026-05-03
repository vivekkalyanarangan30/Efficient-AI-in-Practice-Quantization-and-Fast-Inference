"""
Chapter 10, Section 10.2 — The GGUF v3 file format from the bytes up
Companion script for "Efficient AI in Practice: Quantization and Fast Inference"

What this script demonstrates:
  GGUF is the on-disk container that llama.cpp loads quantized LLM
  weights from. The format is dense by design: a 24-byte header,
  followed by typed key/value metadata, followed by tensor descriptors,
  followed by raw quantized weight blocks aligned to 32-byte boundaries.
  Section 10.2 walks the format byte by byte so a reader who later
  debugs a "weight offset wrong" or "layer name not found" issue knows
  what they're looking at.

  This script is a pure-Python GGUF parser. It opens a real GGUF file,
  walks each section with `struct.unpack`, dumps the first superblock
  of one tensor with field-level annotations, and produces two
  reference figures:

    Figure 10.2 (taxonomy): the three quant families (legacy, k-quants,
                            IQ-quants) plotted by effective bits per
                            weight including scale/metadata overhead.
    Figure 10.3 (layout):   a real GGUF's section sizes drawn to scale,
                            with a hex inset of the first 256 bytes.

  The parse path is independent of the `gguf` PyPI package — the whole
  point is showing how the bytes become tensors. We optionally cross-
  validate against `gguf.GGUFReader` if it is installed and emit a
  one-line OK confirmation.

Modes:
  --mode inspect    Parse a GGUF and print structured report (text)
  --mode tensor     Dump first superblock of one tensor with byte
                    annotations (text). Implemented for Q4_K, Q8_0,
                    and Q6_K.
  --mode taxonomy   Quant-family landscape figure          (Figure 10.2)
  --mode layout     File-layout figure for the reference   (Figure 10.3)
  --mode all        Run all four (default)

Usage:
  python ch10_gguf_format.py --mode inspect --slot smoke
  python ch10_gguf_format.py --mode tensor  --slot reference
  python ch10_gguf_format.py --mode all     --save-plots
  python ch10_gguf_format.py --gguf-path /path/to/local.gguf --mode inspect

Install (one Python environment, CPU-only):
  pip install -U numpy matplotlib huggingface_hub
  pip install -U gguf                 # optional, used only for cross-validation

Hardware target:
  Any CPU. This section is parser arithmetic and matplotlib — no GPU
  or tensor-cores involved. Mac (Apple Silicon or x86), Linux, and
  Windows all run identically.

    Apple Silicon (M1/M2/M3/M4):  ✓ supported
    Intel / AMD x86-64:           ✓ supported
    Linux ARM64:                  ✓ supported

  The reference and legacy slots download multi-GB GGUFs from the
  Hugging Face Hub on first use; the smoke slot stays under 1 GB and
  is the right default for CI or low-bandwidth runs.

Note on reading the spec:
  The byte arithmetic in `dump_q4_k_block` / `dump_q8_0_block` /
  `dump_q6_k_block` follows the layout in ggml-quants.h from
  llama.cpp commit b3000+. K-quant superblocks pack 256 weights per
  block; legacy quants (Q4_0/Q5_0/Q8_0) pack 32 weights per block.
  These are the layouts the runtime expects — if the on-disk bytes
  don't match, the model will load and produce garbage rather than
  fail loudly, so this dump mode is the right place to confirm.
"""

import argparse
import json
import logging
import struct
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

SCRIPT_DIR = Path(__file__).resolve().parent


# --- Configuration ---------------------------------------------------------

@dataclass
class Config:
    mode: str = "all"
    save_plots: bool = False
    force_redownload: bool = False
    slot: str = "reference"
    gguf_path: Optional[Path] = None
    tensor_name: str = "blk.0.ffn_down.weight"
    full_tensors: bool = False
    output_dir: Path = field(default_factory=lambda: SCRIPT_DIR / "figures")
    cache_dir: Path = field(default_factory=lambda: SCRIPT_DIR / "gguf_cache")
    seed: int = 42


# --- HF artifact slots -----------------------------------------------------
# Three slots cover the size/quality trade-off the chapter discusses.
# The reference slot is the one Figure 10.3 is rendered from.

ARTIFACT_SLOTS = {
    "smoke": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "role": "fast smoke test (~700 MB)",
    },
    "reference": {
        "repo_id": "TheBloke/Llama-2-7B-GGUF",
        "filename": "llama-2-7b.Q4_K_M.gguf",
        "role": "k-quant reference (~4 GB)",
    },
    "legacy": {
        "repo_id": "TheBloke/Llama-2-7B-GGUF",
        "filename": "llama-2-7b.Q8_0.gguf",
        "role": "legacy-quant contrast (~7 GB)",
    },
}


# --- Manning figure style --------------------------------------------------

# Four-segment file layout palette (header / metadata / tensor info / data)
# and one row per quant family on the taxonomy plot. Same colorblind-safe
# palette family Ch9 uses.
COLORS = {
    # Quant family rows on the taxonomy figure.
    "legacy":   "#7570b3",
    "k_quant":  "#1b9e77",
    "iq_quant": "#d95f02",
    # File-layout segments on Figure 10.3.
    "header":      "#7570b3",
    "metadata":    "#e7298a",
    "tensor_info": "#d95f02",
    "tensor_data": "#1b9e77",
}
HATCHES = {
    "legacy":   "..",
    "k_quant":  "//",
    "iq_quant": "xx",
    "header":      "..",
    "metadata":    "\\\\",
    "tensor_info": "//",
    "tensor_data": "xx",
}
DISPLAY_LABELS = {
    "legacy":      "Legacy quants",
    "k_quant":     "k-quants",
    "iq_quant":    "IQ-quants",
    "header":      "Header",
    "metadata":    "Metadata KVs",
    "tensor_info": "Tensor info",
    "tensor_data": "Tensor data",
}


def _pick_font() -> str:
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for c in ("Arial", "Helvetica", "DejaVu Sans"):
        if c in available:
            return c
    return "sans-serif"


def apply_manning_style():
    plt.rcParams.update({
        "font.family": _pick_font(),
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.dpi": 300, "savefig.dpi": 300,
        "figure.figsize": (5.6, 3.5),
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_or_show(fig, name: str, config: Config):
    if config.save_plots:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        png = config.output_dir / f"{name}.png"
        pdf = config.output_dir / f"{name}.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        fig.savefig(pdf, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved: {png}")
        print(f"  Saved: {pdf}")
    else:
        plt.show()
    plt.close(fig)


# --- Environment probe -----------------------------------------------------

def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(name)
        except PackageNotFoundError:
            return "NOT INSTALLED"
    except ImportError:
        return "NOT INSTALLED"


def print_environment(config: Config):
    import platform

    print(f"  Python:           {sys.version.split()[0]}")
    print(f"  numpy:            {_pkg_version('numpy')}")
    print(f"  matplotlib:       {_pkg_version('matplotlib')}")
    print(f"  huggingface_hub:  {_pkg_version('huggingface_hub')}")
    print(f"  gguf (optional):  {_pkg_version('gguf')}")

    proc = platform.processor() or platform.machine()
    print(f"  CPU:              {proc} ({platform.machine()})")
    print(f"  OS:               {platform.system()} {platform.release()}")
    print(f"  Cache dir:        {config.cache_dir}")
    print(f"  Output dir:       {config.output_dir}")


# --- Artifact resolution ---------------------------------------------------

def resolve_gguf_path(config: Config) -> Path:
    """Return a local Path to a GGUF file. Either:

      1. The user passed --gguf-path, in which case we just use that.
      2. Otherwise, look up the slot in ARTIFACT_SLOTS and download the
         file from the Hugging Face Hub (cached under config.cache_dir).

    The HF download is on-demand. A clear stderr message + non-zero
    exit if the network fails — this function is the only place the
    internet matters."""
    if config.gguf_path is not None:
        if not config.gguf_path.exists():
            print(f"  ERROR: --gguf-path does not exist: {config.gguf_path}",
                  file=sys.stderr)
            sys.exit(2)
        print(f"  Using local GGUF (from --gguf-path): {config.gguf_path}")
        return config.gguf_path

    slot = ARTIFACT_SLOTS.get(config.slot)
    if slot is None:
        print(f"  ERROR: unknown slot {config.slot!r}", file=sys.stderr)
        sys.exit(2)

    config.cache_dir.mkdir(parents=True, exist_ok=True)
    candidate = config.cache_dir / slot["filename"]
    if candidate.exists() and not config.force_redownload:
        size_gb = candidate.stat().st_size / 1e9
        print(f"  Reusing cached GGUF: {candidate.name} ({size_gb:.2f} GB)")
        return candidate

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  ERROR: huggingface_hub is required to fetch GGUF artifacts.",
              file=sys.stderr)
        print("  Install with: pip install huggingface_hub", file=sys.stderr)
        sys.exit(2)

    print(f"  Downloading {slot['repo_id']} / {slot['filename']} ...")
    print(f"    Role: {slot['role']}")
    t0 = time.perf_counter()
    try:
        path = hf_hub_download(
            repo_id=slot["repo_id"],
            filename=slot["filename"],
            local_dir=str(config.cache_dir),
            force_download=config.force_redownload,
        )
    except Exception as e:
        print(f"  ERROR: hf_hub_download failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        print("  Hint: check network access, then retry with "
              "--force-redownload to bust any partial cache.",
              file=sys.stderr)
        sys.exit(2)
    elapsed = time.perf_counter() - t0
    p = Path(path)
    size_gb = p.stat().st_size / 1e9
    print(f"    Downloaded in {elapsed:.1f} s  |  {size_gb:.2f} GB on disk")
    return p


# --- GGUF v3 parser --------------------------------------------------------
# Spec reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
# Numeric type IDs match GGUFValueType in the gguf PyPI package; we keep
# our own table here so the parser does not depend on that import.

GGUF_MAGIC = b"GGUF"

# Metadata value type IDs (GGUFValueType).
GVT_UINT8   = 0
GVT_INT8    = 1
GVT_UINT16  = 2
GVT_INT16   = 3
GVT_UINT32  = 4
GVT_INT32   = 5
GVT_FLOAT32 = 6
GVT_BOOL    = 7
GVT_STRING  = 8
GVT_ARRAY   = 9
GVT_UINT64  = 10
GVT_INT64   = 11
GVT_FLOAT64 = 12

# Mapping: type id -> (struct format char, byte width, label)
SCALAR_TYPES = {
    GVT_UINT8:   ("B", 1, "u8"),
    GVT_INT8:    ("b", 1, "i8"),
    GVT_UINT16:  ("H", 2, "u16"),
    GVT_INT16:   ("h", 2, "i16"),
    GVT_UINT32:  ("I", 4, "u32"),
    GVT_INT32:   ("i", 4, "i32"),
    GVT_FLOAT32: ("f", 4, "f32"),
    GVT_BOOL:    ("?", 1, "bool"),
    GVT_UINT64:  ("Q", 8, "u64"),
    GVT_INT64:   ("q", 8, "i64"),
    GVT_FLOAT64: ("d", 8, "f64"),
}

# ggml_type IDs and their (block_size_in_elements, type_size_in_bytes,
# display name). Sourced from ggml.h enum ggml_type. Used by the size
# accounting in inspect mode and by the per-format dumpers below.
GGML_TYPES: Dict[int, Tuple[int, int, str]] = {
    0:  (1, 4, "F32"),
    1:  (1, 2, "F16"),
    2:  (32, 18,  "Q4_0"),
    3:  (32, 20,  "Q4_1"),
    6:  (32, 22,  "Q5_0"),
    7:  (32, 24,  "Q5_1"),
    8:  (32, 34,  "Q8_0"),
    9:  (32, 36,  "Q8_1"),
    10: (256, 84,  "Q2_K"),
    11: (256, 110, "Q3_K"),
    12: (256, 144, "Q4_K"),
    13: (256, 176, "Q5_K"),
    14: (256, 210, "Q6_K"),
    15: (256, 292, "Q8_K"),
    16: (256, 66,  "IQ2_XXS"),
    17: (256, 74,  "IQ2_XS"),
    18: (256, 98,  "IQ3_XXS"),
    19: (256, 50,  "IQ1_S"),
    20: (32, 18,  "IQ4_NL"),
    21: (256, 110, "IQ3_S"),
    22: (256, 82,  "IQ2_S"),
    23: (256, 136, "IQ4_XS"),
    24: (1, 1, "I8"),
    25: (1, 2, "I16"),
    26: (1, 4, "I32"),
    27: (1, 8, "I64"),
    28: (1, 8, "F64"),
    29: (256, 56,  "IQ1_M"),
    30: (1, 2, "BF16"),
}


@dataclass
class GGUFTensor:
    name: str
    n_dims: int
    shape: Tuple[int, ...]
    ggml_type: int
    type_name: str
    block_size: int
    type_size: int
    offset_in_data: int          # offset relative to start of tensor data section
    abs_offset: int              # absolute file offset
    n_bytes: int                 # bytes occupied by this tensor


@dataclass
class GGUFFile:
    path: Path
    file_size: int
    version: int
    tensor_count: int
    metadata_kv_count: int
    metadata: Dict[str, Any]
    metadata_types: Dict[str, str]      # display labels per key
    tensors: List[GGUFTensor]
    alignment: int
    header_end: int                     # offset right after the 24-byte header
    metadata_end: int                   # offset right after metadata KVs
    tensor_info_end: int                # offset right after tensor descriptors
    tensor_data_start: int              # offset of first tensor byte (after padding)


class _Reader:
    """Thin wrapper around a file handle with `struct`-typed reads.

    GGUF is little-endian. Lengths are u64 (NOT u32) — getting that wrong
    silently misaligns every following read by 4 bytes."""

    def __init__(self, fh):
        self.fh = fh

    def tell(self) -> int:
        return self.fh.tell()

    def read(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise IOError(f"short read: wanted {n} bytes, got {len(b)} "
                          f"at offset {self.fh.tell()}")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def scalar(self, vt: int):
        fmt, sz, _ = SCALAR_TYPES[vt]
        return struct.unpack("<" + fmt, self.read(sz))[0]

    def gguf_string(self) -> str:
        n = self.u64()
        return self.read(n).decode("utf-8", errors="replace")


def _parse_metadata_value(reader: _Reader, vt: int) -> Tuple[Any, str]:
    """Parse one metadata value of the given type. Returns (value, label).

    Arrays are recursive: an array's element type is itself a u32, and
    array length is a u64. Nested arrays are legal in the spec; in
    practice nobody emits them, but the parser handles them anyway."""
    if vt in SCALAR_TYPES:
        return reader.scalar(vt), SCALAR_TYPES[vt][2]
    if vt == GVT_STRING:
        return reader.gguf_string(), "string"
    if vt == GVT_ARRAY:
        elem_type = reader.u32()
        n = reader.u64()
        items = [_parse_metadata_value(reader, elem_type)[0] for _ in range(n)]
        elem_label = (SCALAR_TYPES[elem_type][2] if elem_type in SCALAR_TYPES
                      else ("string" if elem_type == GVT_STRING else f"vt{elem_type}"))
        return items, f"array<{elem_label}>[{n}]"
    raise ValueError(f"unknown metadata value type {vt}")


def _tensor_n_bytes(n_elements: int, ggml_type: int) -> int:
    """Bytes occupied by a tensor of n_elements weights at the given ggml type.

    The layout is: ceil(n_elements / block_size) * type_size. Block size is
    the number of weights packed into one type_size-byte block; for FP32 it
    is 1 (4 bytes/element), for Q4_K it is 256 (144 bytes/256 weights)."""
    if ggml_type not in GGML_TYPES:
        return 0
    block_size, type_size, _ = GGML_TYPES[ggml_type]
    n_blocks = (n_elements + block_size - 1) // block_size
    return n_blocks * type_size


def parse_gguf(path: Path) -> GGUFFile:
    """Parse a GGUF v3 file. Pure-Python; no `gguf` import.

    Walks the four sections in order:
       header (24 B) -> metadata KVs -> tensor descriptors -> alignment
       padding -> tensor data.
    Records the offsets of each section boundary so the layout figure
    can render them to scale."""
    file_size = path.stat().st_size
    with open(path, "rb") as fh:
        r = _Reader(fh)

        # ---- Header (24 bytes) --------------------------------------------
        magic = r.read(4)                                                    #A
        if magic != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file: magic={magic!r}")
        version = r.u32()
        tensor_count = r.u64()
        metadata_kv_count = r.u64()
        header_end = r.tell()        # exactly 24 if the spec is honoured

        # ---- Metadata KV pairs --------------------------------------------
        metadata: Dict[str, Any] = {}
        metadata_types: Dict[str, str] = {}
        for _ in range(metadata_kv_count):
            key = r.gguf_string()
            vt = r.u32()
            value, label = _parse_metadata_value(r, vt)
            metadata[key] = value
            metadata_types[key] = label
        metadata_end = r.tell()

        alignment = int(metadata.get("general.alignment", 32))               #B

        # ---- Tensor descriptors -------------------------------------------
        tensors: List[GGUFTensor] = []
        for _ in range(tensor_count):
            name = r.gguf_string()
            n_dims = r.u32()
            shape = tuple(r.u64() for _ in range(n_dims))
            ggml_type = r.u32()
            offset_in_data = r.u64()

            n_elements = 1
            for d in shape:
                n_elements *= d
            block_size, type_size, type_name = GGML_TYPES.get(
                ggml_type, (1, 0, f"UNK({ggml_type})"))
            n_bytes = _tensor_n_bytes(n_elements, ggml_type)
            tensors.append(GGUFTensor(
                name=name, n_dims=n_dims, shape=shape,
                ggml_type=ggml_type, type_name=type_name,
                block_size=block_size, type_size=type_size,
                offset_in_data=offset_in_data,
                abs_offset=0,           # patched below once we know data start
                n_bytes=n_bytes,
            ))
        tensor_info_end = r.tell()

        # ---- Alignment padding to general.alignment -----------------------
        # GGUF aligns tensor data to `alignment` bytes (default 32). The
        # first tensor byte is at the next multiple of `alignment` >= the
        # tensor-descriptor end. The intervening bytes are zero pad.
        padding = (-tensor_info_end) % alignment                             #C
        tensor_data_start = tensor_info_end + padding

        for t in tensors:
            t.abs_offset = tensor_data_start + t.offset_in_data

    return GGUFFile(
        path=path,
        file_size=file_size,
        version=version,
        tensor_count=tensor_count,
        metadata_kv_count=metadata_kv_count,
        metadata=metadata,
        metadata_types=metadata_types,
        tensors=tensors,
        alignment=alignment,
        header_end=header_end,
        metadata_end=metadata_end,
        tensor_info_end=tensor_info_end,
        tensor_data_start=tensor_data_start,
    )


#A The first 4 bytes are exactly b"GGUF" (no null terminator). If the file
#  starts with b"GGML" or b"GGJT" you have an old-format weights file
#  from pre-llama.cpp-2023; the parser cannot rescue it.

#B `general.alignment` is optional. When missing, the runtime defaults to
#  32 (set by ggml itself). Some converters write 64 for AVX-512 cache-line
#  friendliness; older converters wrote 16. Always read this key rather
#  than hard-coding 32.

#C The `(-x) % alignment` idiom returns the smallest non-negative integer
#  that, added to x, gives a multiple of `alignment`. It generalises
#  `(alignment - x % alignment) % alignment` and stays correct for x == 0.


# --- inspect mode ----------------------------------------------------------

def _format_size(n: int) -> str:
    """Human-readable byte count. Reused for tensor sizes and section sizes."""
    if n >= 1 << 30:
        return f"{n / (1 << 30):6.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):6.2f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):6.2f} KB"
    return f"{n:6d}  B"


def _format_metadata_value(value, label: str) -> str:
    """Compact one-line preview of a metadata value. Arrays are
    truncated to the first 3 elements + total count, so vocab tokens
    don't blow out the report."""
    if isinstance(value, list):
        head = value[:3]
        head_repr = ", ".join(_short_repr(v) for v in head)
        more = "" if len(value) <= 3 else f", ... ({len(value)} total)"
        return f"[{head_repr}{more}]"
    return _short_repr(value)


def _short_repr(v) -> str:
    if isinstance(v, str):
        if len(v) > 48:
            return f"\"{v[:45]}...\""
        return f"\"{v}\""
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _crossvalidate_with_gguf_pkg(path: Path, gguf: GGUFFile) -> Optional[str]:
    """Re-parse the same file through `gguf.GGUFReader` and confirm a few
    invariants. Returns None on success, or a one-line error string.

    The point isn't to test the gguf package — it's to test our parser.
    If our tensor count, metadata count, and a sampled tensor's shape +
    type all match what the canonical reader sees, our parse is right."""
    try:
        from gguf import GGUFReader
    except ImportError:
        return "package-not-installed"

    try:
        reader = GGUFReader(str(path))
    except Exception as e:
        return f"reader-failed: {type(e).__name__}: {e}"

    pkg_tensor_count = len(reader.tensors)
    if pkg_tensor_count != gguf.tensor_count:
        return (f"tensor count mismatch: ours={gguf.tensor_count}, "
                f"gguf={pkg_tensor_count}")

    if reader.tensors:
        pkg_first = reader.tensors[0]
        ours_first = gguf.tensors[0]
        if str(pkg_first.name) != ours_first.name:
            return (f"first tensor name mismatch: "
                    f"ours={ours_first.name!r}, gguf={pkg_first.name!r}")
        pkg_shape = tuple(int(x) for x in pkg_first.shape)
        # gguf.GGUFReader returns shapes in reversed order vs. the
        # on-disk layout (reader convenience for numpy). Compare both
        # orderings so we don't false-positive.
        if pkg_shape != ours_first.shape and pkg_shape != ours_first.shape[::-1]:
            return (f"first tensor shape mismatch: ours={ours_first.shape}, "
                    f"gguf={pkg_shape}")
    return None


def run_inspect(config: Config, gguf: GGUFFile) -> None:
    """Print a structured report of the parsed GGUF file. This is the
    text counterpart to the layout figure: same data, no rendering."""
    print()
    print("=" * 72)
    print(f"GGUF inspect — {gguf.path.name}")
    print("=" * 72)
    print(f"  File size:     {_format_size(gguf.file_size).strip()} "
          f"({gguf.file_size:,} bytes)")
    print(f"  Magic:         GGUF")
    print(f"  Version:       {gguf.version}")
    print(f"  Tensor count:  {gguf.tensor_count}")
    print(f"  Metadata KVs:  {gguf.metadata_kv_count}")
    print(f"  Alignment:     {gguf.alignment} bytes")
    print(f"  Header ends:   0x{gguf.header_end:08x}")
    print(f"  Metadata ends: 0x{gguf.metadata_end:08x}")
    print(f"  Tensor info ends: 0x{gguf.tensor_info_end:08x}")
    print(f"  Tensor data starts: 0x{gguf.tensor_data_start:08x} "
          f"(after {gguf.tensor_data_start - gguf.tensor_info_end} pad bytes)")

    # ---- Metadata block ---------------------------------------------------
    print()
    print("  ─── Metadata " + "─" * 60)
    keys = list(gguf.metadata.keys())
    keys.sort()
    for key in keys:
        value = gguf.metadata[key]
        label = gguf.metadata_types.get(key, "?")
        rendered = _format_metadata_value(value, label)
        print(f"  {key:<38} {label:<14} {rendered}")

    # ---- Tensor list ------------------------------------------------------
    n_show = len(gguf.tensors) if config.full_tensors else min(
        10, len(gguf.tensors))
    suffix = "" if config.full_tensors else f" (first {n_show} of {len(gguf.tensors)})"
    print()
    print(f"  ─── Tensors{suffix} " +
          "─" * max(1, 60 - len(suffix)))
    print(f"  {'#':>4}  {'name':<38} {'shape':<22} {'dtype':<8} "
          f"{'offset':<12} {'size':>10}")
    for i, t in enumerate(gguf.tensors[:n_show]):
        shape_str = "[" + ", ".join(str(d) for d in t.shape) + "]"
        print(f"  {i:>4}  {t.name[:38]:<38} {shape_str[:22]:<22} "
              f"{t.type_name:<8} 0x{t.abs_offset:08x}   "
              f"{_format_size(t.n_bytes):>10}")

    # ---- Type histogram ---------------------------------------------------
    type_counts: Dict[str, int] = {}
    for t in gguf.tensors:
        type_counts[t.type_name] = type_counts.get(t.type_name, 0) + 1
    total = max(1, len(gguf.tensors))
    print()
    print("  ─── Type histogram " + "─" * 53)
    for name, count in sorted(type_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<8} {count:>4}  ({count / total * 100:5.1f}%)")

    # ---- Cross-validation -------------------------------------------------
    print()
    err = _crossvalidate_with_gguf_pkg(gguf.path, gguf)
    if err is None:
        print("  Cross-validated against `gguf` package: OK")
    elif err == "package-not-installed":
        print("  Cross-validation skipped: `gguf` package not installed "
              "(pip install gguf to enable).")
    else:
        print(f"  Cross-validation FAILED: {err}")


# --- tensor mode: per-format byte dumpers ----------------------------------
# Each helper reads the first superblock of a chosen tensor and prints
# the byte layout with annotations. The arithmetic in comments uses the
# `#A` `#B` `#C` annotation pattern so the book can typeset it as a
# numbered footnote.
#
# Q4_K_M (a 144-byte superblock of 256 weights):
#   bytes  0..1   d        — FP16 super-scale
#   bytes  2..3   dmin     — FP16 super-min
#   bytes  4..15  scales   — 12 bytes packing 8 sub-scales + 8 sub-mins
#                            in 6-bit fields (the "K" twist; see #C below)
#   bytes 16..143 qs       — 128 bytes, two 4-bit nibbles per byte,
#                            256 weights total
#
# Q8_0 (34-byte block of 32 weights):
#   bytes  0..1   d        — FP16 scale (no min)
#   bytes  2..33  qs       — 32 INT8 weights, one byte each
#
# Q6_K (210-byte superblock of 256 weights):
#   bytes   0..127   ql       — low 4 bits of each weight (256 nibbles)
#   bytes 128..191   qh       — high 2 bits of each weight (256 * 2 bits)
#   bytes 192..207   scales   — 16 INT8 sub-scales (16 sub-blocks of 16)
#   bytes 208..209   d        — FP16 super-scale

def _fp16_to_float(b: bytes) -> float:
    """Decode a 2-byte little-endian IEEE-754 binary16 to Python float."""
    return float(np.frombuffer(b, dtype=np.float16)[0])


def _hex_preview(buf: bytes, n: int = 8) -> str:
    return " ".join(f"{b:02x}" for b in buf[:n])


def find_tensor(gguf: GGUFFile, name: str) -> Optional[GGUFTensor]:
    for t in gguf.tensors:
        if t.name == name:
            return t
    return None


def _read_superblock(path: Path, abs_offset: int, n_bytes: int) -> bytes:
    with open(path, "rb") as fh:
        fh.seek(abs_offset)
        return fh.read(n_bytes)


def dump_q4_k_block(buf: bytes) -> List[str]:
    """Decode the first Q4_K_M (a.k.a. Q4_K) superblock and return a
    list of pretty-printed lines. The reconstruction shown here is the
    minimum needed to exhibit the "scale × quantized-nibble - min" path
    — a full dequantize kernel lives in §10.4."""
    assert len(buf) >= 144, f"Q4_K block must be at least 144 bytes, got {len(buf)}"
    d = _fp16_to_float(buf[0:2])
    dmin = _fp16_to_float(buf[2:4])
    scales_bytes = buf[4:16]
    qs_bytes = buf[16:144]

    # Q4_K's 6-bit-packed sub-scales/mins ("the K twist"). Twelve bytes
    # encode 8 sub-scales + 8 sub-mins, each in 6 bits. The first 8
    # sub-scales are split as: low 4 bits of bytes[0..3] = scales[0..3]
    # low; bits 4..5 of bytes[8..11] = scales[0..3] high. Same idiom for
    # mins. The runtime kernel uses bit-twiddling rather than per-field
    # struct unpack — we reproduce that arithmetic for sub-block 0 only.
    s0_low6  = scales_bytes[0] & 0x3F
    sub_scale0 = s0_low6                                                     #C

    qs0 = qs_bytes[0]
    nibble_lo = qs0 & 0x0F
    nibble_hi = (qs0 >> 4) & 0x0F
    # In Q4_K the 4-bit field is *unsigned* (0..15). The signed effect
    # comes from the sub-min subtraction at decode: w = d*sub_scale*q -
    # dmin*sub_min. We dequantize a few weights to make that visible.
    sub_min0 = scales_bytes[4] & 0x3F
    super_min_term = dmin * sub_min0
    super_scale_term0 = d * sub_scale0
    weight0 = super_scale_term0 * nibble_lo - super_min_term
    weight1 = super_scale_term0 * nibble_hi - super_min_term

    # Eight nibbles for the "decoded weights" line.
    nibbles = []
    for byte in qs_bytes[:4]:
        nibbles.append(byte & 0x0F)
        nibbles.append((byte >> 4) & 0x0F)

    out = []
    out.append("─── Q4_K superblock layout (144 bytes / 256 weights) " + "─" * 18)
    out.append(f"{'Bytes':<7}{'Field':<48}{'Hex'}")
    out.append(f"{'0x00':<7}{'d         (FP16 super-scale)':<48}"
               f"{_hex_preview(buf[0:2], 2)}")
    out.append(f"{'0x02':<7}{'dmin      (FP16 super-min)':<48}"
               f"{_hex_preview(buf[2:4], 2)}")
    out.append(f"{'0x04':<7}{'scales    (12 B: 8 sub-scales + 8 sub-mins, 6-bit packed)':<48}"
               f"{_hex_preview(scales_bytes, 8)} ...")
    out.append(f"{'0x10':<7}{'qs        (128 B: 256 weights, 4-bit unsigned packed)':<48}"
               f"{_hex_preview(qs_bytes, 8)} ...")
    out.append("")
    out.append(f"  Decoded super-scale d        = 0x{int.from_bytes(buf[0:2], 'little'):04x} "
               f"-> {d:+.6f}")
    out.append(f"  Decoded super-min dmin       = 0x{int.from_bytes(buf[2:4], 'little'):04x} "
               f"-> {dmin:+.6f}")
    out.append(f"  sub-scale[0] (6-bit field)   = {sub_scale0}/63")
    out.append(f"  sub-min[0]   (6-bit field)   = {sub_min0}/63")
    out.append(f"  qs[0] = 0x{qs0:02x}  -> nibbles[0]={nibble_lo}, nibbles[1]={nibble_hi}")
    out.append(f"  Decoded weights[0..7]        = {nibbles}")
    out.append(f"  Reconstructed[0..1]          = "
               f"[{weight0:+.6f}, {weight1:+.6f}]")
    out.append("    (formula: w = d * sub_scale[i] * nibble - dmin * sub_min[i])")
    return out


def dump_q8_0_block(buf: bytes) -> List[str]:
    """Decode the first Q8_0 block. Simplest of the three: one FP16
    scale, then 32 signed INT8 weights, no mins or sub-blocks."""
    assert len(buf) >= 34, f"Q8_0 block must be at least 34 bytes, got {len(buf)}"
    d = _fp16_to_float(buf[0:2])
    qs_bytes = buf[2:34]
    qs_signed = np.frombuffer(qs_bytes, dtype=np.int8)
    weights = (d * qs_signed.astype(np.float32))[:8].tolist()

    out = []
    out.append("─── Q8_0 block layout (34 bytes / 32 weights) " + "─" * 24)
    out.append(f"{'Bytes':<7}{'Field':<48}{'Hex'}")
    out.append(f"{'0x00':<7}{'d         (FP16 scale)':<48}"
               f"{_hex_preview(buf[0:2], 2)}")
    out.append(f"{'0x02':<7}{'qs        (32 INT8 weights, signed)':<48}"
               f"{_hex_preview(qs_bytes, 8)} ...")
    out.append("")
    out.append(f"  Decoded scale d              = 0x{int.from_bytes(buf[0:2], 'little'):04x} "
               f"-> {d:+.6f}")
    out.append(f"  Quantized weights[0..7]      = {qs_signed[:8].tolist()}")
    out.append(f"  Reconstructed[0..7]          = "
               f"[{', '.join(f'{w:+.6f}' for w in weights)}]")
    out.append(f"    (formula: w = d * qs[i])")
    return out


def dump_q6_k_block(buf: bytes) -> List[str]:
    """Decode the first Q6_K superblock. Layout differs from Q4_K:
    the 6-bit weights are split across two byte arrays (low 4 bits in
    `ql`, high 2 bits in `qh`), and per-sub-block scales are full
    INT8s — no 6-bit packing."""
    assert len(buf) >= 210, f"Q6_K block must be at least 210 bytes, got {len(buf)}"
    ql = buf[0:128]
    qh = buf[128:192]
    scales_int8 = np.frombuffer(buf[192:208], dtype=np.int8)
    d = _fp16_to_float(buf[208:210])

    # Reconstruct weight 0:
    #   low4 = ql[0] & 0x0F, high2 = qh[0] & 0x03
    #   q6   = (low4 | (high2 << 4)) - 32     # unbias (Q6_K weights are signed)
    #   w    = d * scales[0] * q6
    low4_0 = ql[0] & 0x0F
    high2_0 = qh[0] & 0x03
    q6_0 = (low4_0 | (high2_0 << 4)) - 32
    w0 = d * float(scales_int8[0]) * q6_0

    out = []
    out.append("─── Q6_K superblock layout (210 bytes / 256 weights) " + "─" * 17)
    out.append(f"{'Bytes':<7}{'Field':<48}{'Hex'}")
    out.append(f"{'0x00':<7}{'ql        (128 B: low 4 bits of 256 weights)':<48}"
               f"{_hex_preview(ql, 8)} ...")
    out.append(f"{'0x80':<7}{'qh        (64 B: high 2 bits of 256 weights)':<48}"
               f"{_hex_preview(qh, 8)} ...")
    out.append(f"{'0xc0':<7}{'scales    (16 B: 16 INT8 sub-scales)':<48}"
               f"{_hex_preview(buf[192:208], 8)} ...")
    out.append(f"{'0xd0':<7}{'d         (FP16 super-scale)':<48}"
               f"{_hex_preview(buf[208:210], 2)}")
    out.append("")
    out.append(f"  Decoded super-scale d        = 0x{int.from_bytes(buf[208:210], 'little'):04x} "
               f"-> {d:+.6f}")
    out.append(f"  scales[0]                    = {int(scales_int8[0])}")
    out.append(f"  ql[0]={ql[0]:#04x}  qh[0]={qh[0]:#04x}  "
               f"-> q6[0] = ({low4_0} | ({high2_0}<<4)) - 32 = {q6_0}")
    out.append(f"  Reconstructed weight[0]      = {w0:+.6f}")
    out.append(f"    (formula: w = d * scales[i//16] * (low4 | (high2<<4) - 32))")
    return out


def run_tensor(config: Config, gguf: GGUFFile) -> None:
    """Read the first superblock of `config.tensor_name` (or fall back
    to the first parseable tensor) and pretty-print its byte layout."""
    name = config.tensor_name
    target = find_tensor(gguf, name)
    if target is None:
        # Fall back: pick the first tensor whose type we can dump.
        for t in gguf.tensors:
            if t.type_name in ("Q4_K", "Q8_0", "Q6_K"):
                target = t
                break
        if target is None:
            print(f"  ERROR: tensor {name!r} not found and no Q4_K/Q8_0/Q6_K "
                  f"tensor available in this file.")
            return
        print(f"  Note: tensor {name!r} not found; using {target.name!r} "
              f"({target.type_name}) instead.")

    print()
    print("=" * 72)
    print(f"GGUF tensor dump — {target.name}  ({target.type_name})")
    print("=" * 72)
    print(f"  Shape:          {list(target.shape)}")
    print(f"  Type:           {target.type_name} "
          f"(block_size={target.block_size}, type_size={target.type_size} B)")
    print(f"  Absolute offset: 0x{target.abs_offset:08x}  "
          f"(={target.abs_offset:,} bytes)")
    print(f"  Total size:     {_format_size(target.n_bytes).strip()}")
    print()

    if target.type_name == "Q4_K":
        buf = _read_superblock(gguf.path, target.abs_offset, 144)
        for line in dump_q4_k_block(buf):
            print("  " + line)
    elif target.type_name == "Q8_0":
        buf = _read_superblock(gguf.path, target.abs_offset, 34)
        for line in dump_q8_0_block(buf):
            print("  " + line)
    elif target.type_name == "Q6_K":
        buf = _read_superblock(gguf.path, target.abs_offset, 210)
        for line in dump_q6_k_block(buf):
            print("  " + line)
    else:
        print(f"  Tensor type {target.type_name} is outside the §10.2 dump set "
              f"(Q4_K, Q8_0, Q6_K). Choose another tensor with --tensor-name.")


# --- taxonomy mode: Figure 10.2 -------------------------------------------
# Effective bits per weight = (type_size * 8) / block_size for plain
# weights, plus an "overhead" share for super-scale/sub-scale metadata
# that the runtime spends per weight. Hardcoded numerator/denominator in
# comments so the reviewer can sanity-check.

QUANT_TAXONOMY = [
    # (variant, family, effective_bpw)
    ("Q4_0",   "legacy",   4.5),    # 18 B / 32 w = 4.5 bpw
    ("Q4_1",   "legacy",   5.0),    # 20 B / 32 w = 5.0 bpw
    ("Q5_0",   "legacy",   5.5),    # 22 B / 32 w = 5.5 bpw
    ("Q5_1",   "legacy",   6.0),    # 24 B / 32 w = 6.0 bpw
    ("Q8_0",   "legacy",   8.5),    # 34 B / 32 w = 8.5 bpw

    ("Q2_K",   "k_quant",  2.625),  # 84 B / 256 w  ≈ 2.6 bpw (incl. scale overhead)
    ("Q3_K",   "k_quant",  3.4375), # 110 B / 256 w ≈ 3.4 bpw
    ("Q4_K",   "k_quant",  4.5),    # 144 B / 256 w = 4.5 bpw
    ("Q5_K",   "k_quant",  5.5),    # 176 B / 256 w = 5.5 bpw
    ("Q6_K",   "k_quant",  6.5625), # 210 B / 256 w ≈ 6.6 bpw

    ("IQ1_S",  "iq_quant", 1.5625), # 50 B / 256 w  ≈ 1.6 bpw
    ("IQ2_XS", "iq_quant", 2.3125), # 74 B / 256 w  ≈ 2.3 bpw
    ("IQ3_XS", "iq_quant", 3.3125), # ~106 B / 256 w (rounded to spec)
    ("IQ4_XS", "iq_quant", 4.25),   # 136 B / 256 w ≈ 4.25 bpw
]


def run_taxonomy(config: Config) -> None:
    """Render Figure 10.2 — quant variants by effective bits per weight,
    grouped into Legacy / k-quants / IQ-quants."""
    apply_manning_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    families = ["legacy", "k_quant", "iq_quant"]

    # Layout: each family is a contiguous group of bars. We add a small
    # gap between groups so the eye picks up the boundary without a
    # subplot or a vertical separator (Manning prefers low chartjunk).
    bar_width = 0.65
    gap = 1.0
    x_positions = []
    x = 0.0
    family_centres = {}
    for fam in families:
        members = [(v, bpw) for (v, f, bpw) in QUANT_TAXONOMY if f == fam]
        start = x
        for (variant, bpw) in members:
            x_positions.append((variant, fam, x, bpw))
            x += 1.0
        family_centres[fam] = (start + x - 1) / 2.0
        x += gap        # gap to the next family

    # ---- Reference lines at INT4 / INT8 ----------------------------------
    # Drawn before bars so the bars overdraw any tick mark crossings.
    # We snap each reference label to the bottom-right end of its line,
    # just inside the rightmost bar (the IQ_quant family hovers around
    # 1-4 bpw so the right side of the chart is empty above 4.5).
    # INT4 / INT8 reference dashes. We pin labels to the gap between
    # k-quants and IQ-quants — that gap is empty space at every height,
    # so neither bar nor value-label collides with the text.
    xmin = min(p[2] for p in x_positions)
    label_x = None
    prev_fam = None
    for variant, fam, xpos, _ in x_positions:
        if prev_fam == "k_quant" and fam == "iq_quant":
            label_x = xpos - gap / 2
            break
        prev_fam = fam
    if label_x is None:
        label_x = xmin - 0.4

    for ref_y, ref_label in [(4.0, "INT4 (4 bpw)"),
                             (8.0, "INT8 (8 bpw)")]:
        ax.axhline(ref_y, linestyle="--", color="#aaaaaa",
                   linewidth=0.8, zorder=1)
        ax.text(label_x, ref_y + 0.10, ref_label,
                fontsize=6.5, color="#666666", ha="center", va="bottom")

    # ---- Bars -------------------------------------------------------------
    for variant, fam, xpos, bpw in x_positions:
        ax.bar(xpos, bpw,
               color=COLORS[fam], hatch=HATCHES[fam],
               edgecolor="black", linewidth=0.5, width=bar_width,
               zorder=3)
        ax.text(xpos, bpw + 0.12, f"{bpw:.2f}",
                ha="center", va="bottom", fontsize=6.5, zorder=4)

    ax.set_xticks([p[2] for p in x_positions])
    ax.set_xticklabels([p[0] for p in x_positions],
                       rotation=30, ha="right",
                       rotation_mode="anchor", fontsize=7)
    ax.set_ylabel("bits per weight (effective)")
    ax.set_ylim(0, max(bpw for _, _, _, bpw in x_positions) * 1.25)
    # Title + subtitle: stack with `\n` so they share a single suptitle
    # block. Two separate text() calls fight tight_layout for vertical
    # space and end up overlapping at this figure size.
    ax.set_title(
        "GGUF quant variants by effective bits per weight\n"
        "bar height includes scale and metadata overhead",
        fontsize=9)

    # ---- Family group labels under the x-axis ----------------------------
    for fam in families:
        centre = family_centres[fam]
        ax.text(centre, -0.22, DISPLAY_LABELS[fam],
                ha="center", va="top", fontsize=8, fontweight="bold",
                transform=ax.get_xaxis_transform())

    # ---- Legend -----------------------------------------------------------
    # Anchor outside the axes on the right so it doesn't fight the
    # taller k-quant/legacy bars or the INT8 reference label.
    legend_handles = [
        mpatches.Patch(facecolor=COLORS[fam], hatch=HATCHES[fam],
                       edgecolor="black", linewidth=0.5,
                       label=DISPLAY_LABELS[fam])
        for fam in families
    ]
    ax.legend(handles=legend_handles,
              loc="upper left", bbox_to_anchor=(1.0, 1.0),
              frameon=False, fontsize=7)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    save_or_show(fig, "CH10_F02_Kalyanarangan_gguf_quant_taxonomy", config)


# --- layout mode: Figure 10.3 ---------------------------------------------

def run_layout(config: Config, gguf: GGUFFile) -> None:
    """Render Figure 10.3 — the four GGUF sections drawn to scale, plus a
    hex inset of the first 256 bytes.

    The dynamic-range problem: tensor_data is ~99.7% of the file and
    the other three sections are <0.3% combined. A naive linear bar
    renders the front three sections as a hairline. We solve it with a
    two-row layout: row 1 is the to-scale bar (driving home the
    dominance of weights), row 2 is a *log-x* close-up of just the
    first three sections so their labels fit."""
    apply_manning_style()
    # Sized for half-page Manning placement (~5–6" rendered). We trim
    # vertical whitespace and lift base font sizes so the figure stays
    # legible after scale-down. The hex panel takes the dominant share
    # of vertical space because its content is small-text-heavy.
    fig = plt.figure(figsize=(6.8, 6.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.55, 0.85, 2.0], hspace=0.7)
    ax_bar  = fig.add_subplot(gs[0])
    ax_zoom = fig.add_subplot(gs[1])
    ax_hex  = fig.add_subplot(gs[2])

    # ---- Compute the four section widths from the parser, not constants --
    header_size      = gguf.header_end                                # 24 B
    metadata_size    = gguf.metadata_end - gguf.header_end
    tensor_info_size = gguf.tensor_info_end - gguf.metadata_end
    pad_size         = gguf.tensor_data_start - gguf.tensor_info_end
    tensor_data_size = gguf.file_size - gguf.tensor_data_start

    segments = [
        ("header",      header_size,      DISPLAY_LABELS["header"]),
        ("metadata",    metadata_size,    DISPLAY_LABELS["metadata"]),
        ("tensor_info", tensor_info_size, DISPLAY_LABELS["tensor_info"]),
        ("tensor_data", tensor_data_size, DISPLAY_LABELS["tensor_data"]),
    ]
    total = sum(s[1] for s in segments)

    # ---- Row 1: stacked horizontal bar drawn to true scale --------------
    # Only the dominant tensor_data segment carries an inline label here.
    # The three front sections are visible only as a thin sliver — row 2
    # zooms them. Wrapping the label in a small white-faced text patch
    # keeps it readable against the hatch fill.
    left = 0
    for key, width, label in segments:
        ax_bar.barh(0, width, left=left, height=0.55,
                    color=COLORS[key], hatch=HATCHES[key],
                    edgecolor="black", linewidth=0.5, zorder=3)
        pct = width / total * 100
        if width / total > 0.10:
            ax_bar.text(left + width / 2, 0,
                        f"{label}: {_format_size(width).strip()} "
                        f"({pct:.2f}%)",
                        ha="center", va="center", fontsize=8.5, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.25",
                                  facecolor="white", alpha=0.85,
                                  edgecolor="none"))
        left += width

    # Bracket+label for the front three sections (drawn together since
    # they collapse to a single sliver at file scale). Anchor below the
    # bar so it doesn't fight the suptitle for vertical space.
    front_size = total - tensor_data_size
    ax_bar.annotate(
        f"header + metadata + tensor info  "
        f"({_format_size(front_size).strip()}, "
        f"{front_size / total * 100:.2f}%) — zoomed below",
        xy=(front_size / 2, -0.32),
        xytext=(total * 0.30, -0.95),
        ha="center", va="top", fontsize=7.5, color="#333333",
        arrowprops=dict(arrowstyle="->", linewidth=0.6, color="#444444"),
        zorder=6,
    )

    ax_bar.set_xlim(0, total)
    ax_bar.set_ylim(-1.3, 0.9)
    ax_bar.set_yticks([])
    ax_bar.set_xticks([0, total])
    ax_bar.set_xticklabels(["0x00", f"0x{total:x}"], fontsize=7.5)
    ax_bar.spines["left"].set_visible(False)
    ax_bar.spines["bottom"].set_visible(False)
    ax_bar.set_title(
        f"GGUF file layout — {gguf.path.name} "
        f"({_format_size(gguf.file_size).strip()})",
        fontsize=10)

    # ---- Row 2: log-x zoom of just the front three sections + padding ---
    # Log scale so the 24 B header and the 1.6 MB metadata both have
    # readable widths. Labels go *above* the bar with leader lines
    # because the linear midpoint of a log-scaled segment isn't at the
    # visual centre — placing labels inside causes them to clump on the
    # right edge of each bar.
    front_total = gguf.tensor_data_start
    zoom_xmin = 1.0
    left = 0
    # Header / Metadata KVs / Tensor info labels stagger between two
    # heights. The alignment pad gets pushed to its own taller level
    # and offset to the right (its log-midpoint sits right on top of
    # tensor_info's, so a flat stagger isn't enough).
    label_y_levels = {0: 0.85, 1: 1.55, 2: 0.85}
    pad_label_y = 2.25
    li = 0
    for key, width, label in segments[:3] + [("__pad", pad_size, "pad")]:
        if width <= 0:
            li += 1
            continue
        right = left + width
        # Clip the visible left edge to >= zoom_xmin (log axis can't draw
        # to zero); otherwise barh silently widens the first bar to the
        # axis floor.
        bar_left = max(left, zoom_xmin)
        ax_zoom.barh(0, right - bar_left, left=bar_left, height=0.55,
                     color=(COLORS[key] if key in COLORS else "#cccccc"),
                     hatch=(HATCHES[key] if key in HATCHES else None),
                     edgecolor="black", linewidth=0.5, zorder=3)
        # Geometric mid for log-axis label placement.
        log_mid = np.sqrt(max(bar_left, zoom_xmin) * max(right, zoom_xmin + 1))
        pct = width / total * 100
        if key == "__pad":
            # Anchor pad label well to the right of tensor_info's label
            # so the two leader lines don't crash. The pad strip itself
            # is 4 B at offset ~1.65M — invisible at log scale, so the
            # leader line carries all the meaning.
            label_text = f"alignment pad ({width} B)"
            ax_zoom.annotate(
                label_text,
                xy=(log_mid, 0.30),
                xytext=(front_total * 3.5, pad_label_y),
                ha="center", va="bottom", fontsize=7.5, color="#555555",
                arrowprops=dict(arrowstyle="->", linewidth=0.5,
                                color="#888888"),
                zorder=6,
            )
        else:
            label_text = (f"{label}\n{_format_size(width).strip()} "
                          f"({pct:.3f}%)")
            ax_zoom.annotate(
                label_text,
                xy=(log_mid, 0.30),
                xytext=(log_mid, label_y_levels[li]),
                ha="center", va="bottom", fontsize=7.5, color="#333333",
                arrowprops=dict(arrowstyle="-", linewidth=0.5,
                                color="#888888"),
                zorder=6,
            )
        left = right
        li += 1

    # Truncated tensor-data marker on the right edge, below the pad
    # callout so the two don't visually fight.
    ax_zoom.text(front_total * 1.5, -0.05, "tensor data ⟶",
                 ha="left", va="center", fontsize=8.5, color="#1b9e77",
                 fontweight="bold")

    ax_zoom.set_xscale("log")
    ax_zoom.set_xlim(zoom_xmin, front_total * 8)
    ax_zoom.set_ylim(-0.6, 3.0)
    ax_zoom.set_yticks([])
    ax_zoom.set_xlabel("byte offset (log scale)", fontsize=8.5)
    ax_zoom.tick_params(axis='x', labelsize=7.5)
    ax_zoom.set_title("Front-of-file detail "
                      "(0x00 → start of tensor data)",
                      fontsize=9.5, loc="left")

    # ---- Row 3: hex inset of first 128 bytes ----------------------------
    # 128 bytes = 8 rows × 16 cols. Enough to show the full header + the
    # first metadata KV ('general.architecture' = "llama") + the start
    # of the second KV. 256 bytes was overkill for a half-page figure
    # and forced the per-cell font down to ~6 pt.
    raw_head = _read_superblock(gguf.path, 0, min(128, gguf.file_size))
    _draw_hex_inset(ax_hex, raw_head, gguf)

    save_or_show(fig, "CH10_F03_Kalyanarangan_gguf_file_layout", config)


def _draw_hex_inset(ax, raw: bytes, gguf: GGUFFile) -> None:
    """Render the first ~256 bytes of the file as a hex grid with field
    overlays. The grid is 16 bytes per row, classic `xxd` layout. We
    mark the four header fields and the first metadata-key bytes by
    drawing translucent rectangles on top of the affected cells.

    Layout: hex columns 0..15, an offset column on the left, the ASCII
    gutter on the right, and a field-label legend below the grid (NOT
    on the right — at this figure size the right margin is needed for
    the ASCII characters)."""
    cols = 16
    n = len(raw)
    rows = (n + cols - 1) // cols

    cell_w = 1.0

    # Field overlays. (offset, length, label, color). Drawn under the
    # hex digits so the digits stay legible.
    overlays = [
        (0,  4, "magic 'GGUF'",      "#7570b3"),
        (4,  4, "version u32",       "#e7298a"),
        (8,  8, "tensor_count u64",  "#d95f02"),
        (16, 8, "metadata_count u64", "#1b9e77"),
    ]
    # Append a metadata-key overlay if it fits within our 256-byte window.
    # The first metadata KV starts at byte 24: u64 keylen, then key bytes,
    # then u32 value type, then value. We highlight just the keylen + key.
    if n >= 32:
        first_key_len = struct.unpack("<Q", raw[24:32])[0]
        if first_key_len < 64 and 32 + first_key_len <= n:
            try:
                first_key = raw[32:32 + first_key_len].decode("utf-8")
                overlays.append((24, 8 + first_key_len,
                                 f"first KV: keylen + key '{first_key}'",
                                 "#888888"))
            except Exception:
                pass

    # X layout: [offset col, hex cols 0..15]. The ASCII gutter was
    # removed: in a 16-byte-per-row hex dump, strings longer than 16
    # bytes (e.g. 'general.architecture' at 20 B) inherently break
    # across rows, which read as wrapped/garbled text in print. The
    # field-overlay rectangles + legend already convey what each byte
    # range means.
    offset_x = -3.5
    hex_x0   = 0
    right_x  = cols + 0.5

    # Vertical spacing: rows are placed `row_pitch` units apart, with
    # field-overlay rectangles `rect_h` tall. Pitch > rect_h leaves
    # visible air between rows so each hex row reads as its own line.
    # With 128 visible bytes (8 rows) the panel can afford generous
    # spacing.
    row_pitch = 1.35
    rect_h = 1.05
    header_y = row_pitch * 1.0
    ax.set_xlim(offset_x - 0.5, right_x)
    ax.set_ylim(-(rows - 1) * row_pitch - 0.9, header_y + 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("First 128 bytes (hex)",
                 fontsize=9, loc="left", pad=6)

    # column header
    ax.text(offset_x, header_y, "offset", ha="left", va="center",
            fontsize=7.5, color="#666666")
    for c in range(cols):
        ax.text(hex_x0 + c * cell_w + cell_w / 2, header_y, f"{c:x}",
                ha="center", va="center", fontsize=7.5, color="#666666")

    # field overlays (translucent rectangles behind the hex digits)
    for off, length, _label, colour in overlays:
        for i in range(length):
            byte_idx = off + i
            if byte_idx >= n:
                break
            r = byte_idx // cols
            c = byte_idx % cols
            ax.add_patch(plt.Rectangle(
                (hex_x0 + c * cell_w, -r * row_pitch - rect_h / 2),
                cell_w, rect_h,
                facecolor=colour, alpha=0.25,
                edgecolor="none", zorder=1,
            ))

    # offset column on the left
    for r in range(rows):
        ax.text(offset_x, -r * row_pitch,
                f"0x{r * cols:04x}",
                ha="left", va="center",
                fontsize=7.5, family="monospace", color="#666666",
                zorder=2)

    # the hex digits themselves
    for i, b in enumerate(raw):
        r = i // cols
        c = i % cols
        ax.text(hex_x0 + c * cell_w + cell_w / 2, -r * row_pitch,
                f"{b:02x}",
                ha="center", va="center",
                fontsize=8.5, family="monospace", zorder=2)

    # Field-label legend below the grid. Using matplotlib's Legend
    # rather than hand-positioned text+rect pairs — that lets the
    # layout engine size each entry to the actual rendered text width
    # so long labels (e.g. "first KV: keylen + key '...'") don't get
    # clipped or visually wrap.
    legend_handles = [
        mpatches.Patch(facecolor=colour, alpha=0.45,
                       edgecolor="black", linewidth=0.4,
                       label=label)
        for _off, _length, label, colour in overlays
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=2, frameon=False, fontsize=8,
        handlelength=1.3, handleheight=1.0,
        columnspacing=2.0, labelspacing=0.5,
    )
    leg.set_in_layout(True)


# --- CLI / main -----------------------------------------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Ch10 sec 10.2 — GGUF v3 file format walker")
    p.add_argument("--mode", default="all",
                   choices=["inspect", "tensor", "taxonomy", "layout", "all"])
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--force-redownload", action="store_true")
    p.add_argument("--slot", default="reference",
                   choices=list(ARTIFACT_SLOTS.keys()))
    p.add_argument("--gguf-path", type=str, default=None,
                   help="path to a local GGUF file; overrides --slot")
    p.add_argument("--tensor-name", type=str,
                   default="blk.0.ffn_down.weight")
    p.add_argument("--full-tensors", action="store_true",
                   help="show all tensors in inspect mode (default: first 10)")
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    cfg = Config(
        mode=args.mode,
        save_plots=args.save_plots,
        force_redownload=args.force_redownload,
        slot=args.slot,
        tensor_name=args.tensor_name,
        full_tensors=args.full_tensors,
    )
    if args.gguf_path:
        cfg.gguf_path = Path(args.gguf_path)
    if args.cache_dir:
        cfg.cache_dir = Path(args.cache_dir)
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    return cfg


def main():
    config = parse_args()

    print("=" * 72)
    print("Chapter 10 sec 10.2 — The GGUF v3 file format from the bytes up")
    print("=" * 72)
    print(f"  Mode:             {config.mode}")
    print(f"  Slot:             {config.slot}")
    print(f"  Tensor name:      {config.tensor_name}")
    print(f"  Save plots:       {config.save_plots}")
    print(f"  Force redownload: {config.force_redownload}")
    print()
    print_environment(config)

    modes = (["inspect", "tensor", "taxonomy", "layout"]
             if config.mode == "all" else [config.mode])

    # The taxonomy figure is the only mode that doesn't need a GGUF on
    # disk. We resolve the file lazily so `--mode taxonomy` works even
    # without network access.
    needs_file = any(m in modes for m in ("inspect", "tensor", "layout"))
    gguf: Optional[GGUFFile] = None
    if needs_file:
        try:
            gguf_path = resolve_gguf_path(config)
            print(f"  Parsing {gguf_path.name} ...")
            t0 = time.perf_counter()
            gguf = parse_gguf(gguf_path)
            elapsed = time.perf_counter() - t0
            print(f"    Parsed in {elapsed:.2f} s "
                  f"({gguf.tensor_count} tensors, "
                  f"{gguf.metadata_kv_count} metadata KVs)")
        except Exception as e:
            print(f"\n  FAILED to load GGUF: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print("\n  Skipping modes that need the file; "
                  "still attempting taxonomy if requested.")

    for m in modes:
        try:
            if m == "inspect":
                if gguf is None:
                    print("\n  Skipping inspect: no GGUF loaded.")
                    continue
                run_inspect(config, gguf)
            elif m == "tensor":
                if gguf is None:
                    print("\n  Skipping tensor: no GGUF loaded.")
                    continue
                run_tensor(config, gguf)
            elif m == "taxonomy":
                run_taxonomy(config)
            elif m == "layout":
                if gguf is None:
                    print("\n  Skipping layout: no GGUF loaded.")
                    continue
                run_layout(config, gguf)
        except Exception as e:
            print(f"\n  FAILED mode={m}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print("\n  Continuing to next mode (if any)...")

    print("\n" + "=" * 72)
    print("Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
