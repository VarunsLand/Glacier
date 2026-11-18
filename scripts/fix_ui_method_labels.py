"""
Fix UI method-aware labels in visual_monitoring.py:
1. Confusion matrix expander header → show full method name
2. Predicted mask caption in expander side-by-side → use _mask_cap
3. Scientific Analysis text block → method-aware dict lookup (uses _analysis variable already set)
"""

import sys

TARGET = r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\src\ui\visual_monitoring.py"

with open(TARGET, encoding="utf-8") as f:
    src = f.read()

original_len = len(src)
changes = []

# ── Fix 1: Confusion matrix expander header ───────────────────────────────────
OLD1 = (
    "                        st.markdown(f\"Comparing `{selected_method_label.split(':')[0]}` against ground-truth reference annotations for sample `{ls_sample['sample_id']}` at threshold `|\u0394I| \u2265 {ch_threshold:.0f}`:\")"
)
NEW1 = (
    "                        st.markdown(f\"Comparing **{selected_method_label}** against ground-truth reference annotations for sample `{ls_sample['sample_id']}` at threshold T \u2265 {ch_threshold:.0f}:\")"
)
if OLD1 in src:
    src = src.replace(OLD1, NEW1, 1)
    changes.append("Fix 1: Confusion matrix expander header")
else:
    print("WARNING: Fix 1 target not found")

# ── Fix 2: Predicted mask caption inside confusion expander side-by-side ──────
OLD2 = (
    "                            st.image((ch_results[\"detected_mask\"] * 255).astype(np.uint8), caption=f\"Predicted deterministic change mask (|\u0394I| \u2265 {ch_threshold:.0f})\", use_container_width=True)\n"
    "                        with c_side2:"
)
NEW2 = (
    "                            st.image((ch_results[\"detected_mask\"] * 255).astype(np.uint8), caption=f\"{_mask_cap} (T \u2265 {ch_threshold:.0f})\", use_container_width=True)\n"
    "                        with c_side2:"
)
if OLD2 in src:
    src = src.replace(OLD2, NEW2, 1)
    changes.append("Fix 2: Confusion expander prediction image caption")
else:
    print("WARNING: Fix 2 target not found — trying alternate whitespace")
    # Try with \r\n
    OLD2b = OLD2.replace("\n", "\r\n")
    NEW2b = NEW2.replace("\n", "\r\n")
    if OLD2b in src:
        src = src.replace(OLD2b, NEW2b, 1)
        changes.append("Fix 2 (CRLF): Confusion expander prediction image caption")
    else:
        print("WARNING: Fix 2 also not found with CRLF")

# ── Fix 3: Scientific Analysis hardcoded text → _analysis variable ─────────────
OLD3 = (
    "                         \u003cb\u003eScientific Analysis of Radiometric \u0026 Semantic Behavior:\u003c/b\u003e\u003cbr/\u003e\n"
    "                                 In polar multi-year Landsat scenes, raw digital number subtraction suffers from severe illumination and solar elevation offsets. \n"
    "                                 Method A flags widespread static rock and snowpack as change. When using Method E (Illumination Offset-Normalized Difference), \n"
    "                                 global baseline shifts are compensated, significantly curbing false positives."
)
NEW3 = (
    "                         <b>Scientific Analysis of Radiometric &amp; Semantic Behavior ({selected_method_label.split(':')[0]}):</b><br/>\n"
    "                                 {_analysis}"
)
# The HTML entities may be unescaped in the actual file — check both
OLD3_raw = (
    "                                 <b>Scientific Analysis of Radiometric & Semantic Behavior:</b><br/>\n"
    "                                 In polar multi-year Landsat scenes, raw digital number subtraction suffers from severe illumination and solar elevation offsets. \n"
    "                                 Method A flags widespread static rock and snowpack as change. When using Method E (Illumination Offset-Normalized Difference), \n"
    "                                 global baseline shifts are compensated, significantly curbing false positives."
)
NEW3_raw = (
    "                                 <b>Scientific Analysis — {selected_method_label.split(':')[0]}:</b><br/>\n"
    "                                 {_analysis}"
)
if OLD3_raw in src:
    src = src.replace(OLD3_raw, NEW3_raw, 1)
    changes.append("Fix 3: Scientific Analysis text block")
else:
    # Try CRLF
    OLD3c = OLD3_raw.replace("\n", "\r\n")
    NEW3c = NEW3_raw.replace("\n", "\r\n")
    if OLD3c in src:
        src = src.replace(OLD3c, NEW3c, 1)
        changes.append("Fix 3 (CRLF): Scientific Analysis text block")
    else:
        print("WARNING: Fix 3 target not found, printing nearby context...")
        idx = src.find("Scientific Analysis of Radiometric")
        if idx >= 0:
            print(repr(src[idx-20:idx+300]))
        else:
            print("'Scientific Analysis of Radiometric' not found in file")

if changes:
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Applied {len(changes)} fix(es):")
    for c in changes:
        print(f"  ✓ {c}")
    print(f"File size: {len(src):,} bytes (was {original_len:,})")
else:
    print("No changes applied.")

sys.exit(0 if changes else 1)
