"""Uncertainty-aware Hesban metadata comparison and optional shape fusion.

The shape matcher remains the primary retrieval system.  This module supplies
soft, explainable evidence: missing values are neutral, plausible measurement
or transcription differences are weak evidence, and several reliable,
independent incompatibilities can materially change a ranking.  It never
removes a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping


VERSION = "hesban-chip-metadata-v6-final-balanced"


@dataclass(frozen=True)
class Observation:
    value: Any
    reliability: float = 0.65
    uncertainty: float | None = None
    source: str = "unspecified"


@dataclass(frozen=True)
class FieldModel:
    key: str
    label: str
    group: str
    weight: float
    comparator: Callable[[Any, Any, Observation, Observation], tuple[float, str]]


@dataclass
class MetadataFusionConfig:
    """Calibratable research parameters, deliberately separate from code."""

    default_reliability: float = 0.65
    reviewed_reliability: float = 0.90
    automated_reliability: float = 0.55
    diameter_human_sigma_cm: float = 1.5
    diameter_reference_sigma_cm: float = 0.75
    max_field_evidence: float = 2.8
    max_group_evidence: float = 4.0
    # Metadata remains secondary.  These are absolute match-cost adjustments,
    # not a 50/50 blend: agreement gives a small bonus, disagreement a larger
    # but still capped penalty.
    # v4 raises metadata influence proportionally after the full-catalogue v12
    # run showed that physically meaningful differences changed too few close
    # boundary decisions. Shape remains primary, but strong agreement can now
    # reduce normalized cost by up to 3 points and strong conflict can add up
    # to 7.5 points.
    max_metadata_weight: float = 0.22
    max_metadata_bonus: float = 0.030
    max_metadata_penalty: float = 0.075
    # Severe diameter conflicts should not flatten at the generic metadata
    # ceiling. Beyond the broad 5 cm tolerance, add a reliability-scaled
    # quadratic tail. This distinguishes 20->40 from 20->50 while remaining
    # smooth and uncertainty-aware near the plausible region.
    diameter_tail_start_cm: float = 5.0
    diameter_tail_quadratic_per_cm2: float = 0.00011
    max_diameter_tail_penalty: float = 0.06
    metadata_neutral_compatibility: float = 0.65
    metadata_close_compatibility: float = 0.90
    metadata_availability_scale: float = 1.5
    field_reliability: dict[str, float] = field(default_factory=dict)


ALIASES = {
    "diameter": "rim_diameter_cm", "Rim Diameter (cm)": "rim_diameter_cm",
    "Type": "vessel_type", "Vessel Type": "vessel_type",
    "Fabric Color - Exterior": "fabric_exterior",
    "Fabric Color - Core": "fabric_core",
    "Fabric Color - Interior": "fabric_interior",
    "Non-Plastics - Typ": "nonplastics_type",
    "Non-Plastics - Siz": "nonplastics_size",
    "Non-Plastics - Shap": "nonplastics_shape",
    "Non-Plastics - Den": "nonplastics_density",
    "Voids - Ty/Sz": "voids_type_size", "Voids - Den": "voids_density",
    "Man": "manufacture", "Surface Treatment - Ext": "surface_exterior",
    "Surface Treatment - Exterior Color": "surface_exterior_color",
    "Surface Treatment - Int": "surface_interior",
    "Surface Treatment - Interior Color": "surface_interior_color",
    "Decor": "decor", "Fire": "fire", "firing": "fire",
}

EMPTY = {"", "-", "--", "?", "N/A", "NA", "NONE", "NULL", "UNKNOWN", "**"}


def _observation(raw: Any, key: str, config: MetadataFusionConfig) -> Observation | None:
    if isinstance(raw, Mapping):
        value = raw.get("value")
        source = str(raw.get("source") or "unspecified")
        default = (config.reviewed_reliability if raw.get("reviewed") else
                   config.automated_reliability if source in {"model", "ocr", "automatic"}
                   else config.default_reliability)
        reliability = raw.get("reliability", default)
        uncertainty = raw.get("uncertainty")
    else:
        value, source, reliability, uncertainty = raw, "unspecified", config.default_reliability, None
    if value is None or str(value).strip().upper() in EMPTY:
        return None
    try:
        reliability = float(reliability)
    except (TypeError, ValueError):
        reliability = config.default_reliability
    reliability *= config.field_reliability.get(key, 1.0)
    try:
        uncertainty = float(uncertainty) if uncertainty is not None else None
    except (TypeError, ValueError):
        uncertainty = None
    return Observation(value, min(1.0, max(0.0, reliability)), uncertainty, source)


def canonical_metadata(values: Mapping[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in (values or {}).items():
        result[ALIASES.get(str(key), str(key))] = value
    return result


def _tokens(value: Any) -> set[str]:
    return set(re.findall(r"[A-Z]+|\d+(?:\.\d+)?", str(value).upper()))


def _numeric(value: Any) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", "."))
    return float(match.group()) if match else None


def _soft_distance(distance: float, scale: float) -> float:
    # Student-t-like tail: gradual near zero, strong but never absolute far away.
    return 1.0 / (1.0 + (distance / max(scale, 1e-6)) ** 2)


def _diameter(a: Any, b: Any, oa: Observation, ob: Observation) -> tuple[float, str]:
    x, y = _numeric(a), _numeric(b)
    if x is None or y is None or x <= 0 or y <= 0:
        return 0.5, "diameter could not be parsed"
    # compare_metadata supplies the configured defaults when an observation has
    # no explicit uncertainty. Real-sherd forms can override them per query.
    sa = oa.uncertainty or 1.5
    sb = ob.uncertainty or 0.75
    # Diameter is reconstructed from a surviving arc rather than directly
    # measured on a complete vessel.  Keep a broad, smooth compatibility
    # plateau for chart/orientation/scale error, then let disagreement rise
    # more quickly once it is well beyond that plausible region.
    combined_uncertainty = math.sqrt(sa * sa + sb * sb)
    mean_diameter = (x + y) / 2.0
    scale = max(6.0, 0.30 * mean_diameter, 2.5 * combined_uncertainty)
    delta = abs(x - y)
    similarity = 1.0 / (1.0 + (delta / scale) ** 4)
    relative = delta / max(x, y)
    return similarity, (
        f"{x:g} vs {y:g} cm; difference {delta:g} cm ({relative:.1%}), "
        f"effective comparison tolerance {scale:.2f} cm"
    )


HUES = ("R", "YR", "Y", "GY", "G", "BG", "B", "PB", "P", "RP")


def _munsell(value: Any) -> tuple[float, float, float] | None:
    text = re.sub(r"\s+", " ", str(value).upper()).strip()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(YR|GY|BG|PB|RP|R|Y|G|B|P)\s+(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    numeral, hue, val, chroma = match.groups()
    hue_angle = (HUES.index(hue) + float(numeral) / 10.0) % len(HUES)
    return hue_angle, float(val), float(chroma)


@dataclass(frozen=True)
class ArchaeologicalColour:
    family: str
    lightness: int | None
    saturation: str
    tint: str | None = None


def _colour_name(value: Any) -> str:
    """Return the publication's verbal colour label, excluding its chip code."""
    text = str(value).lower().replace("grey", "gray")
    text = re.sub(
        r"\b\d+(?:\.\d+)?\s*(?:yr|gy|bg|pb|rp|r|y|g|b|p)\s*\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?\b",
        " ", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"[^a-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _archaeological_colour(value: Any) -> ArchaeologicalColour | None:
    """Reduce Hesban/Munsell names to a small hierarchy, not hundreds of chips.

    Families carry the archaeological colour meaning.  Lightness and
    saturation remain separate modifiers, so a small lip-relevant distinction
    is not erased merely to reduce the number of classes.
    """
    name = _colour_name(value)
    if not name:
        return None

    tint = next((x for x in ("pink", "red", "brown", "yellow") if x in name), None)
    if any(x in name for x in ("white", "gray", "black")):
        family = "neutral"
    elif "reddish brown" in name or "red brown" in name:
        family = "reddish_brown"
    elif "brown" in name:
        family = "brown"
    elif "pink" in name:
        family = "pink"
    elif "red" in name and "yellow" in name:
        family = "red_yellow"
    elif "red" in name:
        family = "red"
    elif "yellow" in name or "buff" in name:
        family = "yellow"
    else:
        return None

    if "very dark" in name or "black" in name:
        lightness = 0
    elif "dark" in name:
        lightness = 1
    elif "white" in name or "very pale" in name:
        lightness = 4
    elif "light" in name or "pale" in name:
        lightness = 3
    else:
        lightness = 2

    if any(x in name for x in ("gray", "white", "black")):
        saturation = "neutral"
    elif "grayish" in name:
        saturation = "grayish"
    elif "weak" in name or "pale" in name:
        saturation = "weak"
    else:
        saturation = "normal"
    return ArchaeologicalColour(family, lightness, saturation, tint)


_FAMILY_NEIGHBOURS = {
    frozenset(("pink", "red")): .82,
    frozenset(("pink", "reddish_brown")): .62,
    frozenset(("red", "reddish_brown")): .82,
    frozenset(("red", "red_yellow")): .82,
    frozenset(("reddish_brown", "brown")): .86,
    frozenset(("reddish_brown", "red_yellow")): .68,
    frozenset(("brown", "yellow")): .64,
    frozenset(("red_yellow", "yellow")): .82,
}


def _semantic_colour_similarity(x: ArchaeologicalColour,
                                y: ArchaeologicalColour) -> float:
    if x.family == y.family:
        family = 1.0
    elif "neutral" in (x.family, y.family):
        coloured = y if x.family == "neutral" else x
        neutral = x if x.family == "neutral" else y
        family = .48 if neutral.tint == coloured.family else .22
    else:
        family = _FAMILY_NEIGHBOURS.get(frozenset((x.family, y.family)), .20)
    gap = abs((x.lightness or 2) - (y.lightness or 2))
    tone = (1.0, .82, .58, .34, .18)[min(gap, 4)]
    saturation = (1.0 if x.saturation == y.saturation else
                  .78 if "normal" in (x.saturation, y.saturation) else .55)
    return .68 * family + .22 * tone + .10 * saturation


def _munsell_descriptors(chip: tuple[float, float, float]) -> tuple[str, str, str]:
    """Small, interpretable descriptors derived only from H V/C notation.

    These labels explain a continuous comparison; they are not hard classes.
    Adjacent regions remain close on the underlying Munsell axes.
    """
    hue, value, chroma = chip
    hue_centres = (
        (1.00, "R/10R"), (1.25, "2.5YR"), (1.50, "5YR"),
        (1.75, "7.5YR"), (2.10, "10YR/Y"),
    )
    hue_label = min(hue_centres, key=lambda item: abs(hue - item[0]))[1]
    lightness = ("dark" if value < 5 else "middle" if value < 6.5 else
                 "light" if value < 7.5 else "very light")
    saturation = "muted" if chroma <= 2 else "moderate" if chroma <= 4 else "strong"
    return hue_label, lightness, saturation


def _colour(a: Any, b: Any, oa: Observation, ob: Observation) -> tuple[float, str]:
    x, y = _munsell(a), _munsell(b)
    if x and y:
        hue = abs(x[0] - y[0]); hue = min(hue, len(HUES) - hue)
        dv, dc = abs(x[1] - y[1]), abs(x[2] - y[2])
        # Compare the three perceptual Munsell dimensions continuously.  The
        # descriptors are explanatory only, so crossing a named band never
        # creates an artificial scoring cliff.
        hue_similarity = _soft_distance(hue / .25, 2.0)
        value_similarity = _soft_distance(dv, 1.5)
        chroma_similarity = _soft_distance(dc, 2.5)
        # Hue carries the broad colour family, chroma separates vivid fabric
        # from gray/dull fabric, and value handles light/dark variation.  Value
        # is slightly less dominant because illumination and human reading make
        # it the easiest axis to shift in real recording conditions.
        similarity = .40 * hue_similarity + .25 * value_similarity + .35 * chroma_similarity
        left_desc, right_desc = _munsell_descriptors(x), _munsell_descriptors(y)
        return similarity, (
            f"Munsell H/V/C: {left_desc[0]}, {left_desc[1]}, {left_desc[2]} vs "
            f"{right_desc[0]}, {right_desc[1]}, {right_desc[2]}; continuous "
            f"differences hue {hue / .25:.1f} chart step(s), value {dv:g}, chroma {dc:g}"
        )

    # Verbal comparison is retained only for legacy/reference records that do
    # not contain a usable chip. Query-to-reference matching never needs words
    # when both sides supply H V/C notation.
    name_a, name_b = _colour_name(a), _colour_name(b)
    semantic_a, semantic_b = _archaeological_colour(a), _archaeological_colour(b)
    if semantic_a and semantic_b:
        semantic = _semantic_colour_similarity(semantic_a, semantic_b)
        x, y = _munsell(a), _munsell(b)
        # Exact chips resolve ties, but the archaeological verbal regions carry
        # most of the meaning and prevent tiny chip changes acting as classes.
        if x and y:
            hue = abs(x[0] - y[0]); hue = min(hue, len(HUES) - hue)
            chip = _soft_distance(
                math.sqrt((hue / 1.15) ** 2 + ((x[1] - y[1]) / 1.5) ** 2
                          + ((x[2] - y[2]) / 2.0) ** 2), 1.0,
            )
            similarity = .88 * semantic + .12 * chip
        else:
            similarity = semantic
        if name_a and name_a == name_b:
            similarity = max(similarity, .96)
        return similarity, (
            f"archaeological colour: {semantic_a.family}/{semantic_a.lightness}/"
            f"{semantic_a.saturation} vs {semantic_b.family}/{semantic_b.lightness}/"
            f"{semantic_b.saturation}"
        )
    ta, tb = _tokens(a), _tokens(b)
    overlap = len(ta & tb) / max(1, len(ta | tb))
    return 0.35 + 0.5 * overlap, "verbal/partially parsed colour comparison"


ORDINALS = {
    "size": {str(i): i for i in range(1, 8)},
    "letter_density": {"A": 1, "B": 2, "C": 3, "D": 4},
    "density": {"VL": 1, "L": 2, "M": 3, "MH": 4, "H": 5},
    "shape": {"A": 1, "SA": 2, "SR": 3, "R": 4},
}


def _ordinal(kind: str) -> Callable[[Any, Any, Observation, Observation], tuple[float, str]]:
    order = ORDINALS[kind]
    def compare(a: Any, b: Any, _oa: Observation, _ob: Observation) -> tuple[float, str]:
        left, right = _tokens(a) & order.keys(), _tokens(b) & order.keys()
        if not left or not right:
            return 0.5, f"{kind} code could not be parsed"
        gap = min(abs(order[x] - order[y]) for x in left for y in right)
        return _soft_distance(gap, 1.25), f"nearest {kind} category distance {gap}"
    return compare


def _categorical(a: Any, b: Any, _oa: Observation, _ob: Observation) -> tuple[float, str]:
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return 0.5, "category could not be parsed"
    jaccard = len(left & right) / len(left | right)
    # A mismatch is evidence, not an exclusion; multi-code partial overlap counts.
    return 0.12 + 0.88 * jaccard, f"shared codes {sorted(left & right) or 'none'}"


def _void_type_size(a: Any, b: Any, oa: Observation, ob: Observation) -> tuple[float, str]:
    types = {"FS", "FC", "PR", "PA", "JR", "JH", "JB", "JD"}
    la, lb = _tokens(a), _tokens(b)
    type_a, type_b = la & types, lb & types
    type_score = (len(type_a & type_b) / max(1, len(type_a | type_b))) if type_a and type_b else 0.5
    size_score, _ = _ordinal("size")(a, b, oa, ob)
    return 0.65 * type_score + 0.35 * size_score, f"void types {sorted(type_a)} vs {sorted(type_b)}; size considered softly"


MODELS = (
    FieldModel("vessel_type", "Vessel type", "classification", 1.20, _categorical),
    FieldModel("rim_diameter_cm", "Rim diameter", "measurement", 1.65, _diameter),
    FieldModel("fabric_exterior", "Exterior fabric colour", "colour", 0.65, _colour),
    FieldModel("fabric_core", "Core fabric colour", "colour", 0.80, _colour),
    FieldModel("fabric_interior", "Interior fabric colour", "colour", 0.65, _colour),
    FieldModel("nonplastics_type", "Non-plastics type", "inclusions", 0.85, _categorical),
    FieldModel("nonplastics_size", "Non-plastics size", "inclusions", 0.65, _ordinal("size")),
    FieldModel("nonplastics_shape", "Non-plastics shape", "inclusions", 0.60, _ordinal("shape")),
    FieldModel("nonplastics_density", "Non-plastics density", "inclusions", 0.70, _ordinal("density")),
    FieldModel("voids_type_size", "Voids type and size", "voids", 0.65, _void_type_size),
    FieldModel("voids_density", "Voids density", "voids", 0.55, _ordinal("density")),
    FieldModel("manufacture", "Manufacture", "technology", 0.85, _categorical),
    FieldModel("surface_exterior", "Exterior surface treatment", "surface", 0.75, _categorical),
    FieldModel("surface_exterior_color", "Exterior surface colour", "surface", 0.45, _colour),
    FieldModel("surface_interior", "Interior surface treatment", "surface", 0.75, _categorical),
    FieldModel("surface_interior_color", "Interior surface colour", "surface", 0.45, _colour),
    FieldModel("decor", "Decoration and vessel part", "decoration", 0.85, _categorical),
    FieldModel("fire", "Firing", "technology", 0.65, _categorical),
)


def compare_metadata(query: Mapping[str, Any] | None,
                     reference: Mapping[str, Any] | None,
                     config: MetadataFusionConfig | None = None) -> dict[str, Any]:
    """Return soft evidence and auditable field diagnostics.

    A zero evidence score means no usable metadata or balanced evidence.  It is
    not the same as incompatibility. Positive values support a candidate;
    negative values weaken it.
    """
    config = config or MetadataFusionConfig()
    query, reference = canonical_metadata(query), canonical_metadata(reference)
    details, groups, cost_groups = [], {}, {}
    possible = sum(model.weight for model in MODELS)
    observed_weight = 0.0
    for model in MODELS:
        left = _observation(query.get(model.key), model.key, config)
        right = _observation(reference.get(model.key), model.key, config)
        if left is None or right is None:
            details.append({"field": model.key, "label": model.label, "status": "missing", "evidence": 0.0})
            continue
        if model.key == "rim_diameter_cm":
            if left.uncertainty is None:
                left = Observation(left.value, left.reliability,
                                   config.diameter_human_sigma_cm, left.source)
            if right.uncertainty is None:
                right = Observation(right.value, right.reliability,
                                    config.diameter_reference_sigma_cm, right.source)
        compatibility, explanation = model.comparator(left.value, right.value, left, right)
        reliability = math.sqrt(left.reliability * right.reliability)
        strength = model.weight * reliability
        # Centre at neutral 0.5; nonlinear tails reward agreement and emphasize
        # clear incompatibility without ever becoming a binary gate.
        centred = 2.0 * min(1.0, max(0.0, compatibility)) - 1.0
        evidence = math.copysign(abs(centred) ** 1.35, centred) * strength * 2.0
        evidence = max(-config.max_field_evidence, min(config.max_field_evidence, evidence))
        groups.setdefault(model.group, []).append(evidence)
        cost_groups.setdefault(model.group, []).append((1.0 - compatibility, strength))
        observed_weight += strength
        details.append({
            "field": model.key, "label": model.label, "group": model.group,
            "status": "compared", "query": str(left.value), "reference": str(right.value),
            "compatibility": round(compatibility, 6), "reliability": round(reliability, 6),
            "mismatch_cost": round(1.0 - compatibility, 6), "field_weight": model.weight,
            "evidence": round(evidence, 6), "explanation": explanation,
        })
    group_evidence = {}
    for name, values in groups.items():
        raw = sum(values)
        # Correlated columns (three fabric colours, two surfaces) cannot vote
        # without limit merely because the publication records them separately.
        group_evidence[name] = config.max_group_evidence * math.tanh(raw / config.max_group_evidence)
    total = sum(group_evidence.values())
    # Multiple columns from the same archaeological category are correlated.
    # Average their costs, then let the strongest field set that group's weight
    # so three colour columns cannot outvote shape simply by being three columns.
    group_costs: dict[str, float] = {}
    group_strengths: dict[str, float] = {}
    for name, values in cost_groups.items():
        strength = sum(item[1] for item in values)
        group_costs[name] = (
            sum(cost * item_strength for cost, item_strength in values) / strength
            if strength else 0.5
        )
        group_strengths[name] = max(item[1] for item in values)
    metadata_strength = sum(group_strengths.values())
    metadata_cost = (
        sum(group_costs[name] * group_strengths[name] for name in group_costs)
        / metadata_strength
        if metadata_strength else None
    )
    coverage = observed_weight / possible if possible else 0.0
    return {
        "version": VERSION, "evidence": round(total, 6),
        "coverage": round(coverage, 6), "compared_fields": sum(d["status"] == "compared" for d in details),
        "evidence_strength": round(observed_weight, 6),
        "metadata_strength": round(metadata_strength, 6),
        "metadata_cost": round(metadata_cost, 6) if metadata_cost is not None else None,
        "group_costs": {key: round(value, 6) for key, value in group_costs.items()},
        "group_evidence": {k: round(v, 6) for k, v in group_evidence.items()},
        "fields": details,
        # This comparison report does not yet know the shape cost or the
        # clipped fused score. Do not label the evidence as supporting or
        # conflicting here: that older explanation path could disagree with
        # the numeric adjustment ultimately used by fusion.
        "summary": (
            "No comparable metadata; shape ranking is unchanged."
            if not groups else
            f"Compared {sum(d['status'] == 'compared' for d in details)} metadata field(s)."
        ),
    }


def _fusion_summary(shape_score: float, fused_score: float, compared_fields: int) -> str:
    """Explain only the effective score change visible in the final result."""
    if compared_fields <= 0:
        return "No comparable metadata; final cost is unchanged."
    delta = fused_score - shape_score
    if abs(delta) < 5e-9:
        return (
            f"Metadata compared {compared_fields} field(s) and did not change "
            f"the final cost ({shape_score:.6f})."
        )
    direction = "raised" if delta > 0 else "lowered"
    effect = "weakened" if delta > 0 else "strengthened"
    return (
        f"Metadata {direction} the cost by {abs(delta):.6f}, from "
        f"{shape_score:.6f} to {fused_score:.6f}; the match was {effect}."
    )


def fuse_shape_results(results: Iterable[Mapping[str, Any]], query_metadata: Mapping[str, Any],
                       reference_metadata: Mapping[str, Mapping[str, Any]],
                       config: MetadataFusionConfig | None = None) -> list[dict[str, Any]]:
    """Softly rerank shape results without dropping any candidate.

    The metadata influence is derived from observed reliability and coverage,
    capped in robust shape-score units, and therefore is neither fixed 50/50
    nor a hard filter.
    """
    config = config or MetadataFusionConfig()
    rows = [dict(row) for row in results]
    costs = [
        float(
            row.get(
                "shape_score",
                row.get("overall_score", row.get("score", row.get("match_cost", 0.0))),
            )
        )
        for row in rows
    ]
    for row, cost in zip(rows, costs):
        identities = [
            str(row.get("reference_id") or ""),
            str(row.get("source_filename") or ""),
            Path(str(row.get("source_filename") or "")).stem,
            str(row.get("mask_file") or ""),
        ]
        reference_values = next(
            (reference_metadata[key] for key in identities if key in reference_metadata),
            {},
        )
        report = compare_metadata(query_metadata, reference_values, config)
        # Metadata changes the shape cost continuously. Close agreement earns a
        # small bonus; mild disagreement is nearly neutral; strong disagreement
        # earns a larger, capped penalty. There are no exclusion thresholds.
        availability = 1.0 - math.exp(
            -report["metadata_strength"] / config.metadata_availability_scale
        )
        metadata_weight = config.max_metadata_weight * availability
        shape_cost = min(1.0, max(0.0, cost))
        metadata_cost = report["metadata_cost"]
        diameter_tail_penalty = 0.0
        diameter_field = next(
            (
                field for field in report["fields"]
                if field.get("field") == "rim_diameter_cm"
                and field.get("status") == "compared"
            ),
            None,
        )
        if diameter_field:
            query_diameter = _numeric(diameter_field.get("query"))
            reference_diameter = _numeric(diameter_field.get("reference"))
            if query_diameter is not None and reference_diameter is not None:
                severe_gap = max(
                    0.0,
                    abs(query_diameter - reference_diameter)
                    - config.diameter_tail_start_cm,
                )
                diameter_tail_penalty = min(
                    config.max_diameter_tail_penalty,
                    config.diameter_tail_quadratic_per_cm2
                    * severe_gap ** 2
                    * math.sqrt(max(0.0, float(diameter_field.get("reliability", 0.0)))),
                )
        if metadata_cost is None or metadata_weight <= 0:
            fused_score = shape_cost
            metadata_fraction = 0.0
            adjustment = 0.0
        else:
            compatibility = 1.0 - metadata_cost
            neutral = config.metadata_neutral_compatibility
            close = config.metadata_close_compatibility
            if compatibility >= close:
                # Even perfect metadata cannot manufacture a shape match.
                raw_adjustment = -config.max_metadata_bonus * (
                    .35 + .65 * (compatibility - close) / max(1e-6, 1.0 - close)
                )
            elif compatibility >= neutral:
                raw_adjustment = -config.max_metadata_bonus * .35 * (
                    (compatibility - neutral) / max(1e-6, close - neutral)
                )
            else:
                # The farther below the broad compatibility region, the more
                # metadata demotes the result, with a smooth quadratic rise.
                distance = (neutral - compatibility) / max(neutral, 1e-6)
                raw_adjustment = config.max_metadata_penalty * distance ** 1.35
            adjustment = raw_adjustment * availability + diameter_tail_penalty
            fused_score = min(1.0, max(0.0, shape_cost + adjustment))
            metadata_fraction = metadata_weight
        # Explain the effective, clipped score change—not the independent
        # evidence accumulator or the pre-clipping raw adjustment.
        report["summary"] = _fusion_summary(
            shape_cost, fused_score, int(report["compared_fields"])
        )
        row["shape_score"] = cost
        row["metadata"] = report
        row["metadata_score"] = metadata_cost
        row["metadata_weight"] = round(metadata_fraction, 6)
        row["metadata_adjustment"] = round(adjustment, 8)
        row["diameter_tail_penalty"] = round(diameter_tail_penalty, 8)
        row["fused_score"] = round(fused_score, 8)
    rows.sort(key=lambda row: (row["fused_score"], row["shape_score"]))
    for rank, row in enumerate(rows, 1):
        row["fused_rank"] = rank
    return rows


def load_reference_metadata(project_path: Path) -> dict[str, dict[str, str]]:
    """Load reviewed Hesban metadata keyed by both mask filename and stem."""
    path = Path(project_path) / "cards" / "mask_info.csv"
    if not path.exists():
        return {}
    indexed: dict[str, dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            identity = str(row.get("mask_file") or row.get("file") or "").strip()
            if not identity:
                continue
            values = canonical_metadata(row)
            indexed[identity] = values
            indexed[Path(identity).stem] = values
    return indexed
