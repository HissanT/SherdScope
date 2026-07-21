"""Publication table profiles shared by extraction, APIs, and the browser UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ColumnSpec:
    key: str
    aliases: tuple[str, ...]
    header_tier: str
    group: str
    ui_label: str
    csv_label: str

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["aliases"] = list(self.aliases)
        return value


HESBAN_GROUPS = (
    {"key": "identity", "label": "Identity", "header_label": "", "anchor": False},
    {"key": "fabric", "label": "Fabric Color", "header_label": "Fabric Color", "anchor": False},
    {"key": "nonplastics", "label": "Non-Plastics", "header_label": "Non-Plastics", "anchor": False},
    {"key": "voids", "label": "Voids", "header_label": "Voids", "anchor": False},
    {"key": "manufacture", "label": "Manufacture", "header_label": "", "anchor": False},
    {"key": "surface", "label": "Surface Treatment", "header_label": "Surface Treatment", "anchor": False},
    {"key": "finish", "label": "Finish", "header_label": "", "anchor": False},
)


# Physical left-to-right order. Group headings are deliberately absent: only
# these printed headings and subheadings create boundaries.
HESBAN_COLUMN_SPECS = (
    ColumnSpec("table_no", ("no", "number"), "primary", "identity", "No.", "No."),
    ColumnSpec("table_type", ("type",), "primary", "identity", "Type", "Type"),
    ColumnSpec("table_square", ("sq", "square", "area"), "primary", "identity", "Sq/Area", "Sq/Area"),
    ColumnSpec("table_locus", ("loc", "locus"), "primary", "identity", "Loc", "Locus (Loc)"),
    ColumnSpec("table_pail", ("pail",), "primary", "identity", "Pail", "Pail"),
    ColumnSpec("table_registration", ("reg", "registration"), "primary", "identity", "Reg", "Registration (Reg)"),
    ColumnSpec("fabric_exterior", ("exterior",), "secondary", "fabric", "Exterior", "Fabric Color - Exterior"),
    ColumnSpec("fabric_core", ("core",), "secondary", "fabric", "Core", "Fabric Color - Core"),
    ColumnSpec("fabric_interior", ("interior",), "secondary", "fabric", "Interior", "Fabric Color - Interior"),
    ColumnSpec("nonplastics_type", ("typ", "type"), "secondary", "nonplastics", "Typ", "Non-Plastics - Type"),
    ColumnSpec("nonplastics_size", ("siz", "size"), "secondary", "nonplastics", "Siz", "Non-Plastics - Size"),
    ColumnSpec("nonplastics_shape", ("shap", "shape"), "secondary", "nonplastics", "Shap", "Non-Plastics - Shape"),
    ColumnSpec("nonplastics_density", ("den", "density"), "secondary", "nonplastics", "Den", "Non-Plastics - Density"),
    ColumnSpec("voids_type_size", ("tysz", "typesize"), "secondary", "voids", "Ty/Sz", "Voids - Type/Size"),
    ColumnSpec("voids_density", ("den", "density"), "secondary", "voids", "Den", "Voids - Density"),
    ColumnSpec("manufacture", ("man", "manufacture"), "primary", "manufacture", "Man", "Manufacture"),
    ColumnSpec("surface_exterior", ("ext", "exterior"), "secondary", "surface", "Ext", "Surface Treatment - Exterior"),
    ColumnSpec("surface_exterior_color", ("color", "colour"), "secondary", "surface", "Color", "Surface Treatment - Exterior Color"),
    ColumnSpec("surface_interior", ("int", "interior"), "secondary", "surface", "Int", "Surface Treatment - Interior"),
    ColumnSpec("surface_interior_color", ("color", "colour"), "secondary", "surface", "Color", "Surface Treatment - Interior Color"),
    ColumnSpec("decor", ("decor", "decoration"), "primary", "finish", "Decor", "Decoration"),
    ColumnSpec("fire", ("fire",), "primary", "finish", "Fire", "Firing"),
)

HESBAN_TABLE_COLUMNS = [spec.key for spec in HESBAN_COLUMN_SPECS]


def hesban_profile_payload() -> dict[str, object]:
    return {
        "slug": "hesban11",
        "columns": [spec.as_dict() for spec in HESBAN_COLUMN_SPECS],
        "groups": [dict(group) for group in HESBAN_GROUPS],
    }
