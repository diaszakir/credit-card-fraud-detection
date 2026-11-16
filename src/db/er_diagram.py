"""ER diagram generated from the *actual* database (SQLAlchemy introspection).

Outputs
    docs/er_diagram.png   picture for the report / slides (Matplotlib, no Graphviz needed)
    docs/er_diagram.md    Mermaid ``erDiagram`` (rendered natively by GitHub)

Because the diagram is read from the database, it cannot drift away from ``schema.sql``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from sqlalchemy import Engine, inspect  # noqa: E402

from src.config import DOCS_DIR, SQLITE_PATH  # noqa: E402
from src.db.engine import get_engine  # noqa: E402

logger = logging.getLogger(__name__)

_V_COLUMN = re.compile(r"^v\d+$")


@dataclass
class Column:
    name: str
    type: str
    is_pk: bool = False
    is_fk: bool = False


@dataclass
class Table:
    name: str
    columns: list[Column] = field(default_factory=list)


@dataclass
class Relation:
    child: str
    parent: str
    label: str  # constrained columns
    one_to_one: bool = False
    logical: bool = False  # no FOREIGN KEY constraint: a range join described in the docs


# Relations that exist logically but have no FK constraint (joined by a range condition).
LOGICAL_RELATIONS = [
    Relation(child="transactions", parent="amount_bands", label="amount in [lower, upper)", logical=True)
]


def read_model(engine: Engine) -> tuple[list[Table], list[Relation]]:
    """Read tables, columns, keys and foreign keys from the database."""
    insp = inspect(engine)
    tables: list[Table] = []
    relations: list[Relation] = []
    for name in sorted(insp.get_table_names()):
        pk = set(insp.get_pk_constraint(name).get("constrained_columns", []))
        fks = insp.get_foreign_keys(name)
        fk_columns = {c for fk in fks for c in fk["constrained_columns"]}
        tables.append(
            Table(
                name,
                [Column(c["name"], str(c["type"]), c["name"] in pk, c["name"] in fk_columns) for c in insp.get_columns(name)],
            )
        )
        for fk in fks:
            relations.append(
                Relation(
                    child=name,
                    parent=fk["referred_table"],
                    label=", ".join(fk["constrained_columns"]),
                    one_to_one=set(fk["constrained_columns"]) == pk,
                )
            )
    known = {t.name for t in tables}
    relations += [r for r in LOGICAL_RELATIONS if r.child in known and r.parent in known]
    return tables, relations


def _display_columns(table: Table) -> list[tuple[str, str, str]]:
    """(tag, name, type) rows; consecutive v1..v28 columns are collapsed into one row."""
    rows: list[tuple[str, str, str]] = []
    v_cols = [c for c in table.columns if _V_COLUMN.match(c.name)]
    collapsed = False
    for col in table.columns:
        if _V_COLUMN.match(col.name) and len(v_cols) > 3:
            if not collapsed:
                rows.append(("", f"{v_cols[0].name} … {v_cols[-1].name}  ({len(v_cols)} cols)", v_cols[0].type))
                collapsed = True
            continue
        tag = "PK, FK" if col.is_pk and col.is_fk else "PK" if col.is_pk else "FK" if col.is_fk else ""
        rows.append((tag, col.name, col.type))
    return rows


# --------------------------------------------------------------------------- Mermaid
def to_mermaid(tables: list[Table], relations: list[Relation]) -> str:
    """Mermaid ``erDiagram`` source."""
    lines = ["erDiagram"]
    for rel in relations:
        if rel.logical:
            arrow = "||..o{"  # non-identifying, dashed: logical range join
        elif rel.one_to_one:
            arrow = "||--||"
        else:
            arrow = "||--o{"
        parent, child = rel.parent, rel.child
        lines.append(f'    {parent} {arrow} {child} : "{rel.label}"')
    for table in tables:
        lines.append(f"    {table.name} {{")
        for tag, name, col_type in _display_columns(table):
            clean = re.sub(r"[^A-Za-z0-9_]", "_", name.split("  ")[0].replace(" … ", "_to_")).strip("_")
            key = "PK" if "PK" in tag else "FK" if tag == "FK" else ""
            lines.append(f"        {col_type} {clean} {key}".rstrip())
        lines.append("    }")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- Matplotlib
SURFACE, INK, INK2, HAIRLINE = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"
HEADER_FACT, HEADER_DIM, HEADER_FEAT = "#cde2fb", "#ece9e2", "#e1ecf9"
ACCENT = "#2a78d6"

BOX_W = {"transactions": 6.4, "transaction_features": 6.4, "time_slots": 4.6}
DEFAULT_W = 4.2
ROW_H, HEAD_H = 0.46, 0.6

# top-left corner (x, y) of every box in diagram units
LAYOUT = {
    "splits": (0.4, 10.8),
    "classes": (5.8, 10.8),
    "time_slots": (11.2, 10.8),
    "transactions": (4.7, 6.4),
    "amount_bands": (12.8, 5.94),
    "transaction_features": (4.7, 1.8),
}
HEADER_COLORS = {"transactions": HEADER_FACT, "transaction_features": HEADER_FEAT}


def _box_geometry(table: Table) -> tuple[float, float, float, float]:
    x, y_top = LAYOUT[table.name]
    width = BOX_W.get(table.name, DEFAULT_W)
    height = HEAD_H + ROW_H * len(_display_columns(table))
    return x, y_top, width, height


def _draw_table(ax, table: Table) -> None:
    x, y_top, width, height = _box_geometry(table)
    ax.add_patch(FancyBboxPatch((x, y_top - height), width, height, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=SURFACE, edgecolor=INK2, linewidth=1.2, zorder=2))
    ax.add_patch(Rectangle((x + 0.01, y_top - HEAD_H), width - 0.02, HEAD_H - 0.01,
                           facecolor=HEADER_COLORS.get(table.name, HEADER_DIM), edgecolor="none", zorder=3))
    ax.text(x + 0.18, y_top - HEAD_H / 2, table.name, va="center", ha="left", fontsize=12, fontweight="bold", color=INK, zorder=4)
    for i, (tag, name, col_type) in enumerate(_display_columns(table)):
        y = y_top - HEAD_H - ROW_H * (i + 0.5)
        if i:
            ax.plot([x + 0.05, x + width - 0.05], [y + ROW_H / 2] * 2, color=HAIRLINE, linewidth=0.8, zorder=3)
        ax.text(x + 0.18, y, tag, va="center", ha="left", fontsize=8.5, fontweight="bold", color=ACCENT, zorder=4)
        ax.text(x + 1.05, y, name, va="center", ha="left", fontsize=10.5, color=INK, zorder=4)
        ax.text(x + width - 0.18, y, col_type, va="center", ha="right", fontsize=9, color=INK2, zorder=4)


def _anchor(table: Table, side: str, frac: float) -> tuple[float, float]:
    x, y_top, width, height = _box_geometry(table)
    if side == "top":
        return x + width * frac, y_top
    if side == "bottom":
        return x + width * frac, y_top - height
    if side == "left":
        return x, y_top - height * frac
    return x + width, y_top - height * frac


# (parent side, frac, child side, frac) per (child, parent)
CONNECTORS = {
    ("transactions", "splits"): ("bottom", 0.5, "top", 0.12),
    ("transactions", "classes"): ("bottom", 0.5, "top", 0.5),
    ("transactions", "time_slots"): ("bottom", 0.5, "top", 0.88),
    ("transaction_features", "transactions"): ("bottom", 0.5, "top", 0.5),
    ("transactions", "amount_bands"): ("left", 0.5, "right", 0.5),
}


def _draw_relation(ax, rel: Relation, by_name: dict[str, Table]) -> None:
    p_side, p_frac, c_side, c_frac = CONNECTORS[(rel.child, rel.parent)]
    start = _anchor(by_name[rel.parent], p_side, p_frac)
    end = _anchor(by_name[rel.child], c_side, c_frac)
    style = {"color": INK2, "linewidth": 1.6, "zorder": 1, "solid_capstyle": "round"}
    if rel.logical:
        style["linestyle"] = (0, (5, 4))
    if p_side == "bottom":  # vertical - horizontal - vertical elbow
        mid_y = (start[1] + end[1]) / 2
        xs, ys = [start[0], start[0], end[0], end[0]], [start[1], mid_y, mid_y, end[1]]
    else:  # horizontal link, boxes are vertically offset -> horizontal - vertical - horizontal
        mid_x = (start[0] + end[0]) / 2
        xs, ys = [start[0], mid_x, mid_x, end[0]], [start[1], start[1], end[1], end[1]]
    ax.plot(xs, ys, **style)
    parent_mark, child_mark = ("1", "1" if rel.one_to_one else "N")
    ax.text(start[0] + (0.18 if p_side == "bottom" else -0.25), start[1] + (-0.3 if p_side == "bottom" else 0.18),
            parent_mark, fontsize=10, fontweight="bold", color=INK, ha="center", va="center", zorder=5)
    ax.text(end[0] + (0.18 if c_side == "top" else 0.25), end[1] + (0.3 if c_side == "top" else 0.18),
            child_mark, fontsize=10, fontweight="bold", color=INK, ha="center", va="center", zorder=5)


def draw_er_diagram(tables: list[Table], relations: list[Relation], path: Path) -> Path:
    by_name = {t.name: t for t in tables}
    fig, ax = plt.subplots(figsize=(13.5, 9.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 17.4)
    ax.set_ylim(-0.6, 11.7)
    ax.set_aspect("equal")
    ax.axis("off")
    for rel in relations:
        if (rel.child, rel.parent) in CONNECTORS:
            _draw_relation(ax, rel, by_name)
    for table in tables:
        if table.name in LAYOUT:
            _draw_table(ax, table)
    ax.text(0.4, 11.45, "Credit Card Fraud Detection: entity-relationship diagram (SQLite)", fontsize=14, fontweight="bold", color=INK, va="center")
    legend = ("PK primary key · FK foreign key · 1 / N cardinality (one parent row : many child rows) · "
              "solid line = FOREIGN KEY, dashed = logical range join")
    ax.text(0.4, -0.35, legend, fontsize=9, color=INK2, va="center")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_er_diagram(db_path: Path = SQLITE_PATH, out_dir: Path = DOCS_DIR) -> dict[str, Path]:
    """Read the database and write the PNG and the Mermaid markdown."""
    engine = get_engine(db_path)
    tables, relations = read_model(engine)
    engine.dispose()
    out_dir = Path(out_dir)
    png = draw_er_diagram(tables, relations, out_dir / "er_diagram.png")
    markdown = out_dir / "er_diagram.md"
    markdown.write_text(
        "# ER diagram\n\n"
        "Generated from the database by `python -m src.db.er_diagram` (do not edit by hand).\n\n"
        "![ER diagram](er_diagram.png)\n\n"
        "```mermaid\n" + to_mermaid(tables, relations) + "```\n",
        encoding="utf-8",
    )
    logger.info("ER diagram written to %s", out_dir)
    return {"png": png, "markdown": markdown}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    generate_er_diagram()
