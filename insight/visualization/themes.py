from __future__ import annotations

from pydantic import BaseModel, Field


class EditorialTheme(BaseModel):
    name: str = "editorial"
    background: str = "#f2ead8"
    foreground: str = "#1a1a1b"
    accent: str = "#8b1d1d"
    muted: str = "#3a3a3c"
    secondary_muted: str = "#525252"
    tertiary_muted: str = "#7a7a7c"
    warm_neutral: str = "#b09f93"
    font_family: str = "serif"
    font_serif_candidates: list[str] = Field(
        default_factory=lambda: ["EB Garamond", "Georgia", "Times New Roman", "DejaVu Serif"]
    )
    grid_style: str = ":"
    grid_alpha: float = 0.35
    dpi: int = 200

    @property
    def palette(self) -> list[str]:
        return [
            self.foreground,
            self.accent,
            self.muted,
            self.secondary_muted,
            self.tertiary_muted,
            self.warm_neutral,
        ]


EDITORIAL_THEME = EditorialTheme()

MINIMAL_THEME = EditorialTheme(
    name="minimal",
    background="#ffffff",
    foreground="#222222",
    accent="#2563eb",
    muted="#4b5563",
    secondary_muted="#6b7280",
    tertiary_muted="#9ca3af",
    warm_neutral="#d1d5db",
)

DARK_THEME = EditorialTheme(
    name="dark",
    background="#16161a",
    foreground="#e8e6e3",
    accent="#c9a227",
    muted="#a8a29e",
    secondary_muted="#78716c",
    tertiary_muted="#57534e",
    warm_neutral="#44403c",
    grid_alpha=0.25,
)

THEMES: dict[str, EditorialTheme] = {
    "editorial": EDITORIAL_THEME,
    "minimal": MINIMAL_THEME,
    "dark": DARK_THEME,
}
