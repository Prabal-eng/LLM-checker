"""Support gate: language check."""

import pytest

from src.authorship.language import check

ENGLISH = ("The image shows the correct landmarks, but the labels on the left side are cut off and "
           "the legend does not match the colors that are used in the chart, which makes it hard to read.")
NON_ENGLISH = {
    "french": "L'image montre les bons monuments, mais les étiquettes du côté gauche sont coupées et la "
              "légende ne correspond pas aux couleurs utilisées dans le graphique, ce qui la rend difficile à lire.",
    "spanish": "La imagen muestra los monumentos correctos, pero las etiquetas del lado izquierdo están cortadas "
               "y la leyenda no coincide con los colores que se usan en el gráfico, lo que dificulta su lectura.",
    "german": "Das Bild zeigt die richtigen Sehenswürdigkeiten, aber die Beschriftungen auf der linken Seite sind "
              "abgeschnitten und die Legende passt nicht zu den Farben, die im Diagramm verwendet werden.",
    "hindi": "चित्र में सही स्थल दिखाए गए हैं, लेकिन बाईं ओर के लेबल कटे हुए हैं और किंवदंती चार्ट में उपयोग किए गए रंगों से मेल नहीं खाती।",
    "japanese": "画像には正しい名所が表示されていますが、左側のラベルが切れており、凡例がグラフの色と一致していません。",
}


def test_english_prose_passes():
    assert check(ENGLISH).is_english


@pytest.mark.parametrize("language", sorted(NON_ENGLISH))
def test_other_languages_fail(language):
    assert not check(NON_ENGLISH[language]).is_english


def test_text_without_letters_fails():
    assert not check("12345 !!! 678").is_english


def test_layout_is_removed_before_scoring():
    from src.corpus.text import collapse_whitespace

    assert collapse_whitespace("First point.\n\nSecond point.\n  Third\tpoint. ") == "First point. Second point. Third point."


def test_confound_probe_sees_blank_lines():
    import pandas as pd

    from src.corpus.confound import FEATURES, surface_features

    rows = surface_features(pd.Series(["one line", "para one\n\npara two"]))
    blank = FEATURES.index("has_blank_line")
    assert rows[0, blank] == 0.0 and rows[1, blank] == 1.0 and rows.shape[1] == len(FEATURES)


def test_typography_view_maps_device_punctuation_to_ascii():
    """Amendment 5: curly quotes, dashes and ellipses depend on the typing device, not the author."""
    from src.corpus.text import text_view
    view = text_view("typography+collapse_whitespace")
    assert view("“Hi”—it’s ‘ok’… fine\n\n2–3") == "\"Hi\"-it's 'ok'... fine 2-3"


def test_legacy_view_keeps_typography_and_unknown_views_fail_closed():
    from src.corpus.text import text_view
    assert text_view("collapse_whitespace")("it’s\n ok") == "it’s ok"
    assert text_view(None)("it’s\n ok") == "it’s\n ok"
    try:
        text_view("mystery")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown text view must not silently fall back")


def test_detector_prepares_text_with_its_recorded_view():
    from src.authorship.detector import AuthorshipModelDetector
    detector = AuthorshipModelDetector("unused", platt=(1.0, 0.0), text_view="typography+collapse_whitespace")
    assert detector._prep("it’s “fine”") == "it's \"fine\""


def test_shortcut_probe_reads_the_typography_view():
    """AI text that differs from human text only in curly quotes must not look separable to the probe."""
    import pandas as pd
    from src.corpus.confound import shortcut_probe
    rows = []
    for i in range(400):
        base = f"Answer number {i} says it's fine and that the \"plan\" works well for {i % 7} people."
        split = "train" if i < 300 else "val"
        rows.append({"text": base, "label": "human", "split": split, "role": "core", "source_dataset": "s"})
        rows.append({"text": base.replace("'", "’").replace('"plan"', "“plan”"), "label": "ai",
                     "split": split, "role": "core", "source_dataset": "s"})
    probe = shortcut_probe(pd.DataFrame(rows))
    assert probe["text_view"] == "format4+typography+collapse_whitespace"
    assert probe["overall_auc"] < 0.6


def test_format_view_drops_markdown_and_list_markers():
    """Ralph R08-2: asterisks and list markers were in 17% of Dolly's AI answers and 1% of its human
    ones, so formatting could stand in for authorship. The model reads text without them."""
    from src.corpus.text import text_view
    view = text_view("format+typography+collapse_whitespace")
    text = ("## Plan\n1. *Fresh* basil\n2) __dried__ oregano\n- salt\n* pepper\n\u2022 oil\n"
            "> quoted `code`\n---\nsnake_case stays")
    assert view(text) == "Plan Fresh basil dried oregano salt pepper oil quoted code snake_case stays"
    assert text_view("typography+collapse_whitespace")("*Fresh* basil") == "*Fresh* basil"


def test_shortcut_probe_does_not_see_markdown_formatting():
    import pandas as pd
    from src.corpus.confound import shortcut_probe
    rows = []
    for i in range(400):
        base = f"Answer number {i} says the plan works well for {i % 7} people and costs little."
        split = "train" if i < 300 else "val"
        rows.append({"text": base, "label": "human", "split": split, "role": "core", "source_dataset": "s"})
        marked = (f"1. **{base}**", f"- *{base}*", f"## {base}", f"> `{base}`")[i % 4]
        rows.append({"text": marked, "label": "ai", "split": split, "role": "core", "source_dataset": "s"})
    probe = shortcut_probe(pd.DataFrame(rows))
    assert probe["text_view"] == "format4+typography+collapse_whitespace"
    assert probe["overall_auc"] < 0.6


def test_excerpts_come_from_the_formatting_free_text():
    """Ralph R09-1: excerpts were cut from whitespace-collapsed text, so list markers that the view
    strips at line starts survived inside excerpts, and the short model learned them."""
    import re
    from src.authorship.excerpts import excerpt
    lines = [f"{i}. Step {i} keeps the dough resting while the oven heats for a while." for i in range(1, 15)]
    text = "Here is how to bake it well.\n" + "\n".join(lines) + "\n- Serve warm.\n* Enjoy it."
    for key in ("a", "b", "c", "d"):
        cut = excerpt(text, key)
        assert cut and not re.search(r"(^| )(\d{1,3}[.)]|[-*])( |$)", cut), cut


def test_inline_bullets_and_spaced_dashes_are_dropped_but_hyphens_stay():
    """Ralph R09-1 and R09-6: a one-line dash list evaded detection, and a spaced hyphen marked HC3's
    human text (17% against 2%). Word-internal hyphens stay."""
    from src.corpus.text import text_view
    view = text_view("format2+typography+collapse_whitespace")
    assert view("Steps - mix -- bake \u2022 serve \u2014 a well-known trick") == "Steps mix bake serve a well-known trick"


def test_model_view_is_idempotent_and_fingerprinted():
    """A view applied twice (an excerpt already in the view, then the model's _prep) must not change;
    its fingerprint changes when its behaviour does (Ralph R09-3)."""
    from src.corpus.text import MODEL_TEXT_VIEW, text_view, text_view_fingerprint
    view = text_view(MODEL_TEXT_VIEW)
    for text in ("## A\n1. *b* c\n- d - e", "it\u2019s \u201cfine\u201d \u2014 ok", "x_y _z_ `q`"):
        assert view(view(text)) == view(text)
    assert text_view_fingerprint(MODEL_TEXT_VIEW) != text_view_fingerprint("typography+collapse_whitespace")
    assert len(text_view_fingerprint(MODEL_TEXT_VIEW)) == 64


def test_shortcut_probe_sees_spaced_dashes_only_through_the_view():
    """Ralph R09-6: the probe now carries formatting features, read on the model's view."""
    import pandas as pd
    from src.corpus.confound import FEATURES, shortcut_probe
    assert {"asterisks", "backticks", "spaced_dashes", "inline_numbered", "bullets"} <= set(FEATURES)
    rows = []
    for i in range(400):
        base = f"Answer number {i} says the plan works well for {i % 7} people and costs little"
        split = "train" if i < 300 else "val"
        rows.append({"text": base.replace(" says ", " - says - "), "label": "human", "split": split, "role": "core",
                     "source_dataset": "s"})
        rows.append({"text": base, "label": "ai", "split": split, "role": "core", "source_dataset": "s"})
    assert shortcut_probe(pd.DataFrame(rows))["overall_auc"] < 0.6
