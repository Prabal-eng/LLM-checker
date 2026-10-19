"""Legacy in-house evaluation sheets (pseudonymized).

Rater text is presumed human but unaudited, so it is evaluation-only. Template-built
"synthetic AI" and identity-inferred "copypasta" rows are diagnostic and never gate a release.
"""

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..schema import Label, LicenceClass, Role, Sample
from ._common import SourceSpec, make_sample

DATASET = Path(__file__).resolve().parents[3] / "data" / "training" / "dataset_full.csv"

SPEC = SourceSpec(name="legacy_inhouse", licence="internal", licence_class=LicenceClass.INTERNAL)

_SOURCES = {
    "human_gempix": (Label.HUMAN, Role.INHOUSE, "presumed_human_unaudited", None, "gempix_rationale"),
    "human_personality": (Label.HUMAN, Role.INHOUSE, "presumed_human_unaudited", None, "personality_rationale"),
    "synthetic_ai": (Label.AI, Role.DIAGNOSTIC, "template_synthetic", "template", "synthetic_template"),
    "real_ai_copypasta": (Label.AI, Role.DIAGNOSTIC, "inferred_from_rater_identity", "unknown", "gempix_rationale"),
}


def load(raw: Path, cap: int, dataset: Path = DATASET) -> Iterable[Sample]:
    df = pd.read_csv(dataset, keep_default_na=False)
    df = df[df["source"].isin(_SOURCES)]
    for r in df.reset_index().itertuples():
        label, role, verification, generator, domain = _SOURCES[r.source]
        sample = make_sample(SPEC, text=r.text, label=label, role=role, record_id=f"row:{r.index}",
                             domain=domain, group_key=f"rater:{r.rater}", verification=verification,
                             generator=generator)
        if sample is not None:
            yield sample
