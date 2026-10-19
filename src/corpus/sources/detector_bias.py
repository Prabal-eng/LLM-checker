"""Liang et al. (2023) essays: non-native TOEFL writers and native student writers.

Used only as evaluation stress slices for false accusations against writer groups.
"""

import json
from pathlib import Path
from typing import Iterable

from ..schema import Label, LicenceClass, Role, Sample
from ._common import SourceSpec, compact, make_sample

_BASE = "https://raw.githubusercontent.com/Weixin-Liang/ChatGPT-Detector-Bias/main/Data_and_Results/Human_Data/"
# folder -> (label, writer group, transform, generator, domain)
_SETS = {
    "TOEFL_real_91": (Label.HUMAN, "non_native", "none", None, "toefl_essay"),
    "HewlettStudentEssay_real_88": (Label.HUMAN, "native", "none", None, "student_essay"),
    "CollegeEssay_real_70": (Label.HUMAN, "unknown", "none", None, "college_essay"),
    "CS224N_real_145": (Label.HUMAN, "unknown", "none", None, "course_abstract"),
    "TOEFL_gpt4polished_91": (Label.AI, "non_native", "llm_polish", "gpt-4", "toefl_essay"),
    "HewlettStudentEssay_GPTsimplify_88": (Label.AI, "native", "llm_simplify", "chatgpt", "student_essay"),
}

SPEC = SourceSpec(
    name="detector_bias",
    licence="other-open (Zenodo 10.5281/zenodo.7893958)",
    licence_class=LicenceClass.UNKNOWN,
    files=tuple((f"{_BASE}{name}/data.json", f"{name}.json") for name in _SETS),
)


def load(raw: Path, cap: int) -> Iterable[Sample]:
    for name, (label, writer_group, transform, generator, domain) in _SETS.items():
        with open(raw / f"{name}.json", encoding="utf-8") as fh:
            items = json.load(fh)
        yield from compact(
            make_sample(SPEC, text=item.get("document", ""), label=label, role=Role.STRESS,
                        record_id=f"{name}:{i}", domain=domain,
                        # Polished or simplified versions share a group with their originals.
                        group_key=f"{name.split('_')[0]}:{i}",
                        generator=generator, writer_group=writer_group, transform=transform)
            for i, item in enumerate(items[:cap])
        )
