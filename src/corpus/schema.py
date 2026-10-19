"""Provenance record carried by every sample in the authorship corpus."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Label(str, Enum):
    HUMAN = "human"
    AI = "ai"


class LicenceClass(str, Enum):
    PERMISSIVE = "permissive"
    COPYLEFT = "copyleft"
    NON_COMMERCIAL = "non_commercial"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class Role(str, Enum):
    """How a sample may be used. Only CORE samples are eligible for training."""

    CORE = "core"
    STRESS = "stress"          # hard cases: edited, paraphrased, attacked, non-native writers
    INHOUSE = "inhouse"        # presumed-human rater text; evaluation only until audited
    DIAGNOSTIC = "diagnostic"  # unverified labels; never trained on or used as a gate


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    TEST_UNSEEN_GENERATOR = "test_unseen_generator"
    STRESS = "stress"
    INHOUSE = "inhouse"
    DIAGNOSTIC = "diagnostic"


# Splits a model may learn from or be tuned on; everything else is evaluation.
FIT_SPLITS = (Split.TRAIN, Split.VAL)
EVALUATION_SPLITS = (Split.TEST, Split.TEST_UNSEEN_GENERATOR, Split.STRESS, Split.INHOUSE)
LOCKED_SPLITS = EVALUATION_SPLITS


class Sample(BaseModel):
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str = Field(min_length=1)
    label: Label
    role: Role
    source_dataset: str = Field(min_length=1)
    source_record_id: str = Field(min_length=1)
    licence: str = Field(min_length=1)
    licence_class: LicenceClass
    verification: str = Field(min_length=1)
    generator: Optional[str] = None
    generator_family: Optional[str] = None
    domain: str = Field(min_length=1)
    language: str = "en"
    writer_group: str = "unknown"
    transform: str = "none"
    group_key: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_generator(self) -> "Sample":
        if self.label is Label.AI and not (self.generator and self.generator_family):
            raise ValueError("AI samples must record generator and generator_family")
        if self.label is Label.HUMAN and (self.generator or self.generator_family):
            raise ValueError("human samples cannot record a generator")
        return self
