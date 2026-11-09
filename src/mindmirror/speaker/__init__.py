"""Speaker identification, shared context, and runtime profile tuning."""

from mindmirror.speaker.context import SpeakerContext
from mindmirror.speaker.identifier import (
    AcousticVoiceEmbedder,
    SpeakerIdentifierComponent,
    SpeakerRegistry,
)
from mindmirror.speaker.profile import SpeakerProfile, SpeakerProfileStore

__all__ = [
    "AcousticVoiceEmbedder",
    "SpeakerContext",
    "SpeakerIdentifierComponent",
    "SpeakerProfile",
    "SpeakerProfileStore",
    "SpeakerRegistry",
]
