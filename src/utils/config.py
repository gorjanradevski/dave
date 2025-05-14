import argparse
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Union


@dataclass
class DatasetConfig:
    """Configuration class for project settings with flexible overriding."""

    # Default configuration values
    MINIMUM_EVENT_DURATION: float = 2.0
    MAXIMUM_EVENT_DURATION: float = 60.0
    ENFORSE_UNIQUE_EVENTS: bool = False
    MIN_EVENT_DURATION_FOR_OVERLAY: int = 2.0
    MAX_OVERLAP: float = 0.5
    SOUND_CLASSES: List[str] = field(
        default_factory=lambda: [
            "dog",
            "crow",
            "clapping",
            "chainsaw",
            "church_bells",
            "clock_alarm",
            "car_horn",
            "laughing",
            "crying_baby",
            "coughing",
            "sneezing",
            "siren",
            "cat",
        ]
    )
    FADE_IN_OUT_DURATION: float = 0.5
    MAX_SEQUENCE_DURATION: float = 60.0
    # Words to filter an empty list by default
    WORDS_TO_FILTER: List[str] = field(default_factory=lambda: [])
    AUDIO_START_OFFSET: float = 0.0
    AUDIO_SCALE_COEFFICIENT: bool = 1.0
    PARTICIPANTS: List[str] = field(default_factory=lambda: [f"P{idx:02d}" for idx in range(1, 33)])
    VIDEO_RESIZE_PERCENTAGE: float = 0.25
    PROCESSED_VIDEOS_DIR_NAME: str = "EPIC-KITCHENS-processed"

    @classmethod
    def from_args(cls, args: Union[argparse.Namespace, Dict[str, Any]] = None):
        if hasattr(args, "__dict__"):
            args = vars(args)

        # Create a copy of the default configuration
        config_dict = asdict(cls())

        if args:
            for key, value in args.items():
                # Convert the key to uppercase (or lowercase) to match the class attribute format
                normalized_key = key.upper()

                # Only override if the value is not None and the key exists in the config
                if value is not None and normalized_key in config_dict:
                    # Handle case when value is an empty list
                    if isinstance(value, list) and len(value) == 0:
                        value = getattr(cls(), key)  # Use the default value if empty list
                    config_dict[normalized_key] = value

        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        # If the config dict is an empty dictionary report a message that the config is empty
        if not config_dict:
            print("*" * 50)
            print(
                "The configuration dictionary is empty! For AVQA is should be, the print of the default configuration is not relevant!"
            )
            print("*" * 50)
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compare_config(new_config, existing_config):
    # Return the comparison result: Mismatched keys and values if not equal, empty dict otherwise
    mismatched_values = {
        k: {"new": new_config[k], "existing": existing_config[k]}
        for k in new_config.keys()
        if new_config[k] != existing_config[k]
    }

    return mismatched_values
