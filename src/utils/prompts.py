from typing import Callable, List


def _create_prompt_generator(question_template: str) -> Callable:
    """Create a prompt generator function with a specific question template."""

    def generator(raw_choices: List[str], suffix: str, **kwargs) -> str:
        sound_name = kwargs.get("sound_name", None)
        question = kwargs.get("question", None)
        if question is None:
            question = question_template.format(sound_name=sound_name)
        preffixes = [f"({chr(65 + i)})" for i in range(len(raw_choices))]
        choices = "\n".join(
            [f"{preffix} {choice}" for preffix, choice in zip(preffixes, raw_choices)]
        )
        return f"{question}\n{choices}\n{suffix}"

    return generator


# Question templates for different prompt types
QUESTION_TEMPLATES = {
    "multimodal": (
        "What is the person doing when the {sound_name} sound is heard in the background? "
        "Note that the sound might not be present in the video, "
        "in which case the correct answer would be 'None of the above'."
    ),
    "text_only": (
        "Imagine there is a video taken from an egocentric perspective where the person is performing a set of activities. "
        "What is the most likely activity the person might be doing when the {sound_name} sound is heard in the background? "
        "Note that the sound might not be present in the scenario, in which case the correct answer would be 'None of the above'."
    ),
    "silent_video": (
        "I am providing you with a video without sound. "
        "Try to image what is the person doing when the {sound_name} sound is heard in the background of the video, "
        "even though you cannot hear the sound. Note that the sound might not be present at all, "
        "in which case the correct answer would be 'None of the above'."
    ),
    "audio": (
        "I am providing you with an audio extracted from a video. "
        "Try to image what is the most likely activity the person is doing when the {sound_name} sound is heard in the "
        "background of the video, even though you cannot see the video. "
        "Note that the {sound_name} sound might not be present in the audio, "
        "in which case the correct answer would be 'None of the above'."
    ),
    "overlayed_full_audio_classification": (
        "Listen to the following audio clip carefully. "
        "In this clip, there are several kitchen-environment sounds, but one of them is different or out of place. "
        "After listening to the audio, please identify which sound is not like the others. "
        "Choose the correct option from the list of multiple-choice answers below. "
        "Note that there might not be any sound that is different, "
        "in which case the correct answer would be 'None of the above'."
    ),
    "simple_audio_classification": (
        "Listen to the following audio clip carefully. "
        "After listening to the audio, identify which is the sound that you heard. "
        "Choose the correct option from the list of multiple-choice answers below. "
    ),
    "temporal_video": (
        "The following are four actions that occur in a video. "
        "Your task is to order them based on their temporal sequence as they happen in the video."
    ),
    "video_segment": (
        "Watch this short first-person (egocentric) video clip carefully. "
        "From the options below, select the action that most closely matches "
        "what the person is doing in the video. "
        "Choose the most appropriate option, even if it's does not appear to be an exact match. "
    ),
    "pipeline_video": (
        "These are frames from a video. What is the person doing in this video?"
    ),
}

# Create the prompt factory
prompt_factory = {
    prompt_type: _create_prompt_generator(template)
    for prompt_type, template in QUESTION_TEMPLATES.items()
}
