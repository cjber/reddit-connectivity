import os
import re
from pathlib import Path

import polars as pl

pl.Config.set_tbl_formatting("NOTHING")
pl.Config.set_tbl_dataframe_shape_below(True)
pl.Config.set_tbl_rows(6)


class Paths:
    RAW_DATA = Path(os.environ["DATA_DIR"])
    DATA = Path("data")
    PROCESSED = DATA / "processed"
    OUT = DATA / "out"


class Const:
    GER_MODEL = "bert-base-uncased"
    MAX_TOKEN_LEN = 256
    SEED = 42

    with open(Paths.PROCESSED / "exclude.txt", "r") as exclude:
        EXCLUDE = {line.strip() for line in exclude.readlines()}


class Label:
    labels: dict[str, int] = {"O": 0, "B-location": 1, "I-location": 2}
    idx: dict[int, str] = {v: k for k, v in labels.items()}
    count: int = len(labels)


EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "]+",
    flags=re.UNICODE,
)
AT_HASH_PATTERN = re.compile(r"\s([@#])(?!\w+\b)")
RETWEET_PATTERN = re.compile(r"\b(retweet)\b")


def extract_emoji(text):
    """
    Extract all emoji characters from text.
    Args:
        text: A string containing text to extract emoji from.
    Returns:
        A string containing all extracted emoji characters separated by spaces.
    """
    return " ".join(EMOJI_PATTERN.findall(text))


# for reddit or stackoverflow
def _remove_markdown(text):
    """
    Remove markdown formatting and symbols from text.
    Args:
        text: A string containing text to remove markdown from.
    Returns:
        A string containing text with markdown and symbols removed.
    """
    text = re.sub(r"^#{1,6}\s+", "", text)  # Remove headings (e.g. ## Heading)
    text = re.sub(EMOJI_PATTERN, "", text)  # Remove emoji
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # Remove bold (e.g. **Bold**)
    text = re.sub(r"\*(.*?)\*", r"\1", text)  # Remove italic (e.g. *Italic*)
    text = re.sub(
        r"^\>\s+", "", text, flags=re.MULTILINE
    )  # Remove blockquote (e.g. > Quote)
    text = re.sub(
        r"^\d\.\s+", "", text, flags=re.MULTILINE
    )  # Remove ordered list (e.g. 1. Item)
    text = re.sub(
        r"^\-\s+", "", text, flags=re.MULTILINE
    )  # Remove unordered list (e.g. - Item)
    text = re.sub(r"`(.+?)`", r"\1", text)  # Remove code (e.g. `Code`)
    text = re.sub(
        r"^\s*---\s*$", "", text, flags=re.MULTILINE
    )  # Remove horizontal rule (e.g. ---)
    text = re.sub(
        r"\[(.*?)\]\((.*?)\)", r"\1", text
    )  # Remove link (e.g. [Link text](URL))
    text = re.sub(
        r"\!\[(.*?)\]\((.*?)\)", r"", text
    )  # Remove image (e.g. ![Alt text](URL))
    text = re.sub(r"#(\S+)", r"\1", text)  # Remove hashtag symbol (e.g. #hashtag)
    text = re.sub(r"\n", " ", text)  # Remove newline characters
    text = re.sub(r"\t", " ", text)  # Remove tab characters
    # Remove URLs, HTML tags
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"bit.ly\S+", "", text)
    text = re.sub(r"bitly\S+", "", text)
    text = re.sub(r"pic.twitter\S+", "", text)
    text = re.sub(r"youtube.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)  # remove html tags
    text = text.strip()  # Remove leading and trailing whitespace
    text = text.lower()
    return text


def _remove_twitter(text):
    """
    Remove Twitter-specific symbols from text. From
    https://github.com/hengee/preprocessing_for_social_media/blob/main/preprocessing.py

    Args:
        text: A string containing text to remove Twitter-specific symbols from.
    Returns:
        A string containing text with Twitter-specific symbols removed.
    """
    # replace usernames with [user]
    text = re.sub(
        r"@\S+", "@user", text
    )  # replace usernames with @user, @ and user exist in the vocab of sentence
    # transformer
    text = re.sub(AT_HASH_PATTERN, "", text)  # remove @ and # symbols
    text = re.sub(
        RETWEET_PATTERN, "rt", text
    )  # replace the word 'retweet' with 'rt', rt exist in the vocab of sentence
    # transformer but not retweet

    # since it has previously removed the emoji, we don't need to remove it again,
    # but to play safe, we can encode it to utf-8
    text = text.encode(
        "utf-8", errors="ignore"
    )  # usually there's error involve with emoji, so we need to encode it to utf-8

    return text


def preprocess(text, is_twitter=True):
    """
    Preprocess social media text using remove_markdown and remove_twitter.
    Also returns the emojis in the text.
    Args:
        text (str): The social media text to preprocess.
    Returns:
        tuple: A tuple containing two items:
            - str: A string containing all extracted emoji characters
                   separated by spaces.
            - str: The preprocessed text.
    """
    # extract emoji first
    # emojis = extract_emoji(text)

    # remove markdown
    text = _remove_markdown(text)
    # remove twitter specific symbols

    if is_twitter:
        text = _remove_twitter(text)

    # print emojis first and then text in the same line
    return text
