import re
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    __allow_unmapped__ = False


def filter_review_date(review_date_text):
    # Regex pattern to match the date in "day month year" format
    date_pattern = r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b"
    result = re.findall(date_pattern, review_date_text)
    if result:
        return result[0]

    return review_date_text


def filter_review_location(review_date_text):
    # Regex pattern to match everything between "Reviewed in" and "on"
    pattern = r"Reviewed in(.*?)on"

    # Using re.search() to find the match
    match = re.search(pattern, review_date_text)

    if match:
        extracted_text = match.group(1).strip()
        return extracted_text
    else:
        return review_date_text


def remove_control_characters(text: str):
    # control_characters = "".join(
    #     c for c in text if c in string.printable and c not in string.ascii_letters
    # )
    return text.replace("\n", "")


def filter_helpful_vote(helpful_text: str):
    if helpful_text.lower().startswith("one"):
        return 1

    number_pattern = r"\b\d+\b"
    # Using re.search() to find the first occurrence of the number in the text
    match = re.search(number_pattern, helpful_text)
    if match:
        number = match.group()
        return number

    return helpful_text


def filter_review_rating(review_rating_text: str):
    return review_rating_text.split(" out of 5 stars")[0]
