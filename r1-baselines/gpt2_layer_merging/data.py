"""Dataset loading and text preparation."""

from datasets import load_dataset


def join_nonempty(split):
    """Join non-empty WikiText examples into one text string."""
    return "\n\n".join(
        x for x in split["text"]
        if x is not None and x.strip()
    )


def load_wikitext():
    """Load WikiText-2 raw data and return train/validation/test text."""
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
    )

    return (
        join_nonempty(dataset["train"]),
        join_nonempty(dataset["validation"]),
        join_nonempty(dataset["test"]),
    )
