import re
from typing import Any, List, Literal, Optional, Union

from langchain_text_splitters.base import TextSplitter


def _split_text_with_regex(
        text: str, separator: str, keep_separator: Union[bool, Literal["start", "end"]]
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if separator == "<FIXED_DELIMITER>":
            keep_separator = False
        if separator in ["\\\n\\\n", "\\\n", " ", ""]:
            keep_separator = False
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = (
                ([_splits[i] + _splits[i + 1] for i in range(0, len(_splits) - 1, 2)])
                if keep_separator == "end"
                else ([_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)])
            )
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = (
                (splits + [_splits[-1]])
                if keep_separator == "end"
                else ([_splits[0]] + splits)
            )
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != "" and s.strip() != ""]


class RecursiveCharacterTextSplitter(TextSplitter):
    """Splitting text by recursively look at characters.

    Recursively tries to split by different characters to find one
    that works.
    """

    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: Union[bool, Literal["start", "end"]] = "end",
            is_separator_regex: bool = False,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        # special fixed separator is <FIXED_DELIMITER>
        self._separators = separators or ["<FIXED_DELIMITER>", "\n\n", "。", ". ", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if separator == "<FIXED_DELIMITER>":
                final_chunks.append(s)
            else:
                if self._length_function(s) < self._chunk_size:
                    _good_splits.append(s)
                else:
                    if _good_splits:
                        merged_text = self._merge_splits(_good_splits, _separator)
                        final_chunks.extend(merged_text)
                        _good_splits = []
                    if not new_separators:
                        final_chunks.append(s)
                    else:
                        other_info = self._split_text(s, new_separators)
                        final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            self._chunk_overlap = 0
        if '<FIXED_DELIMITER>' in text:
            self._chunk_overlap = 0
        chunks = self._split_text(text, self._separators)
        if self._chunk_overlap > 0:
            overlap_chunks = [chunks[0]]
            for i in range(1, len(chunks)):
                overlap_part = chunks[i - 1][-self._chunk_overlap:]
                new_chunk = overlap_part + chunks[i]
                overlap_chunks.append(new_chunk)

            return overlap_chunks
        else:
            return chunks
