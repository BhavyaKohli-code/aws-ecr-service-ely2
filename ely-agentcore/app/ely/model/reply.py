"""Streams the answering model's reply to the caller while keeping its control text out of sight.

The model starts its reply with a marker when it won't answer ([NOT_AVAILABLE], [ASK_AREA]) and ends it with
a USED_PASSAGES line. A reply that starts with a marker is held back whole, for the caller to decide what to
show; any other reply is passed on as it arrives, except a line that may be the USED_PASSAGES line.

Strands builds the saved message from the same text pieces after the caller has seen them, so what a piece
is changed to here is also what conversation memory keeps (see ReplyFilter.feed).
"""
import re

NOT_AVAILABLE_MARKER = "[NOT_AVAILABLE]"
ASK_AREA_MARKER = "[ASK_AREA]"
MARKERS = (NOT_AVAILABLE_MARKER, ASK_AREA_MARKER)
USED_PASSAGES_LABEL = "USED_PASSAGES"
_USED_LINE = re.compile(r"^[ \t]*USED_PASSAGES\s*:", re.I | re.M)


def _could_be_used_line(line: str) -> bool:
    """Whether an unfinished line is, or may still turn into, the USED_PASSAGES line."""
    text = line.strip().upper()
    return bool(text) and (text.startswith(USED_PASSAGES_LABEL) or USED_PASSAGES_LABEL.startswith(text))


class ReplyFilter:
    """Fed the reply's text pieces in order. marker is set once the reply turns out to start with one."""

    def __init__(self):
        self.marker: str | None = None
        self.decided = False
        self.shown = ""    # text the caller has been given
        self._held = ""    # text received but not yet given to the caller

    def feed(self, text: str) -> tuple[str, str]:
        """(text to show the caller now, text this piece should become in the saved message)."""
        self._held += text
        if not self.decided:
            start = self._held.lstrip()
            if not start or any(m.startswith(start) and m != start for m in MARKERS):
                return "", ""  # may still be a marker: hold it, and save it once decided
            self.decided = True
            self.marker = next((m for m in MARKERS if start.startswith(m)), None)
            if self.marker:  # saved as written, everything held so far in this piece
                held, self._held = self._held, self._held
                return "", held
            self._held = start
        if self.marker:
            return "", text
        # Pass on everything except an unfinished last line that may be the USED_PASSAGES line, and anything
        # from a finished USED_PASSAGES line on
        cut = len(self._held)
        last_line_start = self._held.rfind("\n") + 1
        if _could_be_used_line(self._held[last_line_start:]):
            cut = last_line_start
        used_line = _USED_LINE.search(self._held, 0, cut)
        if used_line:
            cut = used_line.start()
        out, self._held = self._held[:cut], self._held[cut:]
        self.shown += out
        return out, out

    def rest(self) -> str:
        """Everything received but not shown: the whole reply after a marker, else the USED_PASSAGES line
        (or, if the model left that out, the reply's unfinished last line)."""
        return self._held
