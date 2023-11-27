from pathlib import Path
from typing import Optional, Union

from robotools.worklists.base import BaseWorklist

__all__ = ("FluentWorklist",)


class FluentWorklist(BaseWorklist):
    """Context manager for the creation of Tecan Fluent worklists."""

    def __init__(
        self,
        filepath: Optional[Union[str, Path]] = None,
        max_volume: Union[int, float] = 950,
        auto_split: bool = True,
    ) -> None:
        raise NotImplementedError("Be patient.")
        super().__init__(filepath, max_volume, auto_split)
