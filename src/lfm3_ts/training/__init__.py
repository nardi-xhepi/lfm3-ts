# Copyright 2026 Nardi Xhepi.
#
# This file is part of LFM3-TS, distributed under the LFM Open License v1.0: see LICENSE and NOTICE at the root
# of the repository.
"""What training LFM3-TS needs on top of the model: the ChatTS record format, the generated reading questions,
streams that mix datasets, the collator and the token-budget batches.

Install the optional dependencies with `pip install "lfm3-ts[train]"` (accelerate, datasets, peft). `import lfm3_ts`
does not import this subpackage.
"""

from .data import (
    CHAT,
    CHATTS_DATASET,
    GENERATED,
    HOLDOUT,
    INJECTED,
    REAL,
    ChatReplay,
    InjectedAnomalies,
    Lfm3TsDataCollator,
    MixedStream,
    ReadingTasks,
    RealSeries,
    ScheduledStream,
    TokenBudgetBatches,
    chatts_record_to_messages,
    fix_stated_lengths,
    load_chatts_holdout,
    load_chatts_stream,
    written_value,
)


__all__ = [
    "CHAT",
    "CHATTS_DATASET",
    "GENERATED",
    "HOLDOUT",
    "INJECTED",
    "REAL",
    "ChatReplay",
    "InjectedAnomalies",
    "Lfm3TsDataCollator",
    "MixedStream",
    "ReadingTasks",
    "RealSeries",
    "ScheduledStream",
    "TokenBudgetBatches",
    "chatts_record_to_messages",
    "fix_stated_lengths",
    "load_chatts_holdout",
    "load_chatts_stream",
    "written_value",
]
