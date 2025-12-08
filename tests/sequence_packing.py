"""Test the sequence packing function."""

from rlkit.data.sequence_packing import pack_sequences, distribute_bins_for_dp


def test_pack_sequences():
    """Test the sequence packing function."""
    SEPARATOR = {"input_ids": 98, "loss_mask": 0, "advantage": 0.0}
    MAX_BIN_SIZE = 20
    NUM_BINS = 2

    sequences = [
        {
            "input_ids": [0, 1, 2, 3, 4],
            "loss_mask": [1, 1, 1, 1, 1],
            "advantage": [0.1, 0.2, 0.3, 0.4, 0.5],
        },
        {
            "input_ids": [10, 11, 12, 13, 14, 15],
            "loss_mask": [1, 1, 1, 1, 1, 1],
            "advantage": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        },
        {
            "input_ids": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            "loss_mask": [1] * 10,
            "advantage": [2.0] * 10,
        },
        {
            "input_ids": [30, 31, 32],
            "loss_mask": [1, 1, 1],
            "advantage": [3.0, 3.1, 3.2],
        },
    ]

    bins, remainder = pack_sequences(sequences, max_bin_size=MAX_BIN_SIZE, num_bins=NUM_BINS, separator_value=SEPARATOR)

    assert len(bins) == NUM_BINS
    assert remainder == []

    # First bin: longest seq (10 tokens) + second longest (6 tokens) = 18 tokens (with separators)
    assert bins[0]["input_ids"] == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 98, 10, 11, 12, 13, 14, 15, 98]
    assert bins[0]["loss_mask"] == [1] * 10 + [0] + [1] * 6 + [0]
    assert bins[0]["advantage"] == [2.0] * 10 + [0.0] + [1.0, 1.1, 1.2, 1.3, 1.4, 1.5] + [0.0]

    # Second bin: 5 tokens + 3 tokens = 10 tokens (with separators)
    assert bins[1]["input_ids"] == [0, 1, 2, 3, 4, 98, 30, 31, 32, 98]
    assert bins[1]["loss_mask"] == [1, 1, 1, 1, 1, 0, 1, 1, 1, 0]
    assert bins[1]["advantage"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 3.0, 3.1, 3.2, 0.0]


def test_pack_sequences_with_remainder():
    """Test packing returns leftover sequences that don't fit."""
    SEPARATOR = {"input_ids": 99, "advantage": 0.0}
    MAX_BIN_SIZE = 10
    NUM_BINS = 2

    # Create sequences that won't all fit:
    # - Two 8-token sequences will each take a full bin (8 + 1 separator = 9)
    # - The 3-token and 2-token sequences won't fit anywhere
    sequences = [
        {"input_ids": [1, 2, 3], "advantage": [0.1, 0.2, 0.3]},                     # 3 tokens
        {"input_ids": [10, 11, 12, 13, 14, 15, 16, 17], "advantage": [1.0] * 8},   # 8 tokens
        {"input_ids": [20, 21], "advantage": [2.0, 2.1]},                           # 2 tokens
        {"input_ids": [30, 31, 32, 33, 34, 35, 36, 37], "advantage": [3.0] * 8},   # 8 tokens
    ]

    bins, remainder = pack_sequences(
        sequences,
        max_bin_size=MAX_BIN_SIZE,
        num_bins=NUM_BINS,
        separator_value=SEPARATOR,
    )

    assert len(bins) == NUM_BINS

    # The two 8-token sequences should be packed (one per bin)
    assert bins[0]["input_ids"] == [10, 11, 12, 13, 14, 15, 16, 17, 99]
    assert bins[0]["advantage"] == [1.0] * 8 + [0.0]
    assert bins[1]["input_ids"] == [30, 31, 32, 33, 34, 35, 36, 37, 99]
    assert bins[1]["advantage"] == [3.0] * 8 + [0.0]

    # The 3-token and 2-token sequences should be in remainder
    # (sorted by length descending, so 3-token first)
    assert remainder == [
        {"input_ids": [1, 2, 3], "advantage": [0.1, 0.2, 0.3]},
        {"input_ids": [20, 21], "advantage": [2.0, 2.1]},
    ]


def test_distribute_bins_for_dp_snake_ordering():
    """Test that snake ordering balances token counts across shards."""
    NUM_SHARDS = 2

    # Bins with varying lengths (token counts)
    # Sorted descending: [10, 8, 6, 4] tokens
    bins = [
        {"input_ids": [1] * 10, "advantage": [0.1] * 10},  # 10 tokens
        {"input_ids": [2] * 4, "advantage": [0.2] * 4},    # 4 tokens
        {"input_ids": [3] * 8, "advantage": [0.3] * 8},    # 8 tokens
        {"input_ids": [4] * 6, "advantage": [0.4] * 6},    # 6 tokens
    ]

    shards = distribute_bins_for_dp(bins, num_shards=NUM_SHARDS)

    assert len(shards) == NUM_SHARDS

    # Snake ordering on sorted [10, 8, 6, 4]:
    # Shard 0 gets positions 0, 3 -> bins with 10, 4 tokens = 14 total
    # Shard 1 gets positions 1, 2 -> bins with 8, 6 tokens = 14 total
    def count_tokens(shard_bins):
        return sum(len(bin_dict["input_ids"]) for bin_dict in shard_bins)

    shard_0_tokens = count_tokens(shards[0])
    shard_1_tokens = count_tokens(shards[1])

    assert shard_0_tokens == 14, f"Shard 0 should have 14 tokens, got {shard_0_tokens}"
    assert shard_1_tokens == 14, f"Shard 1 should have 14 tokens, got {shard_1_tokens}"

    # Each shard should have 2 bins
    assert len(shards[0]) == 2
    assert len(shards[1]) == 2

    # Each bin should be a dict with the expected keys
    for shard in shards:
        for bin_dict in shard:
            assert "input_ids" in bin_dict
            assert "advantage" in bin_dict


def test_distribute_bins_for_dp_preserves_data():
    """Test that distribution preserves original bin data."""
    bins = [
        {"input_ids": [1, 2, 3], "loss_mask": [1, 1, 1]},           # 3 tokens
        {"input_ids": [4, 5, 6, 7], "loss_mask": [1, 1, 1, 1]},     # 4 tokens
    ]

    shards = distribute_bins_for_dp(bins, num_shards=2)

    # Each shard should have 1 bin
    assert len(shards[0]) == 1
    assert len(shards[1]) == 1

    # Snake ordering: sorted descending is [4-token, 3-token]
    # Position 0 -> shard 0, position 1 -> shard 1
    assert shards[0][0]["input_ids"] == [4, 5, 6, 7]
    assert shards[0][0]["loss_mask"] == [1, 1, 1, 1]
    assert shards[1][0]["input_ids"] == [1, 2, 3]
    assert shards[1][0]["loss_mask"] == [1, 1, 1]


def test_pack_sequences_empty():
    """Test packing with empty sequences dict."""
    SEPARATOR = {"input_ids": 0, "advantage": 0.0}
    bins, remainder = pack_sequences([], max_bin_size=10, num_bins=2, separator_value=SEPARATOR)

    assert len(bins) == 2
    assert all(b["input_ids"] == [] for b in bins)
    assert all(b["advantage"] == [] for b in bins)
    assert remainder == []


def test_distribute_bins_for_dp_empty():
    """Test distribution with empty bins list."""
    shards = distribute_bins_for_dp([], num_shards=2)

    assert len(shards) == 2
    assert shards[0] == []
    assert shards[1] == []


def test_pack_sequences_with_doc_priorities_basic():
    """Test packing with priority keys ensures high-priority (stale) sequences are packed first."""
    SEPARATOR = {"input_ids": 99, "value": 0.0}
    MAX_BIN_SIZE = 20
    NUM_BINS = 2

    # Without priority: FFD would pack by length [8, 6, 4, 3]
    #   -> bin0: [8-tok, 6-tok, sep, sep] = 16, bin1: [4-tok, 3-tok, sep, sep] = 10
    # With priority [0, 3, 1, 2]: pack order is seq[1], seq[3], seq[2], seq[0]
    #   i.e., lengths in order: [6, 4, 8, 3]
    sequences = [
        {"input_ids": [1] * 8, "value": [1.0] * 8},   # idx=0, len=8, priority=0
        {"input_ids": [2] * 6, "value": [2.0] * 6},   # idx=1, len=6, priority=3 (highest)
        {"input_ids": [3] * 3, "value": [3.0] * 3},   # idx=2, len=3, priority=1
        {"input_ids": [4] * 4, "value": [4.0] * 4},   # idx=3, len=4, priority=2
    ]
    doc_priorities = [0, 3, 1, 2]

    bins, remainder = pack_sequences(
        sequences,
        max_bin_size=MAX_BIN_SIZE,
        num_bins=NUM_BINS,
        separator_value=SEPARATOR,
        doc_priorities=doc_priorities,
    )

    assert len(bins) == NUM_BINS
    assert remainder == []

    # Packing order by (priority desc, length desc): seq[1], seq[3], seq[2], seq[0]
    # - seq[1] (len=6, pri=3): bin0 fits -> bin0=[2]*6+[99], size=7
    # - seq[3] (len=4, pri=2): bin0 fits (7+5=12<=20) -> bin0=[2]*6+[99]+[4]*4+[99], size=12
    # - seq[2] (len=3, pri=1): bin0 fits (12+4=16<=20) -> bin0=[2]*6+[99]+[4]*4+[99]+[3]*3+[99], size=16
    # - seq[0] (len=8, pri=0): bin0 can't fit (16+9=25>20), bin1 fits -> bin1=[1]*8+[99], size=9

    # bin0 should have: seq[1], seq[3], seq[2] in that order
    assert bins[0]["input_ids"] == [2]*6 + [99] + [4]*4 + [99] + [3]*3 + [99]
    assert bins[0]["value"] == [2.0]*6 + [0.0] + [4.0]*4 + [0.0] + [3.0]*3 + [0.0]

    # bin1 should have: seq[0]
    assert bins[1]["input_ids"] == [1]*8 + [99]
    assert bins[1]["value"] == [1.0]*8 + [0.0]


def test_pack_sequences_doc_priorities_same_priority_uses_length():
    """Test that within the same priority level, longer sequences are packed first."""
    SEPARATOR = {"input_ids": 99}
    MAX_BIN_SIZE = 25
    NUM_BINS = 2

    # All sequences have same priority, so should fall back to FFD (length descending)
    sequences = [
        {"input_ids": [1] * 3},   # len=3
        {"input_ids": [2] * 10},  # len=10
        {"input_ids": [3] * 5},   # len=5
        {"input_ids": [4] * 8},   # len=8
    ]
    doc_priorities = [1, 1, 1, 1]  # all same priority

    bins, remainder = pack_sequences(
        sequences,
        max_bin_size=MAX_BIN_SIZE,
        num_bins=NUM_BINS,
        separator_value=SEPARATOR,
        doc_priorities=doc_priorities,
    )

    # Should pack in order: seq[1](len=10), seq[3](len=8), seq[2](len=5), seq[0](len=3)
    # - seq[1] (len=10): bin0=[2]*10+[99], size=11
    # - seq[3] (len=8): bin0 fits (11+9=20<=25) -> bin0=[2]*10+[99]+[4]*8+[99], size=20
    # - seq[2] (len=5): bin0 can't fit (20+6=26>25), bin1 fits -> bin1=[3]*5+[99], size=6
    # - seq[0] (len=3): bin0 can't fit (20+4=24<=25!) wait, 20+4=24 which is <=25
    #   Actually bin0 can fit! -> bin0=[2]*10+[99]+[4]*8+[99]+[1]*3+[99], size=24

    assert bins[0]["input_ids"] == [2]*10 + [99] + [4]*8 + [99] + [1]*3 + [99]
    assert bins[1]["input_ids"] == [3]*5 + [99]
    assert remainder == []


def test_pack_sequences_doc_priorities_with_remainder():
    """Test priority packing with remainder returns high-priority items first, low-priority in remainder."""
    SEPARATOR = {"input_ids": 99}
    MAX_BIN_SIZE = 10
    NUM_BINS = 1

    sequences = [
        {"input_ids": [1] * 8},   # len=8, priority=0 (low)
        {"input_ids": [2] * 7},   # len=7, priority=2 (high)
        {"input_ids": [3] * 3},   # len=3, priority=1 (medium)
    ]
    doc_priorities = [0, 2, 1]

    bins, remainder = pack_sequences(
        sequences,
        max_bin_size=MAX_BIN_SIZE,
        num_bins=NUM_BINS,
        separator_value=SEPARATOR,
        doc_priorities=doc_priorities,
    )

    # Pack order: seq[1](pri=2,len=7), seq[2](pri=1,len=3), seq[0](pri=0,len=8)
    # - seq[1] (len=7): bin0=[2]*7+[99], size=8
    # - seq[2] (len=3): bin0 can't fit (8+4=12>10), no more bins -> remainder
    # - seq[0] (len=8): already in remainder phase

    assert bins[0]["input_ids"] == [2]*7 + [99]

    # Remainder should be seq[2] then seq[0] (in priority order)
    assert len(remainder) == 2
    assert remainder[0]["input_ids"] == [3]*3  # pri=1
    assert remainder[1]["input_ids"] == [1]*8  # pri=0
