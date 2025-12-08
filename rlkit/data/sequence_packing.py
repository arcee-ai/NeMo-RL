"""Sequence packing and bin distribution utilities."""

PackableDataDict = dict[str, list[int | float]]

def pack_sequences(
    documents: list[PackableDataDict],
    max_bin_size: int,
    num_bins: int,
    separator_value: dict[str, int | float],
    doc_priorities: list[int] | None = None,
) -> tuple[list[PackableDataDict], list[PackableDataDict]]:
    """Pack documents into bins based on the modified first-fit decreasing algorithm.

    Args:
        documents: List of dictionaries, each containing a sequence of data.
                    All dictionaries must have the same keys and all lists within a dictionary must have the same length.
        max_bin_size: Maximum size of each bin.
        num_bins: Number of bins to pack into.
        separator_value: Dictionary mapping field names to their separator values.
        doc_priorities: Optional list of priority values (e.g., ages) for each document. Higher values
            are packed first. Within the same priority level, longer sequences are packed first.
            If None, uses standard first-fit decreasing (longest sequences first).

    Returns:
        Tuple of (bins, remainder). Bins is a list of num_bins bins, each containing at most
        max_bin_size tokens. Each bin is a dictionary with the same keys as the input documents.
        Remainder is a list of documents that did not fit into the bins.
    """
    # Handle empty list of documents.
    if len(documents) == 0:
        return [{k: [] for k in separator_value}] * num_bins, []

    if doc_priorities is not None and len(doc_priorities) != len(documents):
        raise ValueError(f"doc_priorities length ({len(doc_priorities)}) must match documents length ({len(documents)})")

    keys = list(documents[0].keys())

    # Get sequence length for a document index
    def seq_len(doc_idx: int) -> int:
        return len(documents[doc_idx][keys[0]])

    # Initialize bins as list of dicts
    bins: list[PackableDataDict] = [{k: [] for k in keys} for _ in range(num_bins)]
    bin_sizes = [0] * num_bins

    # Sort document indices by priority (descending) then by sequence length (descending)
    if doc_priorities is not None:
        sorted_doc_indices = sorted(
            range(len(documents)),
            key=lambda i: (doc_priorities[i], seq_len(i)),
            reverse=True
        )
    else:
        sorted_doc_indices = sorted(range(len(documents)), key=seq_len, reverse=True)

    for sorted_indices_idx, doc_idx in enumerate(sorted_doc_indices):
        length = seq_len(doc_idx)
        for bin_idx, bin_dict in enumerate(bins):
            # Check if the sequence can fit in the bin (plus separator)
            if bin_sizes[bin_idx] + length + 1 <= max_bin_size:
                for key in keys:
                    bin_dict[key].extend(documents[doc_idx][key])
                    bin_dict[key].append(separator_value[key])
                bin_sizes[bin_idx] += length + 1
                break
            elif length == max_bin_size and bin_sizes[bin_idx] == 0:
                # If the sequence takes up the entire bin, add it to this empty bin with no separator
                for key in keys:
                    bin_dict[key].extend(documents[doc_idx][key])
                bin_sizes[bin_idx] = length
                break
        else:
            remaining_indices = sorted_doc_indices[sorted_indices_idx:]
            remaining_documents = [documents[i] for i in remaining_indices]
            return bins, remaining_documents

    return bins, []


def distribute_bins_for_dp(
    bins: list[PackableDataDict],
    num_shards: int,
) -> list[list[PackableDataDict]]:
    """Distribute packed sequences for DP sharding.

    Uses snake ordering on bins sorted by token count to balance load across shards.
    E.g., with 4 shards, bins are assigned: 0,1,2,3,3,2,1,0,0,1,2,3,...

    Args:
        bins: List of bins. Each bin is a dictionary where all values are equal-sized lists.
        num_shards: Number of shards to distribute the bins into.

    Returns:
        List of lists (one per shard), where each inner list contains the bins
        assigned to that shard. Each bin is a dictionary with the same keys as the input.
    """
    if len(bins) % num_shards != 0:
        raise ValueError(f"Number of bins ({len(bins)}) must be divisible by num_shards ({num_shards})")

    if not bins:
        return [[] for _ in range(num_shards)]

    # Get keys from first bin
    keys = list(bins[0].keys())

    # Get bin length from the first key (all fields have same length)
    def bin_len(bin_dict: dict[str, list]) -> int:
        return len(bin_dict[keys[0]])

    # Sort bins by token count (descending) for snake distribution
    sorted_bins = sorted(bins, key=bin_len, reverse=True)

    # Snake ordering: assign bins to shards
    shard_bins: list[list[dict[str, list[int | float]]]] = [[] for _ in range(num_shards)]
    for i, bin_dict in enumerate(sorted_bins):
        cycle = i // num_shards
        pos_in_cycle = i % num_shards
        # Reverse direction on odd cycles
        shard_idx = pos_in_cycle if cycle % 2 == 0 else (num_shards - 1 - pos_in_cycle)
        shard_bins[shard_idx].append(bin_dict)

    return shard_bins
