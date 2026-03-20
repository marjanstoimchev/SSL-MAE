def collate_fn(instances):
    """Custom collate that groups each element position across the batch."""
    batch = []
    for i in range(len(instances[0])):
        batch.append([instance[i] for instance in instances])
    return batch
