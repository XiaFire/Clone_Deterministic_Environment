import torch


def collate_fn_maze(batch):
    batch.sort(key=lambda x: x["pose"].shape[0], reverse=True)

    poses = [item["pose"] for item in batch]
    actions = [item["action"] for item in batch]
    has_image = all("image" in item for item in batch)
    if has_image:
        images = [item["image"] for item in batch]

    seq_lengths = torch.tensor([x.shape[0] for x in poses])

    padded_poses = torch.nn.utils.rnn.pad_sequence(
        poses, batch_first=True, padding_value=0.0
    )
    padded_actions = torch.nn.utils.rnn.pad_sequence(
        actions, batch_first=True, padding_value=-100.0
    )
    if has_image:
        padded_images = torch.nn.utils.rnn.pad_sequence(
            images, batch_first=True, padding_value=0.0
        )

    max_len_in_batch = padded_poses.shape[1]
    padding_mask = torch.arange(max_len_in_batch).expand(len(seq_lengths), max_len_in_batch) < seq_lengths.unsqueeze(1)

    return {
        "pose": padded_poses,
        "action": padded_actions,
        **({"image": padded_images} if has_image else {}),
        "padding_mask": padding_mask,
        "seq_lengths": seq_lengths,
    }


def collate_fn_precomputed(batch):
    elem = batch[0]
    collated_batch = {}
    if batch[0]["latents"].ndim == 3:
        max_len = max([d["latents"].shape[1] for d in batch])
    else:
        max_len = max([d["latents"].shape[0] for d in batch])

    for key in elem:
        if not all(key in d for d in batch):
            continue

        if isinstance(elem[key], torch.Tensor):
            if elem[key].dim() == 0:
                collated_batch[key] = torch.stack([d[key] for d in batch])
                continue

            padded_tensors = []
            for d in batch:
                tensor = d[key]
                pad_len = max_len - tensor.shape[0]
                if pad_len > 0:
                    pad_shape = list(tensor.shape)
                    pad_shape[0] = pad_len
                    padded_tensor = torch.cat(
                        [tensor, torch.zeros(pad_shape, dtype=tensor.dtype)], dim=0
                    )
                    padded_tensors.append(padded_tensor)
                else:
                    padded_tensors.append(tensor)
            collated_batch[key] = torch.stack(padded_tensors)
    return collated_batch
