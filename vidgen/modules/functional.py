import torch.nn.functional as F
import torch


CONTEXT_ENUM = {
    "forward_context": lambda _: 0,
    "reverse_context": lambda _: -1,
    "midpoint_context": lambda window_size: window_size // 2,
}


def find_neighboring_ts(t, ts):
    if t < ts[0]:
        return False, torch.tensor(0).to(ts.device)

    tmp_feat = 1 / (1 + F.relu(t - ts))
    mat_tmp = torch.max(tmp_feat, dim=0)
    idx = mat_tmp.indices
    if torch.isclose(ts[idx], t):
        return True, idx
    return False, idx + (0 if t < ts[idx] else 1)


def grab_backward_anchor_ts(t, ts, window_size):
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t).to(ts.device).to(ts.dtype)

    _, insert_idx = find_neighboring_ts(t, ts)

    return ts[max(insert_idx - window_size, 0) : insert_idx]


def grab_neighboring_anchor_ts(t, ts, window_size):
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t).to(ts.device).to(ts.dtype)

    is_match, insert_idx = find_neighboring_ts(t, ts)
    zero_idx = torch.tensor(0).to(ts.device)
    max_idx = torch.tensor(ts.size(0) - 1).to(ts.device)

    if insert_idx == 0:
        anchor_idx = [
            max(min(insert_idx + i, max_idx), zero_idx).item()
            for i in range(window_size)
        ]
        anchor_ts = ts[anchor_idx]
        if is_match:
            return "forward_context", anchor_idx
        return "forward_context", torch.cat([t.unsqueeze(0), anchor_ts[:-1]])

    elif insert_idx == ts.size(0):
        anchor_idx = [
            max(min(insert_idx - window_size + i, max_idx), zero_idx).item()
            for i in range(window_size)
        ]
        anchor_ts = ts[anchor_idx]
        if is_match:
            return "reverse_context", anchor_ts
        return "reverse_context", torch.cat([anchor_ts[1:], t.unsqueeze(0)])

    else:
        half_window_size = window_size // 2
        left_anchor_idx = [
            max(min(insert_idx - half_window_size + i, max_idx), zero_idx).item()
            for i in range(half_window_size)
        ]
        if is_match:
            half_window_size += 1

        right_anchor_idx = [
            max(min(insert_idx + i, max_idx), zero_idx).item()
            for i in range(half_window_size)
        ]
        left_anchor_ts, right_anchor_ts = ts[left_anchor_idx], ts[right_anchor_idx]
        if is_match:
            return "midpoint_context", torch.cat([left_anchor_ts, right_anchor_ts])

        return "midpoint_context", torch.cat(
            [left_anchor_ts, t.unsqueeze(0), right_anchor_ts]
        )
