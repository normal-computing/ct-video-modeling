def postprocess(frames):
    frames = (frames.clamp(-1.0, 1.0) + 1.0) / 2.0 * 255
    return frames.moveaxis(2, -1).detach().cpu().numpy()
