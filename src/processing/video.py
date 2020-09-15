import ffmpeg
import numpy as np

# ffmpeg에서 크롭 방법:
# https://video.stackexchange.com/questions/4563/how-can-i-crop-a-video-with-ffmpeg


def create_clip(filename: str,
                starttime: str,
                duration: str,
                fps: float,
                resolution: str = "768x480") -> np.ndarray:
    width, height = np.array(resolution.split('x')).astype('int')

    # GPU 가속 및 기타 옵션
    config = {
        # "hwaccel_output_format": "cuda",
        # "c:v": "hevc_cuvid",
        "ss": starttime,
        "t": duration,
    }
    frames, _ = (ffmpeg.input(filename, **config).filter(
        'fps', fps=fps, round='up').filter('scale', size=resolution).output(
            'pipe:', format='rawvideo',
            pix_fmt='rgb24').run(capture_stdout=True))
    frames = (np.frombuffer(frames, np.uint8).reshape([-1, height, width, 3]))
    return np.einsum('ijkl->lijk', frames)  # C, L, H, W