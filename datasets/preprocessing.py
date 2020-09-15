# from time import time

# import ffmpeg
# import numpy as np

# def _make_dataset(root_path: str, class_to_idx: Dict[str, int],
#                   ext: Optional[Tuple[str, ...]]) -> List[Tuple[str, int]]:
#     """
#     `[class명] xxx/yyy/zzz.mp4 or .xml` 폴더 구조에서 `list(영상 경로, 클래스)` 변환
#     """
#     items = []
#     directories = [d for d in os.scandir(root_path) if d.is_dir()]
#     for d in directories:
#         kls = class_to_idx[d.name[1:3]]
#         for root, _, filenames in os.walk(d.path, followlinks=True):
#             for fn in filenames:
#                 if fn.lower().endswith(ext):
#                     items.append((os.path.join(root, fn), kls))
#     return items

# def create_clip(file_path: str,
#                 skip_sec: int = 5,
#                 resolution: str = "768x480") -> np.ndarray:
#     """영상으로 부터 `skip_sec`만큼 건너띄며 클립 생성

#     Args:
#         file_path (str): 비디오 파일 경로

#         skip_sec (int, optional): 건너뛸 초

#         resolution (str, optional): ffmpeg scale 옵션, 1280x720

#     Returns:
#         np.ndarray: [description]
#     """
#     width, height = np.array(resolution.split('x')).astype('int')

#     # xml 파싱
#     event = et.parse(os.path.splitext(file_path)[0] + '.xml').find('event')
#     starttime = event.findtext('starttime')
#     duration = event.findtext('duration')

#     # GPU 가속 및 기타 옵션
#     config = {
#         # "hwaccel_output_format": "cuda",
#         # "c:v": "hevc_cuvid",
#         "ss": starttime,
#         "t": duration,
#     }
#     frames, _ = (ffmpeg.input(file_path, **config).filter(
#         'fps', fps=1 / 5, round='up').filter('scale', size=resolution).output(
#             'pipe:', format='rawvideo',
#             pix_fmt='rgb24').run(capture_stdout=True))
#     frames = (np.frombuffer(frames, np.uint8).reshape([-1, height, width, 3]))
#     return np.einsum('ijkl->lijk', frames)  # C, L, H, W

# def preprocess(root_path: str, output_path: str):
#     classes, class_to_idx = _find_classes(root_path)
#     samples = _make_dataset(root_path, class_to_idx, ('mp4'))

#     clips = []
#     targets = []
#     num_samples = len(samples)
#     cnt = 0
#     starttime = time()
#     for file, target in samples:
#         print(file)
#         targets.append(target)
#         clips.append(create_clip(file))
#         cnt += 1
#         elapsed = time() - starttime
#         avg = elapsed / cnt
#         print(
#             f"{cnt}/{num_samples} complete, {round((cnt / num_samples) * 100,2)}%"
#         )
#         print(
#             f"avg. {round(avg,2)}s, estimation. {round(avg * num_samples / 60,2)}m"
#         )

#     np.savez_compressed(output_path,
#                         clips=np.array(clips),
#                         targets=np.array(targets))

# def save_processed_data(file: str, a: np.ndarray):
#     np.savez_compressed(file, a)

# def load_processed_data(file: str) -> np.ndarray:
#     return np.load(file)
