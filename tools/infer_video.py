import argparse

import cv2
import numpy as np
import torch
import time
from vedacore.image import imread, imwrite
from vedacore.misc import Config, color_val, load_weights
from vedacore.parallel import collate, scatter
from vedadet.datasets.pipelines import Compose
from vedadet.engines import build_engine
from tqdm import tqdm
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Infer a detector')
    parser.add_argument('config', help='config file path')
    parser.add_argument("-i", "--input-path", help="Input video path.", required=True)
    parser.add_argument("-o", "--output-path", help="Output video path.", required=True)
    parser.add_argument("--size", help="Image size (default=%(default)s).", type=float, default=None)
    parser.add_argument("--fps", help="Video output FPS (default=%(default)s).", type=float, default=None)
    parser.add_argument("--run-per-x-frames", help="Run per X frames (default=%(default)s).", type=float, default=1)
    parser.add_argument("--frame-count", help="Number of video frames to process (including skipped ones) (default=%(default)s).", type=float, default=None)
    args = parser.parse_args()
    return args


def prepare(cfg):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'
    engine = build_engine(cfg.infer_engine)

    engine.model.to(device)
    load_weights(engine.model, cfg.weights.filepath)

    data_pipeline = Compose(cfg.data_pipeline)
    return engine, data_pipeline, device


def draw_box(result, img, class_names):
    font_scale = 1
    bbox_color = 'green'
    text_color = 'green'
    thickness = 3

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)

    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], idx, dtype=np.int32)
        for idx, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    return img
def save_output(video_path, output_path, engine, pipeline, class_names, device, size=None, roi=(), 
frame_count=None, run_per_x_frames=None, fps=None):
    """
        Run detector on the video and save the output as another video.
        Args:
            video_path: Path to the source video.
            output_path: Path to the output video.
            engine: Model for detecting Face
            size: Image size (longer side) for the detector.
            roi: A region of interest in a format of tuple of (xmin, ymin, xmax, ymax).
            frame_count: Number of frames to detect (including skipped frames).
            run_per_x_frames: Run an inference every X frames.
            fps: Output video FPS.
        Returns:
            A performance dict.
    """
    cap = cv2.VideoCapture(video_path)
    fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = fps or cap.get(cv2.CAP_PROP_FPS) / run_per_x_frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if roi:
        xmin, ymin, xmax, ymax = roi
        frame_width, frame_height = xmax - xmin, ymax - ymin
    else:
        xmin = 0
        ymin = 0
        xmax = frame_width
        ymax = frame_height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    fc = frame_count or fc
    frame_count = fc
    bar = tqdm(total=int(frame_count))
    timer = defaultdict(float)
    print(f"{time.ctime()}: Start.")
    total_start = time.time()
    while True:
        start = time.time()
        ret = cap.grab()
        frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if not ret or frame_idx > frame_count:
            break
        if int(frame_idx) % run_per_x_frames != 0:
            bar.update()
            continue
        _, frame = cap.retrieve()
        cv2.imwrite('./temp.jpg', frame)
        data = dict(img_info=dict(filename='./temp.jpg'), img_prefix=None)

        data = pipeline(data)
        data = collate([data], samples_per_gpu=1)
        data = scatter(data, [device])[0]
        timer["decode"] += time.time() - start
        
        start = time.time()
        frame = frame[ymin:ymax, xmin:xmax]
        timer["crop"] += time.time() - start
        
        start = time.time()
        result = engine.infer(data['img'], data['img_metas'])[0]
        timer["detect"] += time.time() - start
        
        start = time.time()
        frame = draw_box(result, frame, class_names)
        timer["draw"] += time.time() - start
        
        start = time.time()
        out.write(frame)
        timer["write"] += time.time() - start
        
        timer["count"] += 1
        bar.update()
    bar.close()
    cap.release()
    out.release()
    timer["total (incl. all)"] = time.time() - total_start
    print(f"{time.ctime()}: Finished. ({time.time()-total_start:.4f} seconds.)")
    return timer

def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)
    class_names = cfg.class_names

    engine, data_pipeline, device = prepare(cfg)
    perf = save_output(
        args.input_path, 
        args.output_path, 
        engine, 
        data_pipeline,
        class_names,
        device,
        frame_count=args.frame_count, 
        run_per_x_frames=args.run_per_x_frames, 
        fps=args.fps
    )
    print("\n" + "=" * 25, "Performance", "=" * 25 + "\n")
    count = perf["count"]
    for k, v in perf.items():
        if k == "count":
            continue
        print(f"{k.upper():<30}: {v:.4f} seconds ({count / v:.4f} FPS)")
    print("\nNOTE: If the times do not add up to `TOTAL (INCL. ALL)`, it it because the total includes time wasted on grabbing frames that were skipped.")

if __name__ == '__main__':
    main()
