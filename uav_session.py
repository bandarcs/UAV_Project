"""Replay a local video as a UAV session, or save clips for later analysis."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import hashlib
from itertools import islice
import math
from pathlib import Path
import shutil
import sys

import cv2

from automation import checked_video, open_result, read_frames, run, write_json


ROOT = Path(__file__).resolve().parent
DEMO = ROOT / 'datasets' / 'Halmstad' / 'reference_samples' / 'V_AIRPLANE_001.mp4'
DATASET_COMMIT = '442d0c6750708cc221636f7a1fd31cbd8cb0aeb8'
DATASET = 'https://github.com/DroneDetectionThesis/Drone-detection-dataset'
CLIP_SOURCE = f'{DATASET}/blob/{DATASET_COMMIT}/Data/Video_V'
DEMO_URL = f'{CLIP_SOURCE}/{DEMO.name}'
PAPER = 'https://doi.org/10.1016/j.dib.2021.107521'
REFERENCE_CLIPS = {
    'V_AIRPLANE_001.mp4': (213050, '6bb61e33c12f48a19dcfb77293ac000626c18a18326957ead37101d31c91a143'),
    'V_BIRD_001.mp4': (331120, '062059c958360e552fa475f73ad32431441abc3970939e6a12f464a76d1095bf'),
    'V_DRONE_001.mp4': (144518, '9db5a800377c01db8369dfba808d400ca52cd393e0a32850e1d20a4861eaecbe'),
    'V_HELICOPTER_001.mp4': (351761, '027e197f3c7872b495163bd30eb39c7b8fef4c4aef15cee01f170da46825046c'),
}


@contextmanager
def video_capture(path: Path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f'Could not open video: {path}')
        yield capture
    finally:
        capture.release()


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frames: int
    width: int
    height: int

    @classmethod
    def inspect(cls, path: Path):
        with video_capture(path) as capture:
            fps = capture.get(cv2.CAP_PROP_FPS)
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or min(frames, width, height) < 1:
            raise ValueError('The video must report its frame rate, frame count and dimensions.')
        return cls(fps, frames, width, height)

    def seconds(self, frame: int) -> float:
        return round(frame / self.fps, 6)


def save_clip(destination: Path, frames, info: VideoInfo) -> int:
    """Write a clip to a temporary file and publish it when the writer closes."""
    temporary = destination.with_name(f'{destination.stem}.partial.mp4')
    size = (info.width + info.width % 2, info.height + info.height % 2)
    try:
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'mp4v'), info.fps, size)
        count = 0
        try:
            if not writer.isOpened():
                raise OSError(f'Could not create a clip in {destination.parent}')
            for frame in frames:
                if frame is None or frame.shape[:2] != (info.height, info.width):
                    raise ValueError('The source contains a missing frame or a change in video dimensions.')
                if size != (info.width, info.height):
                    frame = cv2.copyMakeBorder(frame, 0, size[1] - info.height, 0, size[0] - info.width, cv2.BORDER_REPLICATE)
                writer.write(frame)
                count += 1
        finally:
            writer.release()
        if count == 0:
            raise ValueError(f'No frames were available to record: {destination.name}')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def source_reference(source: Path):
    # A downloaded file keeps its citation when moved to another computer.
    expected = REFERENCE_CLIPS.get(source.name)
    if expected is None or source.stat().st_size != expected[0]:
        return None
    sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if sha256 != expected[1]:
        return None
    return {'dataset': DATASET, 'clip': f'{CLIP_SOURCE}/{source.name}', 'paper': PAPER, 'sha256': sha256}


def outage_frames(info: VideoInfo, interval):
    if interval is None:
        first, end = info.frames // 3, info.frames * 2 // 3
    else:
        if any(not math.isfinite(seconds) or seconds <= 0 for seconds in interval):
            raise ValueError('Outage times must be positive, finite seconds.')
        first, end = (math.ceil(seconds * info.fps) for seconds in interval)
    if not 0 < first < end < info.frames:
        raise ValueError('Choose an outage that begins and ends within the video, with live frames on each side.')
    return first, end


def replay_live(capture, info: VideoInfo, outage, output: Path, event):
    """Release frames in source order, withholding the gap until reconnection."""
    first, end = outage
    frames = read_frames(capture)
    event('connected', 0)
    yield from islice(frames, first)

    event('link_lost', first)
    onboard = output / 'onboard' / 'outage.mp4'
    onboard.parent.mkdir()
    count = save_clip(onboard, islice(frames, end - first), info)
    if count != end - first:
        raise ValueError('Source ended before the simulated link could be restored. The partial onboard recording was saved.')

    event('link_restored', end)
    recovered = output / 'recovered' / onboard.name
    recovered.parent.mkdir()
    staged = recovered.with_name('outage.partial.mp4')
    shutil.copyfile(onboard, staged)
    staged.replace(recovered)
    event('recording_transferred', end, file=recovered.relative_to(output).as_posix(), frames=count,
          recording_from=info.seconds(first), recording_until=info.seconds(end))
    decoded = 0
    with video_capture(recovered) as transferred:
        for frame in read_frames(transferred):
            if decoded == count:
                raise ValueError('The recovered clip contains more frames than the onboard recording.')
            yield frame[:info.height, :info.width]
            decoded += 1
    if decoded != count:
        raise ValueError('The recovered clip is incomplete.')
    event('live_analysis_resumed', end)
    yield from frames


def record_periodic(source: Path, output: Path, info: VideoInfo, seconds: float, event):
    per_clip = max(1, round(seconds * info.fps))
    folder = output / 'clips'
    folder.mkdir()
    start = 0
    with video_capture(source) as capture:
        frames = read_frames(capture)
        for number, first_frame in enumerate(frames, start=1):
            clip = folder / f'clip_{number:03}.mp4'

            def clip_frames():
                yield first_frame
                yield from islice(frames, per_clip - 1)

            count = save_clip(clip, clip_frames(), info)
            event('clip_saved', start, file=clip.relative_to(output).as_posix(), frames=count,
                  end_seconds=info.seconds(start + count))
            start += count
    if start != info.frames:
        raise ValueError(f'The source reported {info.frames} frames and yielded {start}. The saved clips were retained.')


def execute(args):
    source = checked_video(args.video or DEMO)
    info = VideoInfo.inspect(source)
    outage = outage_frames(info, args.outage) if args.mode == 'live' else None
    if args.mode == 'periodic' and (not math.isfinite(args.clip_seconds) or args.clip_seconds <= 0):
        raise ValueError('Clip length must be a positive, finite number of seconds.')
    output = args.output_dir or ROOT / 'runs' / 'uav_sessions' / f'{args.mode}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'mode': args.mode, 'status': 'running', 'input_video': str(source),
                'source_duration_seconds': info.seconds(info.frames),
                'source_reference': source_reference(source), 'events': []}

    def save_manifest():
        temporary = output / 'session.partial.json'
        write_json(temporary, manifest)
        temporary.replace(output / 'session.json')

    def event(name, frame, **details):
        manifest['events'].append({'event': name, 'source_frame': frame, 'source_seconds': info.seconds(frame), **details})
        save_manifest()
        print(f'[{info.seconds(frame):.2f}s source time] {name.replace("_", " ")}', flush=True)

    save_manifest()
    summary = None
    try:
        if args.mode == 'periodic':
            record_periodic(source, output, info, args.clip_seconds, event)
        elif outage is not None:
            def frame_source(capture):
                return replay_live(capture, info, outage, output, event)

            summary = run(source, output / 'analysis', ROOT, frame_source=frame_source,
                          calibration_end=outage[0], rate_limited=True)
        else:
            summary = run(source, output / 'analysis', ROOT, rate_limited=args.mode == 'live')
        manifest['status'] = summary['status'] if summary else 'completed'
    except (Exception, KeyboardInterrupt) as error:
        manifest['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        manifest['error'] = str(error) or type(error).__name__
        save_manifest()
        raise
    event('session_finished', summary['decoded_frames'] if summary else info.frames)

    lines = ['UAV video session', f'Mode: {args.mode}', f'Source: {source.name}',
             f'Video duration: {manifest["source_duration_seconds"]:.2f} seconds', f'Status: {manifest["status"]}']
    if summary:
        lines.extend([f'Model: {Path(summary["selection"]["weights"]).name}',
                      f'Analysed frames: {summary["analysed_frames"]}', f'Detection records: {summary["detection_rows"]}',
                      f'Processing time: {summary["processing_seconds"]:.2f} seconds',
                      'Supported classes: ' + ', '.join(summary['supported_classes'])])
    if manifest['source_reference']:
        reference = manifest['source_reference']
        lines.extend(['', f'Source clip: {reference["clip"]}', f'Dataset paper: {reference["paper"]}'])
    (output / 'print_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\nFiles saved to: {output}', flush=True)
    if summary and summary['written_frames'] and not args.no_open:
        open_result(output / 'analysis' / 'annotated.mp4')
    return output


def menu_arguments():
    print('\nUAV Project\n1. Replay live monitoring and link loss\n2. Record periodic clips\n3. Analyse a saved clip')
    choice = input('Choose 1, 2 or 3: ').strip()
    options = {'1': 'live', '2': 'periodic', '3': 'analyse'}
    if choice not in options:
        raise ValueError('Choose 1, 2 or 3.')
    video = input('Enter a video path or press Enter for the Halmstad sample: ').strip().strip('"')
    return [options[choice], *([video] if video else [])]


def main(argv=None):
    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('live', 'periodic', 'analyse'))
    parser.add_argument('video', nargs='?', help='Local recording. Defaults to the Halmstad airplane sample.')
    parser.add_argument('--outage', nargs=2, type=float, metavar=('LOSS_SECONDS', 'RETURN_SECONDS'))
    parser.add_argument('--clip-seconds', type=float, default=3.0, help='Clip length in periodic mode.')
    parser.add_argument('--output-dir', type=Path, help='A new results folder.')
    parser.add_argument('--no-open', action='store_true', help='Save results without opening the video player.')
    try:
        args = parser.parse_args(menu_arguments() if args_list == ['--menu'] else args_list)
        if args.video is None and not DEMO.is_file():
            raise ValueError(f'Download the sample to {DEMO}, or provide another video.\n{DEMO_URL}')
        execute(args)
    except (cv2.error, EOFError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f'\n{error}\n')
    except KeyboardInterrupt:
        parser.exit(130, '\nSession stopped.\n')


if __name__ == '__main__':
    main()
