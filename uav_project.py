"""Analyse a saved video, replay a simulated outage, or split a video into clips."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from itertools import chain, islice
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent
DEMO = ROOT / 'datasets' / 'Halmstad' / 'reference_samples' / 'V_AIRPLANE_001.mp4'  # تعطي هذه العينة الفريق مقطعًا جاهزًا للتجربة عند ترك مسار الفيديو فارغًا.
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


TARGET_ALIASES = {  # تربط هذه القائمة أسماء النماذج المختلفة باسم موحّد. مثال ذلك تخزين aeroplane باسم airplane في النتائج.
    "airplane": {"airplane", "aeroplane", "aircraft", "plane"},
    "helicopter": {"helicopter", "chopper"},
    "drone": {"drone", "uav", "quadcopter", "quadrotor"},
    "bird": {"bird"},
}
POLICY = {
    "version": "validation-aware-policy-v4",
    "max_analysis_fps": 30,
    "confidence": 0.25,
    "sample_fractions": [0.1, 0.5, 0.9],  # توزيع العيّنات على أجزاء مختلفة يساعد على تقدير وضوح المقطع حتى لو تغيرت الإضاءة أثناء التصوير.
    "sizes": [640, 960, 1280],
    "timing_margin": 1.25,
    "schedule_window": 20,
    "output_long_edge": 1280,
}


def menu_arguments():
    print('\nUAV Project\n1. Simulate signal loss\n2. Split a video\n3. Analyse a video automatically')
    choice = input('Choose 1, 2 or 3: ').strip()  # يحدد هذا الرقم المهمة المطلوبة. الخيار 3 هو كشف الأجسام وتتبعها في فيديو محفوظ.
    options = {'1': 'live', '2': 'periodic', '3': 'analyse'}
    if choice not in options:
        raise ValueError('Choose 1, 2 or 3.')
    video = input('Enter a video path or press Enter for the Halmstad sample: ').strip().strip('"')
    arguments = [options[choice]]
    if video:
        arguments.append(video)  # يصل المسار الذي يكتبه المستخدم إلى نفس المحلل الذي يستقبل ملفات الفيديو من سطر الأوامر.
    return arguments


def main(argv=None):  # نقطة البداية التي تقرر استخدام القائمة أو قراءة خيارات التشغيل من سطر الأوامر.
    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode_or_video', metavar='MODE_OR_VIDEO', nargs='?',
                        help='live, periodic, analyse, or a path to a local video')
    parser.add_argument('video', nargs='?', help='Local video for a mode. Defaults to the Halmstad sample.')
    parser.add_argument('--menu', action='store_true', help='Show the interactive menu.')
    parser.add_argument('--outage', nargs=2, type=float, metavar=('LOSS_SECONDS', 'RETURN_SECONDS'))
    parser.add_argument('--clip-seconds', type=float, default=3.0, help='Clip length in periodic mode.')
    parser.add_argument('--output-dir', type=Path, help='A new results folder.')
    parser.add_argument('--no-open', action='store_true', help='Save results without opening the video player.')
    try:
        args = parser.parse_args(menu_arguments() if not args_list or args_list == ['--menu'] else args_list)
        if args.menu or args.mode_or_video is None:
            parser.error('Use --menu on its own, or specify a mode or a video path.')
        if args.mode_or_video not in ('live', 'periodic', 'analyse'):
            if args.video is not None or args.outage is not None or args.clip_seconds != 3.0:
                parser.error('Use a named mode when passing session options.')
            analyse_direct(args.mode_or_video, args.output_dir, args.no_open)
            return
        args.mode = args.mode_or_video
        if args.video is None and not DEMO.is_file():
            raise ValueError(f'Download the sample to {DEMO}, or provide another video.\n{DEMO_URL}')
        execute(args)  # بعد قراءة الخيارات، تمر مهمة القائمة ومهمة سطر الأوامر على نفس طريقة تشغيل الجلسة وحفظها.
    except (cv2.error, EOFError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f'\n{error}\n')
    except KeyboardInterrupt:
        parser.exit(130, '\nSession stopped.\n')


def execute(args):  # يدير جلسة واحدة من اختيار الفيديو حتى حفظ السجل والمخرجات.
    source = checked_video(args.video or DEMO)  # يفحص المسار والامتداد مبكرًا حتى تظهر رسالة مفهومة قبل بدء تجربة النماذج.
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
        temporary.replace(output / 'session.json')  # استبدال السجل بعد اكتمال كتابة النسخة المؤقتة يقلل احتمال ترك ملف JSON ناقص عند التوقف.

    def event(name, frame, **details):
        manifest['events'].append({'event': name, 'source_frame': frame, 'source_seconds': info.seconds(frame), **details})
        save_manifest()
        print(f'[{info.seconds(frame):.2f}s source time] {name.replace("_", " ")}', flush=True)

    save_manifest()
    summary = None
    try:
        if args.mode == 'periodic':  # هذا الخيار يحفظ أجزاء قصيرة من الفيديو لتجربتها لاحقًا كل مقطع على حدة.
            record_periodic(source, output, info, args.clip_seconds, event)
        elif outage is not None:
            def frame_source(capture):  # مصدر الإطارات هنا هو سيناريو المحاكاة. يعيد الجزء المحفوظ في موضعه قبل متابعة بقية الفيديو.
                return replay_live(capture, info, outage, output, event)

            summary = run(
                source, output / 'analysis', ROOT, frame_source=frame_source,
                calibration_end=outage[0], rate_limited=True,
            )
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


def analyse_direct(video_path, output_dir, no_open):
    video = checked_video(video_path)  # تمرير مسار وحده يتيح بدء التحليل مباشرة من نافذة الأوامر.
    output = output_dir or ROOT / 'runs' / 'automation' / f'{video.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    result = run(video, output, ROOT)
    print(json.dumps({  # الملخص يسهّل على الفريق قراءة النموذج المختار ومسار النتائج أو التقاطهما من برنامج آخر.
        'status': result['status'],
        'weights': result['selection']['weights'],
        'imgsz': result['selection']['imgsz'],
        'analysed_frames': result['analysed_frames'],
        'unsupported_classes': result['unsupported_classes'],
        'output': str(output),
    }, indent=2))
    print(f'Annotated video: {output / "annotated.mp4"}')
    print(f'Detection table: {output / "detections.csv"}')
    if not no_open:
        open_result(output / 'annotated.mp4')


def read_frames(capture):
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        yield frame  # نسلّم كل صورة فور قراءتها حتى تبقى ذاكرة الجهاز متاحة حتى مع المقاطع الطويلة.


def run(video: Path, output: Path, project_root: Path, *, frame_source=read_frames,
        calibration_end=None, rate_limited=False):
    import torch
    from ultralytics import YOLO

    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "initialising", "source": str(video), "policy": POLICY,
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "purpose": "Offline academic recorded-video experiment",
              "accuracy_evaluated": False,
              "frame_policy": "adaptive_replay" if rate_limited else "all_recorded_frames"}
    status_path = output / "summary.json"
    write_json(status_path, status)
    capture = writer = frame_iterator = None
    counts = {"decoded_frames": 0, "analysed_frames": 0, "skipped_frames": 0,
              "written_frames": 0, "detection_rows": 0}
    try:
        with video_capture(video) as sample_capture:
            info = VideoInfo.from_capture(sample_capture)  # تُستخدم خصائص المصدر لتحديد الفاصل الزمني للإطارات ولتشغيل الناتج بسرعة الفيديو الأصلية.
            width, height, fps, total = info.width, info.height, info.fps, info.frames
            if min(width, height) < 2:
                raise ValueError("Video width and height must each be at least two pixels.")
            samples, positions, quality = sample_video(
                sample_capture, total, calibration_end
            )  # تقيّم العيّنات الإضاءة والتباين والوضوح قبل اختيار إعدادات التحليل للمقطع.
        status.update(calibration_sample_frames=positions, image_quality=quality,
                      source_properties={"width": width, "height": height, "fps": fps,
                                         "reported_frames": total})

        device = 0 if torch.cuda.is_available() else "cpu"  # يستخدم كرت الشاشة على جهاز التشغيل عند توفر CUDA، وإلا ينفذ التحليل على المعالج.
        status["device"] = str(device)  # يساعد سجل الجهاز على فهم سبب اختلاف السرعة بين جهاز الفريق والسيرفر.
        long_edge = max(width, height)
        profiles = calibrate_models(
            project_root, output, samples, long_edge, device
        )  # نقارن زمن معالجة نفس العيّنات لكل نموذج وحجم مرشح حتى يكون قراره مبنيًا على سرعة الجهاز الحالي.
        difficult = quality["low_visibility_heuristic"]
        model_settings, frame_step, reason = select_profile(
            profiles, fps, difficult, recorded=not rate_limited
        )  # يحدد البرنامج ملف النموذج وحجم الصورة، ويختار في المحاكاة كم صورة يمر بينها قبل التحليل التالي.
        model = YOLO(model_settings["weights"])  # يُحمَّل الملف الذي نجح في الاختيار ليستخدمه المتتبّع في بقية الجلسة.
        mapping = class_mapping(model.names)
        missing = sorted(set(TARGET_ALIASES) - set(mapping.values()))
        status.update(selection=model_settings, selection_reason=reason, supported_classes=sorted(set(mapping.values())),
                      unsupported_classes=missing, initial_stride=frame_step,
                      class_coverage_complete=not missing,
                      calibration_seconds=time.perf_counter() - started,
                      note="No accuracy or superiority claim. Unsupported classes cannot be recognised with these weights.")
        write_json(output / "decision.json", status)  # سجل القرار يخلّي الفريق يعرف أي إعداد فاز ولماذا، ويبين أسماء الفئات التي يدعمها النموذج.
        print(f"Selected {Path(model_settings['weights']).name}, size {model_settings['imgsz']}, stride {frame_step}", flush=True)
        class_warning = ""
        if missing:
            print("Weights do not support: " + ", ".join(missing), flush=True)
            class_warning = "Limited classes: " + ", ".join(status["supported_classes"])
        capture = cv2.VideoCapture(str(video))  # فحص العيّنات يتنقل بين مواقع مختلفة، لذلك نبدأ قراءة جديدة حتى تدخل أول صورة في فيديو النتائج.
        if not capture.isOpened():
            raise ValueError("Could not reopen source after calibration.")
        scale = min(1.0, POLICY["output_long_edge"] / max(width, height))
        output_size = tuple(max(2, int(dim * scale) // 2 * 2) for dim in (width, height))
        writer = cv2.VideoWriter(str(output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
        if not writer.isOpened():
            raise ValueError("Could not create output video.")
        recent_times = []
        frame_schedule = RateController(fps, frame_step)
        settings = {
            "persist": True, "tracker": "bytetrack.yaml",  # يحتفظ ByteTrack بحالة التتبّع بين الصور حتى يحاول إعطاء الجسم نفسه الرقم نفسه عبر اللقطات.
            "imgsz": model_settings["imgsz"], "conf": POLICY["confidence"],
            "classes": sorted(mapping), "device": device, "verbose": False,
        }
        processing_started = time.perf_counter()
        with (
            (output / "detections.csv").open("w", newline="", encoding="utf-8") as detections,
            (output / "frames.csv").open("w", newline="", encoding="utf-8") as frames,
            (output / "decisions.csv").open("w", newline="", encoding="utf-8") as decisions,
        ):
            detection_table = csv.writer(detections)
            frame_table = csv.writer(frames)
            schedule_table = csv.writer(decisions)
            detection_table.writerow(["frame", "time_seconds", "class", "confidence", "track_id", "x1", "y1", "x2", "y2"])
            frame_table.writerow(["frame", "analysed", "stride", "detections", "analysis_ms"])
            schedule_table.writerow(["effective_frame", "stride", "reason"])
            schedule_table.writerow([0, frame_step, "startup calibration" if rate_limited else "all recorded frames"])
            frame_iterator = iter(frame_source(capture))  # مصدر الإطارات هو الملف المسجّل في التحليل العادي، أو تسلسل الإطارات المسترجعة في المحاكاة.
            for index, frame in enumerate(frame_iterator):
                if frame is None or not frame.size:
                    raise ValueError("An empty decoded frame was encountered.")
                counts["decoded_frames"] += 1
                step = frame_schedule.stride
                analysed = index % step == 0  # القيمة 1 تعني تحليل كل صورة. مثال المحاكاة مع القيمة 2 هو تحليل صورة من كل صورتين.
                count, analysis_ms = 0, None
                legend = []
                if analysed:
                    sync_device(device)
                    tick = time.perf_counter()
                    result = model.track(frame, **settings)[0]  # يحدد النموذج مواقع الأجسام في الصورة الحالية ثم يربطها ByteTrack بالأرقام التي يتتبعها عبر الفيديو.
                    legend = write_detections(result, mapping, detection_table, index, fps)
                    count = len(legend)
                    display = result.plot(labels=False, line_width=2)  # ترسم هذه الخطوة حدود كل جسم، وتحفظ مساحة الصورة بوضع الأسماء والأرقام في قائمة جانبية.
                    sync_device(device)
                    analysis_ms = (time.perf_counter() - tick) * 1000
                    if rate_limited:
                        recent_times.append(analysis_ms)
                    counts["analysed_frames"] += 1
                    counts["detection_rows"] += count
                    del result
                else:
                    display = frame  # الإطار المتجاوز يبقى في مكانه في الناتج ليحافظ الفيديو على مدته الأصلية.
                    counts["skipped_frames"] += 1
                writer.write(annotate_frame(display, output_size, legend, analysed, class_warning))  # تُضاف الصورة إلى الفيديو الناتج، ويظهر تنبيه على الصور التي تجاوزتها المعالجة في المحاكاة.
                counts["written_frames"] += 1
                frame_table.writerow([index, int(analysed), step, count if analysed else "",
                                      analysis_ms if analysed else ""])
                if len(recent_times) == POLICY["schedule_window"]:
                    new_step = frame_schedule.update(statistics.median(recent_times))
                    recent_times.clear()
                    if new_step is not None:
                        schedule_table.writerow([index + 1, new_step, "rolling measured analysis latency"])
                if counts["decoded_frames"] % 100 == 0:
                    print(f"Decoded {counts['decoded_frames']}/{total} frames", flush=True)
        if counts["written_frames"] == 0:
            raise ValueError("No video frames were written. Check that the input video can be decoded.")
        elapsed = time.perf_counter() - processing_started
        status.update(status="completed" if counts["decoded_frames"] >= total else "completed_with_decode_warning", **counts, processing_seconds=elapsed,
                      end_to_end_processing_fps=counts["decoded_frames"] / elapsed,
                      analysed_frames_per_second=counts["analysed_frames"] / elapsed,
                      decode_ended_before_reported_count=counts["decoded_frames"] < total,
                      output_video_size=list(output_size), final_stride=frame_schedule.stride,
                      total_seconds=time.perf_counter() - started)
        write_json(status_path, status)  # الملخص يجمع عدد الصور المقروءة والمحللة وزمن التنفيذ ليتحقق الفريق من اكتمال التجربة.
        return status
    except BaseException as exc:
        status.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=f"{type(exc).__name__}: {exc}", **counts)
        write_json(status_path, status)
        raise
    finally:
        close_frames = getattr(frame_iterator, "close", None)
        if close_frames is not None:
            close_frames()
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()


@contextmanager
def video_capture(path: Path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f'Could not open video: {path}')
        yield capture
    finally:
        capture.release()  # تحرير قارئ الفيديو يمنع بقاء الملف محجوزًا، ويحدث أيضًا عند انتهاء القراءة بخطأ.


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frames: int
    width: int
    height: int

    @classmethod
    def inspect(cls, path: Path):
        with video_capture(path) as capture:
            return cls.from_capture(capture)

    @classmethod
    def from_capture(cls, capture):
        fps = capture.get(cv2.CAP_PROP_FPS)  # معدل الصور في الثانية يحوّل رقم كل إطار إلى وقت يمكن العثور عليه داخل المقطع.
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not all(math.isfinite(value) and value > 0 for value in (fps, frames, width, height)):
            raise ValueError('The video must report its frame rate, frame count and dimensions.')
        if min(int(frames), int(width), int(height)) < 1:
            raise ValueError('The video must contain at least one frame with valid dimensions.')
        return cls(fps, int(frames), int(width), int(height))

    def seconds(self, frame: int) -> float:
        return round(frame / self.fps, 6)


def save_clip(destination: Path, frames, info: VideoInfo) -> int:
    temporary = destination.with_name(f'{destination.stem}.partial.mp4')  # يُكتب المقطع باسم مؤقت، ثم يأخذ الاسم النهائي بعد اكتمال الكتابة.
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
    expected = REFERENCE_CLIPS.get(source.name)
    if expected is None or source.stat().st_size != expected[0]:
        return None
    sha256 = hashlib.sha256(source.read_bytes()).hexdigest()  # بصمة الملف تتأكد أن المقطع يطابق نسخة الباحثين قبل إرفاق رابط المصدر في الملخص.
    if sha256 != expected[1]:
        return None
    return {'dataset': DATASET, 'clip': f'{CLIP_SOURCE}/{source.name}', 'paper': PAPER, 'sha256': sha256}


def outage_frames(info: VideoInfo, interval):
    if interval is None:  # في التجربة الافتراضية نختار الثلث الأوسط حتى تحتوي العينة على فترة قبل الانقطاع وفترة بعده.
        first, end = info.frames // 3, info.frames * 2 // 3
    else:
        if any(not math.isfinite(seconds) or seconds <= 0 for seconds in interval):
            raise ValueError('Outage times must be positive, finite seconds.')
        first, end = (math.ceil(seconds * info.fps) for seconds in interval)  # يضرب البرنامج وقت الانقطاع في معدل الإطارات ليحدد الصور التي يبدأ عندها التسجيل وينتهي.
    if not 0 < first < end < info.frames:
        raise ValueError('Choose an outage that begins and ends within the video, with live frames on each side.')
    return first, end


def replay_live(capture, info: VideoInfo, outage, output: Path, event):
    first, end = outage
    frames = read_frames(capture)
    event('connected', 0)
    yield from islice(frames, first)

    event('link_lost', first)
    onboard = output / 'onboard' / 'outage.mp4'  # الملف المحلي يمثل ما كانت ستسجله الكاميرا خلال الفترة التي انقطع فيها البث في المحاكاة.
    onboard.parent.mkdir()
    count = save_clip(onboard, islice(frames, end - first), info)
    if count != end - first:
        raise ValueError('Source ended before the simulated link could be restored. The partial onboard recording was saved.')

    event('link_restored', end)
    recovered = output / 'recovered' / onboard.name  # تسبق الصور المسترجعة الصور التالية للانقطاع، حتى يحافظ فيديو النتيجة على الترتيب الأصلي.
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
    per_clip = max(1, round(seconds * info.fps))  # يتحول طول المقطع بالثواني إلى عدد صور لكل ملف، ويحفظ الملف الأخير ما تبقّى.
    folder = output / 'clips'
    folder.mkdir()
    start = 0
    with video_capture(source) as capture:
        frames = read_frames(capture)
        for number, first_frame in enumerate(frames, start=1):
            clip = folder / f'clip_{number:03}.mp4'

            clip_frames = chain((first_frame,), islice(frames, per_clip - 1))  # تُضاف الصورة الأولى إلى بقية صور المقطع تدريجيًا، فيُكتب كل جزء أثناء القراءة.
            count = save_clip(clip, clip_frames, info)
            event('clip_saved', start, file=clip.relative_to(output).as_posix(), frames=count,
                  end_seconds=info.seconds(start + count))
            start += count
    if start != info.frames:
        raise ValueError(f'The source reported {info.frames} frames and yielded {start}. The saved clips were retained.')


def frame_stride(source_fps, latency_ms):
    rate_limit = math.ceil(source_fps / POLICY["max_analysis_fps"])
    time_limit = math.ceil(source_fps * latency_ms * POLICY["timing_margin"] / 1000)
    return max(1, rate_limit, time_limit)  # الفاصل في المحاكاة يحترم حد معدل الصور والزمن المقاس للتحليل، ويحدد عدد الصور بين تحليلين.


@dataclass
class RateController:
    source_fps: float
    stride: int
    lower_windows: int = 0

    def update(self, median_ms):
        suggested = frame_stride(self.source_fps, median_ms)
        if suggested > self.stride:
            self.stride, self.lower_windows = suggested, 0  # زيادة الفاصل تقلل عدد الصور المطلوب تحليلها فور ارتفاع زمن المعالجة.
            return self.stride
        if suggested < self.stride:
            self.lower_windows += 1
            if self.lower_windows >= 2:  # يتطلب الرجوع إلى تحليل صور أكثر قياسين أسرع متتاليين حتى يستقر الحمل.
                self.stride, self.lower_windows = suggested, 0
                return self.stride
        else:
            self.lower_windows = 0
        return None


def class_mapping(names):
    pairs = names.items() if isinstance(names, dict) else enumerate(names)
    mapping = {}
    for index, name in pairs:
        normal = str(name).strip().lower().replace("-", "").replace("_", "")
        for canonical, aliases in TARGET_ALIASES.items():
            if normal in aliases:
                mapping[int(index)] = canonical  # يظل رقم الفئة كما يخرجه النموذج، ويستخدم الجدول الاسم المتفق عليه داخل المشروع.
                break
    return mapping


def select_profile(profiles, source_fps, difficult, *, recorded=False):  # يختار من نتائج التجارب الناجحة وفق الفئات المتاحة وزمن المعالجة والتقييمات المحفوظة.
    good = [p for p in profiles if p.get("ok") and p["latency_ms"] > 0
            and math.isfinite(p["latency_ms"]) and p["supported_classes"]]
    if not good:  # تحتاج الجلسة تجربة واحدة صالحة على الأقل حتى تقدر تختار نموذجًا.
        raise ValueError("No local detection model completed a valid calibration trial.")
    if recorded:
        good = [p for p in good if p.get("recorded_selection_error") is None]  # نجاح تشغيل النموذج وحده لا يكفي لاختيار أوزان تجريبية للفيديو المسجّل.
        if not good:
            raise ValueError("No model is eligible for recorded analysis. Install baseline weights or validate the custom weights and input size first.")
    groups = {}
    for p in good:
        groups.setdefault(tuple(sorted(p["supported_classes"])), []).append(p)
    supported = max(sorted(groups), key=lambda key: (len(key), len(groups[key])))
    comparable = groups[supported]  # تدخل المقارنة إعدادات من مجموعة الفئات المختارة حتى تراعي عدد أنواع الأجسام المدعومة.
    target_fps = min(source_fps, POLICY["max_analysis_fps"])
    budget_ms = 1000 / target_fps  # هذا الزمن المتوقع لمعالجة صورة واحدة ضمن المعدل المستهدف.
    feasible = [p for p in comparable
                if p["latency_ms"] * POLICY["timing_margin"] <= budget_ms]
    eligible = comparable if recorded else feasible  # يسمح الفيديو المسجل بإعداد يستغرق وقتًا أطول من العرض عند توفر تقييم سابق لهذا الإعداد.
    validated = [p for p in eligible if "validation_map50" in p]
    if validated:  # يعتمد قرار الدقة على نتائج تحقق محفوظة عندما تتوفر لإعداد مؤهل.
        selected = max(validated, key=lambda p: (p["validation_map50"], -p["latency_ms"]))
        reason = "Selected by labelled validation mAP50 within the eligible class coverage and timing constraints."
    elif feasible:
        if difficult:  # عند غياب تقييم مناسب، يعطي البرنامج أولوية للإعدادات بحسب وضوح العينة وزمن التشغيل المقاس.
            selected = max(feasible, key=lambda p: (p["parameters"], p["imgsz"], -p["latency_ms"]))
        else:
            selected = max(feasible, key=lambda p: (p["imgsz"], p["parameters"], -p["latency_ms"]))
        reason = "Measured timing fits the source-rate budget; detail/model preference is heuristic."
    else:
        selected = min(comparable, key=lambda p: p["latency_ms"])
        reason = "No profile meets the timing budget; selected the fastest compatible profile."
    stride = 1 if recorded else frame_stride(source_fps, selected["latency_ms"])
    return dict(selected), stride, reason  # تتضمن النتيجة إعداد النموذج، والفاصل الذي سيستخدم للتحليل، ونص السبب الذي يظهر في سجل القرار.


def validation_profiles(project_root):
    path = project_root / "docs" / "validation_profiles.json"  # يقرأ البرنامج تقييمات تحقق سابقة مرتبطة ببصمة ملف النموذج وحجم الإدخال اللذين جرى تقييمهما فعليًا.
    if not path.is_file():
        return None, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Validation profiles must be a JSON object.")
    protocol = data.get("protocol")
    profiles_by_hash = data.get("map50_by_weights_sha256")
    if not isinstance(protocol, str) or not protocol.strip() or not isinstance(profiles_by_hash, dict):
        raise ValueError("Validation profiles require a named protocol and scores by fingerprint.")
    scores = {}
    for digest, profiles in profiles_by_hash.items():
        if (len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)
                or not isinstance(profiles, dict)):
            raise ValueError("Invalid weights fingerprint in validation profiles.")
        for size, score in profiles.items():
            if (size not in {str(value) for value in POLICY["sizes"]}
                    or type(score) not in (int, float) or not 0 <= score <= 1):
                raise ValueError("Invalid size or mAP50 in validation profiles.")
            scores[digest, int(size)] = score
    return protocol, scores


def checked_video(value):
    if "://" in str(value) or str(value).isdigit():  # تعمل هذه التجربة على فيديو محفوظ على الجهاز حتى يستطيع الزملاء إعادة تشغيل نفس المقطع.
        raise ValueError("Use a local recorded video file, not a camera number or stream URL.")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Video file does not exist: {path}")
    if path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
        raise ValueError("Unsupported recorded-video extension.")
    return path


def write_json(path, data):
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(text, encoding="utf-8")  # يُحفظ السجل بصيغة منظّمة يمكن فتحها كنص وقراءتها برمجيًا، مع عرض الأحرف العربية بوضوح.


def sample_video(capture, total, calibration_end):
    sample_count = total if calibration_end is None else min(total, calibration_end)  # في محاكاة الانقطاع يختار البرنامج إعداداته من الصور التي كان يمكن قراءتها قبل فقد الاتصال.
    positions = sorted({round((sample_count - 1) * fraction) for fraction in POLICY["sample_fractions"]})
    samples, quality = [], []
    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, frame = capture.read()
        if not ok or frame is None or not frame.size:
            continue
        samples.append(frame)
        scale = min(1.0, 640 / max(frame.shape[:2]))
        probe = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
        gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)  # تُستخدم درجات الرمادي لحساب إضاءة الصورة وتباينها وحدّة تفاصيلها على نفس المقياس تقريبًا.
        quality.append({"brightness": float(gray.mean()), "contrast": float(gray.std()),
                        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var())})
    if not samples:
        raise ValueError("No readable calibration frames were found.")
    means = {key: statistics.mean(q[key] for q in quality) for key in quality[0]}
    difficult = means["brightness"] < 45 or means["brightness"] > 210 or means["contrast"] < 25 or means["laplacian_variance"] < 70
    quality_summary = {
        **means,
        "low_visibility_heuristic": difficult,
        "note": "Quality thresholds are heuristic, not a ground-truth quality rating.",
    }
    return samples, positions, quality_summary


def sync_device(device):
    if device != "cpu":
        import torch

        torch.cuda.synchronize()  # أوامر كرت الشاشة قد تُرسل قبل اكتمالها. الانتظار هنا يجعل زمن القياس يشمل العمل الفعلي على الكرت.


def calibrate_models(project_root, output, samples, video_long_edge, device):
    import torch
    from ultralytics import YOLO

    weights = project_root / "weights"  # المجلد هو نقطة البحث عن النماذج الجاهزة وعن الملفات الناتجة من تدريب الفريق.
    official = {path.name: path for path in sorted(project_root.glob("yolov8*.pt"))}
    official.update({path.name: path for path in sorted(weights.glob("yolov8*.pt"))})
    paths = [official[name] for name in sorted(official)]
    academic_paths = sorted((weights / "academic").glob("*.pt"))
    paths.extend(academic_paths)
    if not paths:  # يفشل الاختبار برسالة إعداد واضحة إذا لم يجد ملفات نموذج محلية.
        raise ValueError("Put local YOLOv8 model files in weights/, or trained models in weights/academic/.")
    protocol, validation_scores = validation_profiles(project_root)
    profiles = []
    for path in paths:
        model = None
        try:
            model = YOLO(str(path))
            if model.task != "detect":
                profiles.append({"weights": str(path), "ok": False, "error": "Not a detection model."})
                continue
            mapping = class_mapping(model.names)
            if not mapping:
                profiles.append({"weights": str(path), "ok": False, "error": "No requested classes in weights."})
                continue
            parameters = sum(p.numel() for p in model.model.parameters())
            digest = None
            if validation_scores:
                with path.open("rb") as file:
                    digest = hashlib.file_digest(file, "sha256").hexdigest()  # تمنع البصمة تطبيق تقييم نموذج قديم على ملف أوزان آخر حتى إذا تساوت أسماؤهما.
            sizes = [size for size in POLICY["sizes"]
                     if size == 640 or video_long_edge >= size or (digest, size) in validation_scores]
            for size in sizes:
                trial = {"weights": str(path), "imgsz": size, "parameters": parameters,
                         "supported_classes": sorted(set(mapping.values()))}
                if (digest, size) in validation_scores and set(mapping.values()) == set(TARGET_ALIASES):
                    trial.update(validation_map50=validation_scores[digest, size],
                                 validation_protocol=protocol, weights_sha256=digest)
                if trial.get("validation_map50") == 0:
                    trial["recorded_selection_error"] = "No correct detections in labelled validation."
                elif path in academic_paths and "validation_map50" not in trial:
                    trial["recorded_selection_error"] = "Custom weights and input size lack matching labelled validation."  # التحقق يرتبط ببصمة الأوزان وحجم الإدخال نفسهما، حتى لا تنتقل نتيجة قديمة إلى إعداد جديد.
                try:
                    kwargs = dict(imgsz=size, conf=POLICY["confidence"], classes=sorted(mapping), device=device, verbose=False)
                    model.predict(samples[0], **kwargs)  # أول توقع يهيّئ النموذج وموارد التشغيل قبل تسجيل زمن المعالجة على الصور التالية.
                    times = []  # نسجل أوقات الصور بعد التهيئة حتى يمثل القياس سرعة التحليل أثناء العمل المعتاد.
                    for frame in samples:
                        sync_device(device)
                        tick = time.perf_counter()
                        result = model.predict(frame, **kwargs)[0]
                        sync_device(device)
                        times.append((time.perf_counter() - tick) * 1000)
                        del result
                    trial.update(ok=True, latency_ms=statistics.median(times))  # الزمن الأوسط يقلل أثر قراءة واحدة بطيئة على قرار اختيار الإعداد.
                    print(f"Calibrated {path.name}, size {size}: {trial['latency_ms']:.1f} ms", flush=True)
                except Exception as exc:
                    trial.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                profiles.append(trial)
                write_json(output / "calibration.json", profiles)
        except Exception as exc:
            profiles.append({"weights": str(path), "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            del model
            gc.collect()
            if device != "cpu":
                torch.cuda.empty_cache()
    write_json(output / "calibration.json", profiles)
    return profiles


def write_detections(result, names, table, index, fps):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    positions = boxes.xyxy.cpu().tolist()
    classes = boxes.cls.int().cpu().tolist()
    scores = boxes.conf.cpu().tolist()
    track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [""] * len(boxes)
    labels = []
    for class_id, score, position, track_id in zip(classes, scores, positions, track_ids):
        name = names[class_id]
        table.writerow([index, index / fps, name, score, track_id, *position])  # يسمح السطر بربط كل كشف بوقت ظهوره وفئته ورقم تتبّعه وموقع مربعه داخل الصورة.
        shown_id = "-" if track_id == "" else track_id  # تظهر شرطة إذا لم يسند المتتبّع رقمًا لهذا الكشف عند حفظه.
        labels.append(f"ID {shown_id}  {name}  {score:.0%}")
    return labels


def annotate_frame(image, size, labels, analysed, class_warning):
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)  # تُوحّد أبعاد الصورة مع أبعاد ملف الفيديو الناتج قبل إرساله إلى كاتب الفيديو.
    lines = labels[:8]
    if len(labels) > 8:
        lines.append(f"+ {len(labels) - 8} more detections")
    for row, label in enumerate(lines):
        y = 22 + row * 22
        corner = (min(size[0] - 5, 290), y + 4)
        cv2.rectangle(image, (5, y - 17), corner, (25, 35, 30), -1)  # تجعل الخلفية اسم الفئة ورقم التتبع واضحين فوق صور الفيديو الفاتحة أو المزدحمة.
        cv2.putText(image, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    if not analysed:
        cv2.putText(image, "Not analysed (automatic scheduling)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 165, 255), 2)  # تُظهر هذه العبارة أن الإطار بقي في فيديو المحاكاة بينما تجاوزته المعالجة بسبب الجدولة.
    if class_warning:
        origin = (10, size[1] - 12)
        cv2.putText(image, class_warning, origin, cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 3)
        cv2.putText(image, class_warning, origin, cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
    return image


def open_result(video: Path) -> bool:
    try:
        if sys.platform == "win32":
            os.startfile(str(video.resolve()))  # بعد اكتمال التجربة يفتح المشغّل الذي يستخدمه ويندوز عادة للفيديو حتى يقدر المستخدم يراجع النتيجة.
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(video.resolve())])
        return True
    except OSError as exc:
        print(f"Could not open the video player: {exc}. Open this file manually: {video}")
        return False


if __name__ == '__main__':  # يتحقق هذا الشرط أن بايثون بدأ من هذا الملف نفسه، كما يحدث عند تشغيل run.bat.
    main()  # تقرأ main طريقة التشغيل. إذا شغّل المستخدم الملف بدون خيارات، تعرض قائمة التجارب وتنتظر اختياره.
