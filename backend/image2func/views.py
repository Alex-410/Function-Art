import os
import time
import uuid
import cv2
import numpy as np
import psutil
import json
import signal
import pickle
from concurrent.futures import ProcessPoolExecutor
from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import ensure_csrf_cookie

from .utils.contour_processor import extract_contours_from_array, get_contour_colors
from .utils.function_fitter import fit_all_contours
from .utils.frame_worker import process_single_frame

# Global cache for processed video frames
_frame_cache = {}
_cache_lock = False

def kill_residual_workers():
    """Kill any residual Python worker processes from previous runs."""
    current_pid = os.getpid()
    killed = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = proc.info['cmdline'] or []
                if proc.info['pid'] != current_pid and any('frame_worker' in str(c) for c in cmdline):
                    try:
                        os.kill(proc.info['pid'], signal.SIGTERM)
                        killed += 1
                    except (OSError, ProcessLookupError):
                        pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return killed


def save_frame_to_disk(cache_id, frame_idx, frame_data, cache_dir):
    """Save a single frame's curve data to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, f"frame_{frame_idx:06d}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(frame_data, f)


def load_frame_from_disk(cache_dir, frame_idx):
    """Load a single frame's curve data from disk."""
    filepath = os.path.join(cache_dir, f"frame_{frame_idx:06d}.pkl")
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None


def clear_disk_cache(cache_dir):
    """Remove all cached frame files."""
    if os.path.exists(cache_dir):
        for f in os.listdir(cache_dir):
            if f.startswith('frame_') and f.endswith('.pkl'):
                try:
                    os.remove(os.path.join(cache_dir, f))
                except OSError:
                    pass
        try:
            os.rmdir(cache_dir)
        except OSError:
            pass


def get_system_info():
    """Get real CPU and memory info using psutil."""
    cpu_count = psutil.cpu_count(logical=True)
    cpu_physical = psutil.cpu_count(logical=False)
    mem = psutil.virtual_memory()
    total_gb = round(mem.total / (1024**3), 1)
    available_gb = round(mem.available / (1024**3), 1)
    return {
        'cpu_logical': cpu_count,
        'cpu_physical': cpu_physical,
        'memory_total_gb': total_gb,
        'memory_available_gb': available_gb,
    }


@ensure_csrf_cookie
def index(request):
    return render(request, 'image2func/index.html')


def system_info(request):
    """API endpoint to get real system hardware info."""
    info = get_system_info()
    return JsonResponse(info)


def _process_image_array(img_bgr, img_w, img_h, min_contour_length, smoothing, max_contours):
    """Process a single image from BGR array (no disk I/O)."""
    contours, original_img, edges_img, img_w_actual, img_h_actual = extract_contours_from_array(
        img_bgr, min_contour_length=min_contour_length
    )
    fitted_results = fit_all_contours(contours, smoothing=smoothing, max_contours=max_contours)
    colors = get_contour_colors(contours, original_img, max_contours=max_contours)

    curve_data = []
    for i, r in enumerate(fitted_results):
        xs = r['x']
        ys = [img_h - v for v in r['y']]
        curve_data.append({
            'index': r['index'],
            'num_points': r.get('num_points', 0),
            'expression': r.get('expression', 'Complex Curve'),
            'color': colors[i] if i < len(colors) else '#333333',
            'x': [round(v, 2) for v in xs],
            'y': [round(v, 2) for v in ys],
        })

    bounds = {
        'x_min': 0,
        'x_max': img_w,
        'y_min': 0,
        'y_max': img_h,
    }

    return {
        'curves': curve_data,
        'bounds': bounds,
        'total_contours': len(contours),
        'fitted_contours': len(fitted_results),
    }


def _extract_video_frames(video_path, target_fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(video_fps / target_fps)))

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_idx += 1

    cap.release()
    height = frames[0].shape[0] if frames else 0
    width = frames[0].shape[1] if frames else 0
    return frames, width, height, video_fps


def _process_video_parallel_with_progress(frames, img_w, img_h, min_contour_length, smoothing, max_contours, num_workers):
    """Process video frames in parallel with progress callback."""
    args_list = [
        (frame, img_w, img_h, min_contour_length, smoothing, max_contours)
        for frame in frames
    ]

    max_workers = min(num_workers, len(frames))
    total = len(frames)

    frame_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_single_frame, args_list)
        for i, result in enumerate(results):
            result['frame_index'] = i
            frame_results.append(result)
            yield {'type': 'progress', 'current': i + 1, 'total': total}

    global_bounds = {
        'x_min': 0,
        'x_max': img_w,
        'y_min': 0,
        'y_max': img_h,
    }

    total_contours = sum(fr['total_contours'] for fr in frame_results)
    total_fitted = sum(fr['fitted_contours'] for fr in frame_results)

    yield {
        'type': 'complete',
        'frames': frame_results,
        'bounds': global_bounds,
        'frame_count': len(frame_results),
        'total_contours': total_contours,
        'fitted_contours': total_fitted,
    }


def upload_and_process(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)

    file_key = 'image' if 'image' in request.FILES else ('video' if 'video' in request.FILES else None)
    if not file_key:
        return JsonResponse({'error': 'No file provided'}, status=400)

    uploaded_file = request.FILES[file_key]
    ext = os.path.splitext(uploaded_file.name)[1] or '.png'
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(settings.MEDIA_ROOT, filename)

    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    with open(save_path, 'wb+') as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)

    try:
        min_contour_length = int(request.POST.get('min_length', 50))
        smoothing = float(request.POST.get('smoothing', 5))
        max_contours = int(request.POST.get('max_contours', 50))
        target_fps = int(request.POST.get('target_fps', 3))
        num_workers = int(request.POST.get('num_workers', 4))
        mem_limit = float(request.POST.get('mem_limit', 0.5))
        downsample = float(request.POST.get('downsample', 1.0))
    except (ValueError, TypeError):
        min_contour_length = 50
        smoothing = 5
        max_contours = 50
        target_fps = 3
        num_workers = 4
        mem_limit = 0.5
        downsample = 1.0

    t_start = time.time()

    original_size = None
    processed_size = None

    try:
        if file_key == 'video':
            # Step 1: Extract frames to memory (single-threaded, I/O bound)
            frames, img_w, img_h, video_fps = _extract_video_frames(save_path, target_fps)
            if not frames:
                raise ValueError("No frames extracted from video")

            original_size = f"{img_w}x{img_h}"

            # Apply downsample for faster processing
            if downsample < 1.0 and downsample > 0:
                new_w = int(img_w * downsample)
                new_h = int(img_h * downsample)
                frames = [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames]
                img_w, img_h = new_w, new_h

            processed_size = f"{img_w}x{img_h}"

            # Apply memory limit: adjust max_contours based on memory limit
            if mem_limit < 1.0:
                adjusted_max_contours = max(5, int(max_contours * mem_limit))
            else:
                adjusted_max_contours = max_contours

            # Step 2: Process frames in parallel (CPU bound)
            result = _process_video_parallel(
                frames, img_w, img_h, min_contour_length, smoothing, adjusted_max_contours, num_workers
            )
            result['image_size'] = original_size
            result['processed_size'] = processed_size
            result['video_fps'] = round(video_fps, 2)
            result['total_frames'] = len(frames)
            result['adjusted_max_contours'] = adjusted_max_contours
            is_video = True
        else:
            img = cv2.imread(save_path)
            if img is None:
                raise ValueError(f"Cannot read image: {save_path}")
            img_h = img.shape[0]
            img_w = img.shape[1]
            original_size = f"{img_w}x{img_h}"

            # Apply downsample for faster processing
            if downsample < 1.0 and downsample > 0:
                new_w = int(img_w * downsample)
                new_h = int(img_h * downsample)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                img_w, img_h = new_w, new_h

            processed_size = f"{img_w}x{img_h}"

            result = _process_image_array(img, img_w, img_h, min_contour_length, smoothing, max_contours)
            result['image_size'] = original_size
            result['processed_size'] = processed_size
            is_video = False
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

    t_elapsed = time.time() - t_start

    summary = {
        'total_contours': result['total_contours'],
        'fitted_contours': result['fitted_contours'],
        'image_size': result.get('image_size', 'unknown'),
        'processed_size': result.get('processed_size', 'unknown'),
        'processing_time': round(t_elapsed, 2),
    }

    if is_video:
        return JsonResponse({
            'success': True,
            'is_video': True,
            'summary': summary,
            'frames': result['frames'],
            'bounds': result['bounds'],
            'frame_count': result['frame_count'],
            'video_fps': result.get('video_fps', 0),
        })
    else:
        return JsonResponse({
            'success': True,
            'is_video': False,
            'summary': summary,
            'curves': result['curves'],
            'bounds': result['bounds'],
        })


def upload_and_process_stream(request):
    """SSE streaming endpoint for video processing with real-time progress."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)

    file_key = 'image' if 'image' in request.FILES else ('video' if 'video' in request.FILES else None)
    if not file_key:
        return JsonResponse({'error': 'No file provided'}, status=400)

    uploaded_file = request.FILES[file_key]
    ext = os.path.splitext(uploaded_file.name)[1] or '.png'
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(settings.MEDIA_ROOT, filename)

    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    with open(save_path, 'wb+') as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)

    try:
        min_contour_length = int(request.POST.get('min_length', 50))
        smoothing = float(request.POST.get('smoothing', 5))
        max_contours = int(request.POST.get('max_contours', 50))
        target_fps = int(request.POST.get('target_fps', 3))
        num_workers = int(request.POST.get('num_workers', 4))
        mem_limit = float(request.POST.get('mem_limit', 0.5))
        downsample = float(request.POST.get('downsample', 1.0))
    except (ValueError, TypeError):
        min_contour_length = 50
        smoothing = 5
        max_contours = 50
        target_fps = 3
        num_workers = 4
        mem_limit = 0.5
        downsample = 1.0

    def event_stream():
        # Kill any residual worker processes first
        killed = kill_residual_workers()
        if killed > 0:
            yield f"data: {json.dumps({'type': 'status', 'message': f'清理 {killed} 个残留进程...'})}\n\n"
            time.sleep(0.5)

        # Create cache directory for this job
        cache_id = uuid.uuid4().hex[:12]
        cache_dir = os.path.join(settings.MEDIA_ROOT, 'cache', cache_id)
        os.makedirs(cache_dir, exist_ok=True)

        t_start = time.time()
        try:
            if file_key == 'video':
                # Step 1: Extract frames
                yield f"data: {json.dumps({'type': 'status', 'message': '正在提取视频帧...'})}\n\n"
                frames, img_w, img_h, video_fps = _extract_video_frames(save_path, target_fps)
                if not frames:
                    raise ValueError("No frames extracted from video")

                original_size = f"{img_w}x{img_h}"

                # Apply downsample for faster processing
                if downsample < 1.0 and downsample > 0:
                    new_w = int(img_w * downsample)
                    new_h = int(img_h * downsample)
                    frames = [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames]
                    img_w, img_h = new_w, new_h

                processed_size = f"{img_w}x{img_h}"
                total_frames = len(frames)
                yield f"data: {json.dumps({'type': 'status', 'message': f'共 {total_frames} 帧，开始处理...', 'total_frames': total_frames})}\n\n"

                # Apply memory limit
                if mem_limit < 1.0:
                    adjusted_max_contours = max(5, int(max_contours * mem_limit))
                else:
                    adjusted_max_contours = max_contours

                # Step 2: Process frames with progress - save to disk immediately
                args_list = [
                    (frame, img_w, img_h, min_contour_length, smoothing, adjusted_max_contours)
                    for frame in frames
                ]
                max_workers_actual = min(num_workers, len(frames))

                with ProcessPoolExecutor(max_workers=max_workers_actual) as executor:
                    results = executor.map(process_single_frame, args_list)
                    for i, result in enumerate(results):
                        result['frame_index'] = i
                        # Save to disk immediately, don't keep in memory
                        save_frame_to_disk(cache_id, i, result, cache_dir)
                        yield f"data: {json.dumps({'type': 'progress', 'current': i + 1, 'total': total_frames})}\n\n"

                global_bounds = {
                    'x_min': 0, 'x_max': img_w,
                    'y_min': 0, 'y_max': img_h,
                }

                # Calculate summary from first frame (we don't need all in memory)
                first_frame = load_frame_from_disk(cache_dir, 0)
                total_contours = first_frame['total_contours'] * total_frames if first_frame else 0
                total_fitted = first_frame['fitted_contours'] * total_frames if first_frame else 0

                t_elapsed = time.time() - t_start
                avg_time = round(t_elapsed / total_frames, 3) if total_frames > 0 else 0
                summary = {
                    'total_contours': total_contours,
                    'fitted_contours': total_fitted,
                    'image_size': original_size,
                    'processed_size': processed_size,
                    'processing_time': round(t_elapsed, 2),
                    'avg_time_per_frame': avg_time,
                    'total_frames': total_frames,
                }

                # Store cache info in global dict
                _frame_cache[cache_id] = {
                    'cache_dir': cache_dir,
                    'frame_count': total_frames,
                    'bounds': global_bounds,
                    'video_fps': round(video_fps, 2),
                    'summary': summary,
                }

                yield f"data: {json.dumps({'type': 'complete', 'success': True, 'is_video': True, 'cache_id': cache_id, 'summary': summary, 'bounds': global_bounds, 'frame_count': total_frames, 'video_fps': round(video_fps, 2)})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'status', 'message': '正在处理图片...'})}\n\n"
                img = cv2.imread(save_path)
                if img is None:
                    raise ValueError(f"Cannot read image: {save_path}")
                img_h = img.shape[0]
                img_w = img.shape[1]
                original_size = f"{img_w}x{img_h}"

                # Apply downsample for faster processing
                if downsample < 1.0 and downsample > 0:
                    new_w = int(img_w * downsample)
                    new_h = int(img_h * downsample)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    img_w, img_h = new_w, new_h

                processed_size = f"{img_w}x{img_h}"
                result = _process_image_array(img, img_w, img_h, min_contour_length, smoothing, max_contours)
                result['image_size'] = original_size
                result['processed_size'] = processed_size

                t_elapsed = time.time() - t_start
                summary = {
                    'total_contours': result['total_contours'],
                    'fitted_contours': result['fitted_contours'],
                    'image_size': original_size,
                    'processed_size': processed_size,
                    'processing_time': round(t_elapsed, 2),
                    'avg_time_per_frame': 0,
                    'total_frames': 1,
                }

                yield f"data: {json.dumps({'type': 'complete', 'success': True, 'is_video': False, 'summary': summary, 'curves': result['curves'], 'bounds': result['bounds']})}\n\n"
        except Exception as e:
            # Clean up cache on error
            clear_disk_cache(cache_dir)
            if cache_id in _frame_cache:
                del _frame_cache[cache_id]
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response


def get_frame(request):
    """API to get a single frame's data from disk cache."""
    cache_id = request.GET.get('cache_id')
    frame_idx = int(request.GET.get('frame_idx', 0))

    if not cache_id or cache_id not in _frame_cache:
        return JsonResponse({'error': 'Cache not found'}, status=404)

    cache_info = _frame_cache[cache_id]
    frame_data = load_frame_from_disk(cache_info['cache_dir'], frame_idx)

    if frame_data is None:
        return JsonResponse({'error': 'Frame not found'}, status=404)

    return JsonResponse({'success': True, 'frame': frame_data})


def get_all_frames(request):
    """API to get all frame indices from cache."""
    cache_id = request.GET.get('cache_id')

    if not cache_id or cache_id not in _frame_cache:
        return JsonResponse({'error': 'Cache not found'}, status=404)

    cache_info = _frame_cache[cache_id]
    return JsonResponse({
        'success': True,
        'frame_count': cache_info['frame_count'],
        'bounds': cache_info['bounds'],
        'video_fps': cache_info['video_fps'],
        'summary': cache_info['summary'],
    })


def clear_cache(request):
    """API to clear disk cache."""
    cache_id = request.GET.get('cache_id')
    if cache_id and cache_id in _frame_cache:
        cache_info = _frame_cache[cache_id]
        clear_disk_cache(cache_info['cache_dir'])
        del _frame_cache[cache_id]
        return JsonResponse({'success': True, 'message': 'Cache cleared'})
    return JsonResponse({'error': 'Cache not found'}, status=404)
