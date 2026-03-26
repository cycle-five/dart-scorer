#!/usr/bin/env python3
"""
audio_trigger.py — ML-based dart impact audio detection.

Collects audio samples, trains a small classifier on MFCC features,
and provides a real-time trigger for dart impacts.

Usage:
    python audio_trigger.py --record             # Record labeled samples
    python audio_trigger.py --record --device 9  # Specific mic
    python audio_trigger.py --train              # Train classifier
    python audio_trigger.py --test               # Test classifier live
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np
import config

AUDIO_DATA_DIR = config.PROJECT_ROOT / "data" / "audio"
MODEL_PATH = AUDIO_DATA_DIR / "dart_classifier.pkl"
SAMPLES_DIR = AUDIO_DATA_DIR / "samples"

# Audio params
SAMPLE_RATE = 44100
CLIP_DURATION = 0.3  # seconds per clip
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)
N_MFCC = 13


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def compute_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """Compute MFCCs from a short audio clip using scipy (no librosa needed).

    Returns a feature vector of length n_mfcc * 3 (mean, std, max of each coeff).
    """
    from scipy.fft import dct
    from scipy.signal import get_window

    # Pre-emphasis
    emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # Frame the signal
    frame_size = 512
    hop = 256
    n_frames = max(1, (len(emphasized) - frame_size) // hop + 1)

    window = get_window('hann', frame_size)
    frames = np.zeros((n_frames, frame_size))
    for i in range(n_frames):
        start = i * hop
        end = start + frame_size
        if end <= len(emphasized):
            frames[i] = emphasized[start:end] * window
        else:
            chunk = emphasized[start:]
            frames[i, :len(chunk)] = chunk[:frame_size] * window[:len(chunk)]

    # Power spectrum
    mag = np.abs(np.fft.rfft(frames, n=frame_size)) ** 2
    mag = np.maximum(mag, 1e-10)

    # Mel filterbank
    n_filt = 26
    low_freq = 0
    high_freq = sr / 2
    low_mel = 2595 * np.log10(1 + low_freq / 700)
    high_mel = 2595 * np.log10(1 + high_freq / 700)
    mel_points = np.linspace(low_mel, high_mel, n_filt + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bins = np.floor((frame_size + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_filt, frame_size // 2 + 1))
    for i in range(n_filt):
        for j in range(bins[i], bins[i + 1]):
            if bins[i + 1] > bins[i]:
                fbank[i, j] = (j - bins[i]) / (bins[i + 1] - bins[i])
        for j in range(bins[i + 1], bins[i + 2]):
            if bins[i + 2] > bins[i + 1]:
                fbank[i, j] = (bins[i + 2] - j) / (bins[i + 2] - bins[i + 1])

    # Apply filterbank
    filter_banks = np.dot(mag, fbank.T)
    filter_banks = np.maximum(filter_banks, 1e-10)
    filter_banks = 20 * np.log10(filter_banks)

    # DCT to get MFCCs
    mfccs = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]

    # Aggregate across frames: mean, std, max
    features = np.concatenate([
        np.mean(mfccs, axis=0),
        np.std(mfccs, axis=0),
        np.max(mfccs, axis=0),
    ])
    return features


def compute_features(audio, sr=SAMPLE_RATE):
    """Compute full feature vector: MFCCs + temporal features."""
    mfcc = compute_mfcc(audio, sr)

    # Additional temporal features
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    zero_crossings = float(np.sum(np.abs(np.diff(np.sign(audio))) > 0)) / len(audio)

    # Attack time — how quickly energy rises
    frame_size = 256
    n_frames = len(audio) // frame_size
    if n_frames > 1:
        frame_energies = [np.sqrt(np.mean(audio[i*frame_size:(i+1)*frame_size] ** 2))
                          for i in range(n_frames)]
        peak_frame = np.argmax(frame_energies)
        attack_ratio = peak_frame / n_frames  # 0 = instant attack, 1 = slow rise
    else:
        attack_ratio = 0.5

    # Spectral centroid
    fft_mag = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    if np.sum(fft_mag) > 0:
        spectral_centroid = float(np.sum(freqs * fft_mag) / np.sum(fft_mag))
    else:
        spectral_centroid = 0.0

    extra = np.array([rms, peak, zero_crossings, attack_ratio, spectral_centroid / sr])
    return np.concatenate([mfcc, extra])


# ---------------------------------------------------------------------------
# Audio device helpers
# ---------------------------------------------------------------------------

def find_emeet_device():
    """Auto-detect the eMeet C950 mic device index."""
    import sounddevice as sd
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0 and 'eMeet' in d.get('name', ''):
            return i
    return None


# ---------------------------------------------------------------------------
# Recording mode — visual interface with online learning
# ---------------------------------------------------------------------------

def _extract_clip(audio_buffer, buffer_pos, buffer_size):
    """Extract the last CLIP_DURATION from a rolling buffer."""
    pos = buffer_pos
    if pos >= CLIP_SAMPLES:
        return audio_buffer[pos - CLIP_SAMPLES:pos].copy()
    else:
        return np.concatenate([
            audio_buffer[buffer_size - (CLIP_SAMPLES - pos):],
            audio_buffer[:pos]
        ]).copy()


def _retrain_online(info_path, clf_holder):
    """Retrain classifier on all collected samples. Updates clf_holder[0]."""
    from sklearn.ensemble import GradientBoostingClassifier

    if not info_path.exists():
        return 0, 0

    entries = []
    with open(info_path) as f:
        for line in f:
            entries.append(json.loads(line))

    dart_count = sum(1 for e in entries if e["label"] == "dart")
    noise_count = sum(1 for e in entries if e["label"] == "noise")

    if dart_count < 3 or noise_count < 3:
        return dart_count, noise_count

    X, y = [], []
    for entry in entries:
        path = SAMPLES_DIR / entry["filename"]
        if not path.exists():
            continue
        clip = np.load(str(path), allow_pickle=False)
        X.append(compute_features(clip))
        y.append(1 if entry["label"] == "dart" else 0)

    X = np.array(X)
    y = np.array(y)

    clf = GradientBoostingClassifier(
        n_estimators=min(50 + len(y) * 2, 200),
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    clf.fit(X, y)
    clf_holder[0] = clf

    # Also save to disk
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(clf, MODEL_PATH)

    return dart_count, noise_count


def record_samples(device=None, rms_threshold=0.01):
    """Visual interactive sample recording with auto-detection and online learning.

    Monitors mic, auto-pauses when sound detected, asks D/N to classify.
    After enough samples, starts predicting and shows confidence.
    """
    import sounddevice as sd
    import cv2

    if device is None:
        device = find_emeet_device()
    if device is None:
        print("ERROR: No eMeet mic found. Use --device N.")
        return

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    info_path = AUDIO_DATA_DIR / "samples.jsonl"

    device_info = sd.query_devices(device)
    print(f"Recording from: {device_info['name']} (device {device})")

    # Count existing samples
    dart_count = 0
    noise_count = 0
    if info_path.exists():
        with open(info_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry["label"] == "dart":
                    dart_count += 1
                else:
                    noise_count += 1
    sample_idx = dart_count + noise_count

    # Online classifier
    clf_holder = [None]
    if dart_count >= 3 and noise_count >= 3:
        _retrain_online(info_path, clf_holder)
        print(f"Loaded online model from {dart_count + noise_count} samples")

    # Rolling audio buffer
    buffer_seconds = 2.0
    buffer_size = int(SAMPLE_RATE * buffer_seconds)
    audio_buffer = np.zeros(buffer_size, dtype=np.float32)
    buffer_pos = [0]
    current_rms = [0.0]
    rms_history = []
    history_max = 300
    baseline_rms = [0.01]

    def audio_callback(indata, frames, time_info, status):
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        n = len(mono)
        pos = buffer_pos[0]
        if pos + n <= buffer_size:
            audio_buffer[pos:pos + n] = mono
        else:
            overflow = (pos + n) - buffer_size
            audio_buffer[pos:] = mono[:n - overflow]
            audio_buffer[:overflow] = mono[n - overflow:]
        buffer_pos[0] = (pos + n) % buffer_size
        rms = float(np.sqrt(np.mean(mono ** 2)))
        current_rms[0] = rms
        if rms < baseline_rms[0] * 3:
            baseline_rms[0] = 0.99 * baseline_rms[0] + 0.01 * rms

    stream = sd.InputStream(
        device=device, channels=1, samplerate=SAMPLE_RATE,
        blocksize=1024, callback=audio_callback,
    )
    stream.start()

    # Visual window
    win_w, win_h = 700, 350
    win = "Audio Sample Collector"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, win_w, win_h)

    import subprocess
    import tempfile
    import wave

    # States
    STATE_LISTENING = 0
    STATE_PAUSED = 1  # heard something, waiting for classification
    state = STATE_LISTENING
    paused_clip = None
    paused_rms = 0.0
    paused_prediction = None  # (prob, label) or None
    is_playing = [False]  # playback in progress
    last_trigger_time = 0.0
    cooldown = 0.8  # seconds between auto-detections

    _play_proc = [None]

    def play_clip(clip):
        """Play an audio clip via paplay (PipeWire/PulseAudio), normalized."""
        if clip is None:
            return
        # Kill any previous playback
        if _play_proc[0] is not None and _play_proc[0].poll() is None:
            _play_proc[0].kill()
        is_playing[0] = True
        try:
            peak = np.max(np.abs(clip))
            if peak > 0:
                normalized = clip / peak * 0.9
            else:
                normalized = clip
            int16_data = (normalized * 32767).astype(np.int16)

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav_path = tmp.name
            tmp.close()
            with wave.open(wav_path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(int16_data.tobytes())

            _play_proc[0] = subprocess.Popen(
                ["paplay", wav_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"  Playback error: {e}")
            is_playing[0] = False

    print(f"\n=== Audio Sample Collector ===")
    print(f"Existing: {dart_count} dart + {noise_count} noise")
    print(f"RMS threshold: {rms_threshold:.4f} (adjust with +/-)")
    print(f"Controls: D=dart, N=noise, P=playback, +/-=threshold, Q=quit\n")

    try:
        while True:
            rms = current_rms[0]
            rms_history.append(rms)
            if len(rms_history) > history_max:
                rms_history.pop(0)

            # Create display
            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

            if state == STATE_LISTENING:
                # Auto-detect: something loud happened
                now = time.monotonic()
                if rms > rms_threshold and now - last_trigger_time > cooldown:
                    paused_clip = _extract_clip(audio_buffer, buffer_pos[0], buffer_size)
                    paused_rms = rms
                    last_trigger_time = now

                    # Try to predict if we have a model
                    if clf_holder[0] is not None:
                        features = compute_features(paused_clip).reshape(1, -1)
                        prob = float(clf_holder[0].predict_proba(features)[0][1])
                        pred_label = "DART" if prob > 0.5 else "NOISE"
                        paused_prediction = (prob, pred_label)
                    else:
                        paused_prediction = None

                    state = STATE_PAUSED

            # --- Draw waveform ---
            wave_x, wave_y, wave_w, wave_h = 20, 20, win_w - 40, 120
            cv2.rectangle(canvas, (wave_x, wave_y), (wave_x + wave_w, wave_y + wave_h),
                          (40, 40, 40), -1)

            # Auto-scale
            peak_hist = max(rms_history) if rms_history else 0.01
            y_scale = max(rms_threshold * 2, peak_hist * 1.2, 0.01)

            # Threshold line
            thresh_py = wave_y + wave_h - int((rms_threshold / y_scale) * wave_h)
            thresh_py = max(wave_y + 1, min(wave_y + wave_h - 1, thresh_py))
            cv2.line(canvas, (wave_x, thresh_py), (wave_x + wave_w, thresh_py),
                     (0, 255, 255), 1)
            cv2.putText(canvas, f"threshold={rms_threshold:.4f}",
                        (wave_x + wave_w - 170, thresh_py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

            # Waveform
            if len(rms_history) > 1:
                for i in range(1, len(rms_history)):
                    x1 = wave_x + int((i - 1) / history_max * wave_w)
                    x2 = wave_x + int(i / history_max * wave_w)
                    y1 = wave_y + wave_h - int((rms_history[i-1] / y_scale) * wave_h)
                    y2 = wave_y + wave_h - int((rms_history[i] / y_scale) * wave_h)
                    y1 = max(wave_y + 1, min(wave_y + wave_h - 1, y1))
                    y2 = max(wave_y + 1, min(wave_y + wave_h - 1, y2))
                    color = (0, 0, 255) if rms_history[i] > rms_threshold else (0, 180, 0)
                    cv2.line(canvas, (x1, y1), (x2, y2), color, 1)

            # --- Status area ---
            stats_y = wave_y + wave_h + 20

            # Counts
            cv2.putText(canvas, f"Dart: {dart_count}   Noise: {noise_count}   Total: {dart_count + noise_count}",
                        (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            stats_y += 25

            model_status = "ACTIVE" if clf_holder[0] is not None else "need 3+ of each"
            cv2.putText(canvas, f"Online model: {model_status}",
                        (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            stats_y += 30

            if state == STATE_LISTENING:
                # Listening state
                cv2.putText(canvas, "LISTENING...", (20, stats_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                stats_y += 30
                cv2.putText(canvas, f"RMS: {rms:.4f}",
                            (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
                stats_y += 35
                cv2.putText(canvas, "D=dart  N=noise  +/-=threshold  Q=quit",
                            (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            elif state == STATE_PAUSED:
                # Paused — classify this sound
                cv2.putText(canvas, "SOUND DETECTED!", (20, stats_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                stats_y += 30
                cv2.putText(canvas, f"RMS: {paused_rms:.4f}",
                            (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
                stats_y += 30

                if paused_prediction is not None:
                    prob, pred_label = paused_prediction
                    pred_color = (0, 200, 0) if pred_label == "DART" else (0, 100, 200)
                    cv2.putText(canvas, f"Prediction: {pred_label} ({prob:.0%})",
                                (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
                    stats_y += 30
                    cv2.putText(canvas, "ENTER=accept  D=dart  N=noise  P=play  ESC=skip",
                                (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                else:
                    cv2.putText(canvas, "Is this a DART or NOISE?",
                                (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    stats_y += 30
                    cv2.putText(canvas, "D=dart  N=noise  P=play  ESC=skip",
                                (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # Update playback status
                if is_playing[0] and _play_proc[0] is not None:
                    if _play_proc[0].poll() is not None:
                        is_playing[0] = False
                if is_playing[0]:
                    stats_y += 25
                    cv2.putText(canvas, "Playing...", (20, stats_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

            cv2.imshow(win, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('+') or key == ord('='):
                rms_threshold = min(rms_threshold + 0.002, 1.0)
                print(f"  RMS threshold: {rms_threshold:.4f}")

            elif key == ord('-'):
                rms_threshold = max(rms_threshold - 0.002, 0.001)
                print(f"  RMS threshold: {rms_threshold:.4f}")

            elif state == STATE_PAUSED:
                label = None

                if key == ord('p') and paused_clip is not None:
                    play_clip(paused_clip)
                    print("  Playing clip...")
                    continue
                elif key == 13 and paused_prediction is not None:
                    # ENTER = accept prediction
                    _, pred_label = paused_prediction
                    label = "dart" if pred_label == "DART" else "noise"
                elif key == ord('d'):
                    label = "dart"
                elif key == ord('n'):
                    label = "noise"
                elif key == 27:  # ESC = skip
                    state = STATE_LISTENING
                    paused_clip = None
                    paused_prediction = None
                    print("  Skipped")
                    continue

                if label is not None:
                    fname = f"sample_{sample_idx:04d}_{label}.npy"
                    np.save(str(SAMPLES_DIR / fname), paused_clip)

                    entry = {
                        "filename": fname,
                        "label": label,
                        "rms": float(np.sqrt(np.mean(paused_clip ** 2))),
                        "peak": float(np.max(np.abs(paused_clip))),
                        "timestamp": time.time(),
                    }
                    with open(info_path, "a") as f:
                        f.write(json.dumps(entry) + "\n")

                    if label == "dart":
                        dart_count += 1
                    else:
                        noise_count += 1
                    sample_idx += 1

                    print(f"  Saved {label}: {fname} (RMS={entry['rms']:.4f})")

                    # Retrain online every 5 new samples (or first time at 3+3)
                    total = dart_count + noise_count
                    if dart_count >= 3 and noise_count >= 3 and total % 5 == 0:
                        print(f"  Retraining online model ({total} samples)...")
                        _retrain_online(info_path, clf_holder)
                        print(f"  Model updated")

                    state = STATE_LISTENING
                    paused_clip = None
                    paused_prediction = None

            elif state == STATE_LISTENING:
                # Allow manual D/N even in listening mode
                if key == ord('d') or key == ord('n'):
                    clip = _extract_clip(audio_buffer, buffer_pos[0], buffer_size)
                    label = "dart" if key == ord('d') else "noise"
                    fname = f"sample_{sample_idx:04d}_{label}.npy"
                    np.save(str(SAMPLES_DIR / fname), clip)

                    entry = {
                        "filename": fname,
                        "label": label,
                        "rms": float(np.sqrt(np.mean(clip ** 2))),
                        "peak": float(np.max(np.abs(clip))),
                        "timestamp": time.time(),
                    }
                    with open(info_path, "a") as f:
                        f.write(json.dumps(entry) + "\n")

                    if label == "dart":
                        dart_count += 1
                    else:
                        noise_count += 1
                    sample_idx += 1
                    print(f"  Manual save {label}: {fname}")

                    total = dart_count + noise_count
                    if dart_count >= 3 and noise_count >= 3 and total % 5 == 0:
                        print(f"  Retraining online model ({total} samples)...")
                        _retrain_online(info_path, clf_holder)
                        print(f"  Model updated")

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
        cv2.destroyAllWindows()

    # Final retrain and save
    if dart_count >= 3 and noise_count >= 3:
        print("Final model training...")
        _retrain_online(info_path, clf_holder)

    print(f"\nDone. {dart_count} dart + {noise_count} noise samples in {SAMPLES_DIR}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifier():
    """Train a classifier on collected audio samples."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    info_path = AUDIO_DATA_DIR / "samples.jsonl"
    if not info_path.exists():
        print("No samples found. Run --record first.")
        return

    entries = []
    with open(info_path) as f:
        for line in f:
            entries.append(json.loads(line))

    dart_count = sum(1 for e in entries if e["label"] == "dart")
    noise_count = sum(1 for e in entries if e["label"] == "noise")
    print(f"Dataset: {dart_count} dart + {noise_count} noise = {len(entries)} total")

    if dart_count < 5 or noise_count < 5:
        print("Need at least 5 of each class. Collect more samples.")
        return

    # Extract features
    X = []
    y = []
    for entry in entries:
        path = SAMPLES_DIR / entry["filename"]
        if not path.exists():
            continue
        clip = np.load(str(path), allow_pickle=False)
        features = compute_features(clip)
        X.append(features)
        y.append(1 if entry["label"] == "dart" else 0)

    X = np.array(X)
    y = np.array(y)
    print(f"Features: {X.shape[1]} per sample")

    # Train
    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )

    # Cross-validation
    if len(y) >= 10:
        scores = cross_val_score(clf, X, y, cv=min(5, len(y) // 2), scoring='f1')
        print(f"Cross-val F1: {scores.mean():.3f} +/- {scores.std():.3f}")

    # Train on full dataset
    clf.fit(X, y)

    # Save
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Feature importance
    feat_names = [f"mfcc_mean_{i}" for i in range(N_MFCC)]
    feat_names += [f"mfcc_std_{i}" for i in range(N_MFCC)]
    feat_names += [f"mfcc_max_{i}" for i in range(N_MFCC)]
    feat_names += ["rms", "peak", "zero_crossings", "attack_ratio", "spectral_centroid"]

    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("\nTop 10 features:")
    for i in top_idx:
        print(f"  {feat_names[i]:25s} {importances[i]:.4f}")


# ---------------------------------------------------------------------------
# Live test
# ---------------------------------------------------------------------------

def test_live(device=None):
    """Test the trained classifier on live audio."""
    import sounddevice as sd

    if not MODEL_PATH.exists():
        print(f"No model found at {MODEL_PATH}. Run --train first.")
        return

    import joblib
    clf = joblib.load(MODEL_PATH)

    if device is None:
        device = find_emeet_device()
    if device is None:
        print("ERROR: No eMeet mic found.")
        return

    device_info = sd.query_devices(device)
    print(f"Testing on: {device_info['name']}")
    print("Listening for dart impacts... (Ctrl+C to stop)\n")

    buffer_size = int(SAMPLE_RATE * 2.0)
    audio_buffer = np.zeros(buffer_size, dtype=np.float32)
    buffer_pos = [0]
    last_trigger = [0.0]
    cooldown = 1.0

    def audio_callback(indata, frames, time_info, status):
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        n = len(mono)
        pos = buffer_pos[0]
        if pos + n <= buffer_size:
            audio_buffer[pos:pos + n] = mono
        else:
            overflow = (pos + n) - buffer_size
            audio_buffer[pos:] = mono[:n - overflow]
            audio_buffer[:overflow] = mono[n - overflow:]
        buffer_pos[0] = (pos + n) % buffer_size

        # Check for dart every ~50ms worth of audio
        now = time.monotonic()
        if now - last_trigger[0] < cooldown:
            return

        # RMS gate — skip classification if too quiet
        rms = float(np.sqrt(np.mean(mono ** 2)))
        if rms < 0.005:
            return

        # Extract clip and classify
        p = buffer_pos[0]
        if p >= CLIP_SAMPLES:
            clip = audio_buffer[p - CLIP_SAMPLES:p].copy()
        else:
            clip = np.concatenate([
                audio_buffer[buffer_size - (CLIP_SAMPLES - p):],
                audio_buffer[:p]
            ]).copy()

        features = compute_features(clip).reshape(1, -1)
        prob = clf.predict_proba(features)[0][1]

        if prob > 0.7:
            last_trigger[0] = now
            print(f"  DART! (prob={prob:.2f}, rms={rms:.4f})")

    stream = sd.InputStream(
        device=device, channels=1, samplerate=SAMPLE_RATE,
        blocksize=2048, callback=audio_callback,
    )
    stream.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
    print("\nDone.")


# ---------------------------------------------------------------------------
# Dart trigger class (for use in collect.py)
# ---------------------------------------------------------------------------

class DartAudioTrigger:
    """ML-based dart impact trigger for use in the collection loop."""

    def __init__(self, device=None, cooldown=1.0, prob_threshold=0.7):
        self.cooldown = cooldown
        self.prob_threshold = prob_threshold
        self.triggered = False
        self._last_trigger_time = 0.0
        self._stream = None
        self._device = device
        self._clf = None
        self._current_rms = 0.0
        self._current_prob = 0.0
        self._peak_rms = 0.0
        self._baseline_rms = 0.01
        self._rms_history = []
        self._prob_history = []
        self._history_max = 200

        # Rolling audio buffer
        self._buffer_size = int(SAMPLE_RATE * 2.0)
        self._audio_buffer = np.zeros(self._buffer_size, dtype=np.float32)
        self._buffer_pos = 0

    def start(self):
        """Load model and start listening. Returns True on success."""
        import sounddevice as sd

        if not MODEL_PATH.exists():
            print("WARNING: No dart audio model found — audio trigger disabled")
            print(f"  Run 'python audio_trigger.py --record' then '--train' to create one")
            return False

        import joblib
        self._clf = joblib.load(MODEL_PATH)

        device = self._device
        if device is None:
            device = find_emeet_device()
        if device is None:
            print("WARNING: eMeet mic not found — audio trigger disabled")
            return False

        device_info = sd.query_devices(device)
        print(f"Audio trigger (ML): {device_info['name']} (device {device})")
        print(f"  Prob threshold: {self.prob_threshold:.2f}, cooldown: {self.cooldown}s")

        def callback(indata, frames, time_info, status):
            mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            n = len(mono)
            pos = self._buffer_pos

            if pos + n <= self._buffer_size:
                self._audio_buffer[pos:pos + n] = mono
            else:
                overflow = (pos + n) - self._buffer_size
                self._audio_buffer[pos:] = mono[:n - overflow]
                self._audio_buffer[:overflow] = mono[n - overflow:]
            self._buffer_pos = (pos + n) % self._buffer_size

            # RMS tracking
            rms = float(np.sqrt(np.mean(mono ** 2)))
            self._current_rms = rms
            if rms > self._peak_rms:
                self._peak_rms = rms
            else:
                self._peak_rms = max(rms, self._peak_rms * 0.995)
            if rms < self._baseline_rms * 3:
                self._baseline_rms = 0.99 * self._baseline_rms + 0.01 * rms
            self._rms_history.append(rms)
            if len(self._rms_history) > self._history_max:
                self._rms_history.pop(0)

            # Skip classification if very quiet
            now = time.monotonic()
            if rms < 0.005 or now - self._last_trigger_time < self.cooldown:
                self._current_prob = 0.0
                self._prob_history.append(0.0)
                if len(self._prob_history) > self._history_max:
                    self._prob_history.pop(0)
                return

            # Extract clip and classify
            p = self._buffer_pos
            if p >= CLIP_SAMPLES:
                clip = self._audio_buffer[p - CLIP_SAMPLES:p].copy()
            else:
                clip = np.concatenate([
                    self._audio_buffer[self._buffer_size - (CLIP_SAMPLES - p):],
                    self._audio_buffer[:p]
                ]).copy()

            features = compute_features(clip).reshape(1, -1)
            prob = float(self._clf.predict_proba(features)[0][1])
            self._current_prob = prob

            self._prob_history.append(prob)
            if len(self._prob_history) > self._history_max:
                self._prob_history.pop(0)

            if prob > self.prob_threshold:
                self.triggered = True
                self._last_trigger_time = now

        try:
            self._stream = sd.InputStream(
                device=device, channels=1, samplerate=SAMPLE_RATE,
                blocksize=2048, callback=callback,
            )
            self._stream.start()
            return True
        except Exception as e:
            print(f"WARNING: Could not open audio device: {e}")
            return False

    def check_and_reset(self):
        if self.triggered:
            self.triggered = False
            return True
        return False

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def current_rms(self):
        return self._current_rms

    @property
    def current_prob(self):
        return self._current_prob


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dart impact audio detection")
    parser.add_argument("--record", action="store_true", help="Record labeled audio samples")
    parser.add_argument("--train", action="store_true", help="Train classifier on samples")
    parser.add_argument("--test", action="store_true", help="Test classifier live")
    parser.add_argument("--device", type=int, default=None, help="Audio device index")
    parser.add_argument("--rms-threshold", type=float, default=0.01,
                        help="RMS threshold for auto-detection during recording (default: 0.01)")
    args = parser.parse_args()

    if args.record:
        record_samples(device=args.device, rms_threshold=args.rms_threshold)
    elif args.train:
        train_classifier()
    elif args.test:
        test_live(device=args.device)
    else:
        parser.print_help()
