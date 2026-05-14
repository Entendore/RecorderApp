"""
Voice Recorder & Modulator App — with LFO, Volume Control & Auto-Leveling
==========================================================================
Requirements:
    pip install kivy numpy sounddevice scipy
"""

import time
import threading
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import (
    resample as scipy_resample, butter, filtfilt, lfilter, lfilter_zi
)
from datetime import datetime
from pathlib import Path

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.spinner import Spinner
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle, Line
from kivy.metrics import dp, sp
from kivy.properties import ListProperty, BooleanProperty


# ════════════════════════════════════════════════════════════
#  AUDIO UTILITIES
# ════════════════════════════════════════════════════════════

def rms_level(data):
    if len(data) == 0:
        return 0.0
    return float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))


def peak_level(data):
    if len(data) == 0:
        return 0.0
    return float(np.max(np.abs(data.astype(np.float64))))


def normalize_rms(data, target_rms=0.08, min_gain=0.5, max_gain=10.0):
    data = data.astype(np.float64)
    current_rms = rms_level(data)
    if current_rms < 1e-7:
        return data.astype(np.float32)
    gain = np.clip(target_rms / current_rms, min_gain, max_gain)
    return (data * gain).astype(np.float32)


def soft_limiter(data, threshold=0.92, knee=0.08):
    data = data.astype(np.float64)
    abs_data = np.abs(data)
    over = abs_data - (threshold - knee)
    gain = np.ones_like(data)
    mask = over > 0
    if np.any(mask):
        ratio = np.clip(over[mask] / knee, 0, 1)
        target_gain = threshold / np.maximum(abs_data[mask], 1e-7)
        gain[mask] = 1.0 - ratio * (1.0 - target_gain)
    return np.clip(data * gain, -1.0, 1.0).astype(np.float32)


def apply_volume(data, volume_db):
    if abs(volume_db) < 0.01:
        return data.copy().astype(np.float32)
    return (data.astype(np.float64) * (10.0 ** (volume_db / 20.0))).astype(np.float32)


def full_process_pipeline(data, sr, volume_db=0.0, target_rms=0.08):
    result = normalize_rms(data, target_rms=target_rms)
    result = apply_volume(result, volume_db)
    result = soft_limiter(result, threshold=0.92, knee=0.10)
    result = np.clip(result, -1.0, 1.0).astype(np.float32)
    return result, rms_level(result), peak_level(result)


# ════════════════════════════════════════════════════════════
#  LFO (Low-Frequency Oscillator)
# ════════════════════════════════════════════════════════════

def generate_lfo(length, sr, rate, waveform='Sine'):
    t = np.arange(length, dtype=np.float64) / sr
    phase = (2.0 * np.pi * rate * t) % (2.0 * np.pi)

    if waveform == 'Triangle':
        x = phase / (2.0 * np.pi)
        mod = 2.0 * np.abs(2.0 * (x - np.floor(x + 0.5))) - 1.0
    elif waveform == 'Saw':
        x = phase / (2.0 * np.pi)
        mod = 2.0 * (x - np.floor(x + 0.5))
    elif waveform == 'Square':
        mod = np.sign(np.sin(phase))
    else:
        mod = np.sin(phase)

    return mod.astype(np.float32)


def apply_lfo_tremolo(data, sr, rate, depth_pct, waveform):
    depth = depth_pct / 100.0
    lfo = generate_lfo(len(data), sr, rate, waveform)
    gain = 1.0 - depth + depth * (0.5 + 0.5 * lfo)
    return (data.astype(np.float64) * gain).astype(np.float32)


def apply_lfo_vibrato(data, sr, rate, depth_pct, waveform):
    max_delay_sec = (depth_pct / 100.0) * 0.010
    lfo = generate_lfo(len(data), sr, rate, waveform)
    delay_samples = lfo * max_delay_sec * sr
    indices = np.arange(len(data), dtype=np.float64) + delay_samples
    indices = np.clip(indices, 0, len(data) - 1)
    return np.interp(indices, np.arange(len(data)), data).astype(np.float32)


def apply_lfo_wah(data, sr, rate, depth_pct, waveform):
    lfo = generate_lfo(len(data), sr, rate, waveform)
    chunk_size = 512
    output = np.zeros(len(data), dtype=np.float32)
    zi = None

    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        if len(chunk) < 3:
            output[i:i + len(chunk)] = chunk
            break

        chunk_lfo = lfo[i:min(i + chunk_size, len(lfo))]
        lfo_val = float(np.mean(chunk_lfo))
        freq = 300.0 + (lfo_val + 1.0) * 1350.0
        freq = max(min(freq, sr / 2.0 - 100), 100)

        nyq = sr / 2.0
        low = max((freq * 0.5) / nyq, 0.01)
        high = min((freq * 1.5) / nyq, 0.99)
        if low >= high:
            low = high - 0.01

        b, a = butter(2, [low, high], btype='band')

        if zi is None:
            zi = lfilter_zi(b, a) * chunk[0]

        filtered_chunk, zi = lfilter(b, a, chunk, zi=zi)
        output[i:i + len(chunk)] = filtered_chunk

    return output


# ════════════════════════════════════════════════════════════
#  STATIC EFFECTS
# ════════════════════════════════════════════════════════════

def apply_echo(data, sr, delay=0.15, decay=0.5, n_echoes=4):
    delay_samples = int(delay * sr)
    output = data.copy().astype(np.float64)
    for i in range(1, n_echoes + 1):
        offset = i * delay_samples
        decayed = data * (decay ** i)
        if offset < len(output):
            end = min(offset + len(decayed), len(output))
            output[offset:end] += decayed[:end - offset]
    return output.astype(np.float32)


def apply_robot(data, sr, freq=50.0):
    t = np.arange(len(data), dtype=np.float64) / sr
    return (data.astype(np.float64) * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def apply_reverb(data, sr, room_size=0.8, damping=0.5):
    output = data.copy().astype(np.float64)
    for delay, gain in zip([0.023, 0.029, 0.037, 0.041, 0.053, 0.067],
                           [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]):
        ds = int(delay * sr * room_size)
        g = gain * (1.0 - damping)
        if ds > 0 and ds < len(output):
            end = min(ds + len(data), len(output))
            output[ds:end] += data[:end - ds] * g
    return output.astype(np.float32)


def apply_pitch_shift(data, sr, semitones):
    if semitones == 0:
        return data.copy().astype(np.float32)
    new_length = max(100, int(len(data) / (2.0 ** (semitones / 12.0))))
    return scipy_resample(data.astype(np.float64), new_length).astype(np.float32)


def apply_speed_change(data, sr, speed_factor):
    if abs(speed_factor - 1.0) < 0.01:
        return data.copy().astype(np.float32)
    new_length = max(100, int(len(data) / speed_factor))
    return scipy_resample(data.astype(np.float64), new_length).astype(np.float32)


def apply_deep_voice(data, sr):
    return apply_echo(apply_pitch_shift(data, sr, -5), sr, delay=0.05, decay=0.3, n_echoes=2)


def apply_chipmunk(data, sr):
    return apply_pitch_shift(data, sr, 8)


def apply_reverse(data, sr):
    return data[::-1].copy().astype(np.float32)


def apply_tremolo(data, sr, rate=6.0, depth=0.7):
    t = np.arange(len(data), dtype=np.float64) / sr
    lfo = 1.0 - depth * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t))
    return (data.astype(np.float64) * lfo).astype(np.float32)


def apply_telephone(data, sr):
    nyq = sr / 2.0
    low, high = max(300.0 / nyq, 0.01), min(3400.0 / nyq, 0.99)
    if low >= high:
        low = high - 0.01
    b, a = butter(5, [low, high], btype='band')
    output = filtfilt(b, a, data.astype(np.float64))
    return (np.clip(output * 2.5, -1.0, 1.0) * 0.5).astype(np.float32)


def apply_whisper(data, sr, noise_amount=0.15):
    noise = np.random.randn(len(data)).astype(np.float32) * noise_amount
    frame_size = 1024
    envelope = np.zeros(len(data))
    for i in range(0, len(data) - frame_size, frame_size):
        rms = np.sqrt(np.mean(data[i:i + frame_size] ** 2))
        envelope[i:i + frame_size] = rms
    mx = np.max(envelope)
    if mx > 0:
        envelope /= mx
    output = data.astype(np.float64) * 0.4 + (noise * envelope) * 0.8
    return output.astype(np.float32)


# ════════════════════════════════════════════════════════════
#  KV UI DEFINITION
# ════════════════════════════════════════════════════════════

KV = '''
#:import sp kivy.metrics.sp
#:import dp kivy.metrics.dp

<WaveformWidget>:
    size_hint: 1, 1
    pos_hint: {'x': 0, 'top': 1}

<FxButton@ToggleButton>:
    group: 'fx'
    background_normal: ''
    background_down: ''
    background_color: (0.18, 0.50, 0.88, 1) if self.state == 'down' else (0.14, 0.14, 0.24, 1)
    color: (1, 1, 1, 1) if self.state == 'down' else (0.65, 0.72, 0.88, 1)
    font_size: sp(12)
    bold: True if self.state == 'down' else False
    size_hint_y: None
    height: dp(38)

<MainLayout>:
    orientation: 'vertical'
    padding: dp(8)
    spacing: dp(3)

    canvas.before:
        Color:
            rgba: 0.07, 0.07, 0.12, 1
        Rectangle:
            pos: self.pos
            size: self.size

    # ── Title ───────────────────────────────────────────
    Label:
        text: 'Voice Recorder & Modulator'
        font_size: sp(19)
        color: 0.30, 0.70, 1.0, 1
        size_hint_y: None
        height: dp(32)
        bold: True

    # ── Waveform ───────────────────────────────────────
    FloatLayout:
        size_hint_y: None
        height: dp(90)
        canvas.before:
            Color:
                rgba: 0.05, 0.05, 0.09, 1
            Rectangle:
                pos: self.pos
                size: self.size
        WaveformWidget:
            id: waveform
        Label:
            id: waveform_placeholder
            text: 'Record audio to see waveform'
            color: 0.35, 0.35, 0.45, 1
            font_size: sp(12)

    # ── Level Meters ───────────────────────────────────
    BoxLayout:
        size_hint_y: None
        height: dp(46)
        orientation: 'vertical'
        spacing: dp(1)

        BoxLayout:
            size_hint_y: None
            height: dp(20)
            Label:
                text: 'In'
                color: 0.50, 0.50, 0.60, 1
                font_size: sp(9)
                size_hint_x: 0.06
            ProgressBar:
                id: input_meter
                max: 1.0
                value: 0
            Label:
                id: input_db
                text: '-inf'
                color: 0.50, 0.70, 0.50, 1
                font_size: sp(9)
                size_hint_x: 0.14

        BoxLayout:
            size_hint_y: None
            height: dp(20)
            Label:
                text: 'Out'
                color: 0.50, 0.50, 0.60, 1
                font_size: sp(9)
                size_hint_x: 0.06
            ProgressBar:
                id: output_meter
                max: 1.0
                value: 0
            Label:
                id: output_db
                text: '-inf'
                color: 0.50, 0.70, 0.50, 1
                font_size: sp(9)
                size_hint_x: 0.14

    # ── Rec Controls ───────────────────────────────────
    BoxLayout:
        size_hint_y: None
        height: dp(42)
        spacing: dp(5)

        Button:
            id: btn_record
            text: 'REC'
            background_normal: ''
            background_color: 0.88, 0.18, 0.18, 1
            color: 1, 1, 1, 1
            font_size: sp(14)
            bold: True
            on_press: root.start_recording()

        Button:
            id: btn_stop
            text: 'STOP'
            background_normal: ''
            background_color: 0.45, 0.45, 0.50, 1
            color: 1, 1, 1, 1
            font_size: sp(13)
            bold: True
            disabled: True
            on_press: root.stop_recording()

        Button:
            id: btn_play_orig
            text: 'PLAY ORIG'
            background_normal: ''
            background_color: 0.15, 0.50, 0.80, 1
            color: 1, 1, 1, 1
            font_size: sp(12)
            disabled: True
            on_press: root.play_original()

        Button:
            id: btn_stop_play
            text: 'X'
            background_normal: ''
            background_color: 0.55, 0.30, 0.20, 1
            color: 1, 1, 1, 1
            font_size: sp(15)
            size_hint_x: 0.12
            on_press: root.stop_playback()

    # ── Status ─────────────────────────────────────────
    BoxLayout:
        size_hint_y: None
        height: dp(18)
        Label:
            id: lbl_duration
            text: '00:00'
            color: 0.50, 0.78, 1.0, 1
            font_size: sp(11)
            size_hint_x: 0.25
        Label:
            id: lbl_status
            text: 'Ready'
            color: 0.50, 0.75, 0.50, 1
            font_size: sp(11)
            size_hint_x: 0.75
            halign: 'left'

    # ── Separator ──────────────────────────────────────
    Widget:
        size_hint_y: None
        height: dp(2)

    # ── Scrollable Area ────────────────────────────────
    ScrollView:
        do_scroll_x: False
        do_scroll_y: True
        bar_width: dp(4)

        BoxLayout:
            orientation: 'vertical'
            size_hint_y: None
            height: self.minimum_height
            spacing: dp(2)
            padding: dp(2)

            # ── Volume ──────────────────────────────────
            BoxLayout:
                size_hint_y: None
                height: dp(58)
                orientation: 'vertical'
                padding: dp(6), dp(2)
                canvas.before:
                    Color:
                        rgba: 0.10, 0.12, 0.20, 1
                    Rectangle:
                        pos: self.pos
                        size: self.size

                BoxLayout:
                    size_hint_y: None
                    height: dp(18)
                    Label:
                        text: 'VOLUME'
                        color: 0.90, 0.75, 0.20, 1
                        font_size: sp(11)
                        bold: True
                        size_hint_x: 0.28
                        halign: 'left'
                    Label:
                        id: volume_db_label
                        text: '0 dB'
                        color: 0.90, 0.85, 0.40, 1
                        font_size: sp(11)
                        bold: True
                        size_hint_x: 0.22
                    Label:
                        id: volume_hint
                        text: ''
                        color: 0.50, 0.80, 0.50, 1
                        font_size: sp(9)
                        size_hint_x: 0.50
                        halign: 'left'

                BoxLayout:
                    size_hint_y: None
                    height: dp(24)
                    Label:
                        text: 'V'
                        font_size: sp(12)
                        size_hint_x: 0.05
                    Slider:
                        id: volume_slider
                        min: -20
                        max: 20
                        value: 0
                        step: 1
                        size_hint_x: 0.82
                    Label:
                        text: 'V'
                        font_size: sp(12)
                        size_hint_x: 0.05
                    ToggleButton:
                        id: btn_mute
                        text: 'M'
                        group: 'mute'
                        background_normal: ''
                        background_down: ''
                        background_color: (0.85, 0.20, 0.20, 1) if self.state == 'down' else (0.30, 0.30, 0.35, 1)
                        color: 1, 1, 1, 1
                        font_size: sp(9)
                        bold: True
                        size_hint_x: 0.08

            # ── LFO ────────────────────────────────────
            BoxLayout:
                size_hint_y: None
                height: dp(130)
                orientation: 'vertical'
                padding: dp(6), dp(2)
                canvas.before:
                    Color:
                        rgba: 0.13, 0.08, 0.19, 1
                    Rectangle:
                        pos: self.pos
                        size: self.size

                BoxLayout:
                    size_hint_y: None
                    height: dp(20)
                    Label:
                        text: 'LFO'
                        color: 0.85, 0.55, 1.0, 1
                        font_size: sp(12)
                        bold: True
                        size_hint_x: 0.15
                        halign: 'left'
                    ToggleButton:
                        id: lfo_enable
                        text: 'OFF' if self.state == 'normal' else 'ON'
                        group: 'lfo_toggle'
                        background_normal: ''
                        background_down: ''
                        background_color: (0.65, 0.25, 0.90, 1) if self.state == 'down' else (0.30, 0.30, 0.35, 1)
                        color: 1, 1, 1, 1
                        font_size: sp(10)
                        bold: True
                        size_hint_x: 0.12
                    Spinner:
                        id: lfo_target
                        text: 'Tremolo'
                        values: ['Tremolo', 'Vibrato', 'Wah-Wah']
                        size_hint_x: 0.25
                        background_normal: ''
                        background_color: 0.22, 0.15, 0.30, 1
                        color: 0.95, 0.85, 1.0, 1
                        font_size: sp(10)
                    Spinner:
                        id: lfo_waveform
                        text: 'Sine'
                        values: ['Sine', 'Triangle', 'Saw', 'Square']
                        size_hint_x: 0.22
                        background_normal: ''
                        background_color: 0.22, 0.15, 0.30, 1
                        color: 0.95, 0.85, 1.0, 1
                        font_size: sp(10)
                    Label:
                        id: lfo_hint
                        text: 'Off'
                        color: 0.75, 0.65, 0.90, 1
                        font_size: sp(8)
                        size_hint_x: 0.26

                BoxLayout:
                    size_hint_y: None
                    height: dp(22)
                    Label:
                        text: 'Rate'
                        color: 0.65, 0.50, 0.80, 1
                        font_size: sp(9)
                        size_hint_x: 0.08
                    Slider:
                        id: lfo_rate
                        min: 0.1
                        max: 20.0
                        value: 5.0
                        step: 0.1
                        size_hint_x: 0.72
                    Label:
                        id: lfo_rate_val
                        text: '5.0 Hz'
                        color: 0.85, 0.70, 1.0, 1
                        font_size: sp(9)
                        size_hint_x: 0.20

                BoxLayout:
                    size_hint_y: None
                    height: dp(22)
                    Label:
                        text: 'Depth'
                        color: 0.65, 0.50, 0.80, 1
                        font_size: sp(9)
                        size_hint_x: 0.08
                    Slider:
                        id: lfo_depth
                        min: 0
                        max: 100
                        value: 70
                        step: 1
                        size_hint_x: 0.72
                    Label:
                        id: lfo_depth_val
                        text: '70%'
                        color: 0.85, 0.70, 1.0, 1
                        font_size: sp(9)
                        size_hint_x: 0.20

            # ── Effects ─────────────────────────────────
            Label:
                text: 'VOICE EFFECTS'
                font_size: sp(13)
                color: 0.30, 0.70, 1.0, 1
                size_hint_y: None
                height: dp(22)
                bold: True

            GridLayout:
                cols: 3
                size_hint_y: None
                height: self.minimum_height
                spacing: dp(3)
                row_default_height: dp(38)
                row_force_default: True

                FxButton:
                    text: 'Deep'
                    on_press: root.select_effect('deep')
                FxButton:
                    text: 'Chipmunk'
                    on_press: root.select_effect('chipmunk')
                FxButton:
                    text: 'Robot'
                    on_press: root.select_effect('robot')
                FxButton:
                    text: 'Echo'
                    on_press: root.select_effect('echo')
                FxButton:
                    text: 'Reverb'
                    on_press: root.select_effect('reverb')
                FxButton:
                    text: 'Reverse'
                    on_press: root.select_effect('reverse')
                FxButton:
                    text: 'Tremolo'
                    on_press: root.select_effect('tremolo')
                FxButton:
                    text: 'Phone'
                    on_press: root.select_effect('telephone')
                FxButton:
                    text: 'Whisper'
                    on_press: root.select_effect('whisper')

            # ── Pitch ──────────────────────────────────
            BoxLayout:
                size_hint_y: None
                height: dp(50)
                orientation: 'vertical'
                padding: dp(6), dp(1)
                canvas.before:
                    Color:
                        rgba: 0.09, 0.09, 0.16, 1
                    Rectangle:
                        pos: self.pos
                        size: self.size

                Label:
                    text: 'Pitch Shift'
                    color: 0.55, 0.78, 1.0, 1
                    font_size: sp(10)
                    bold: True
                    size_hint_y: None
                    height: dp(14)

                BoxLayout:
                    size_hint_y: None
                    height: dp(22)
                    Label:
                        text: '-12'
                        color: 0.45, 0.45, 0.55, 1
                        font_size: sp(9)
                        size_hint_x: 0.07
                    Slider:
                        id: pitch_slider
                        min: -12
                        max: 12
                        value: 0
                        step: 1
                        size_hint_x: 0.76
                    Label:
                        text: '+12'
                        color: 0.45, 0.45, 0.55, 1
                        font_size: sp(9)
                        size_hint_x: 0.07
                    Label:
                        id: pitch_val
                        text: '0 st'
                        color: 0.7, 0.9, 1.0, 1
                        font_size: sp(10)
                        bold: True
                        size_hint_x: 0.10

            # ── Speed ──────────────────────────────────
            BoxLayout:
                size_hint_y: None
                height: dp(50)
                orientation: 'vertical'
                padding: dp(6), dp(1)
                canvas.before:
                    Color:
                        rgba: 0.09, 0.09, 0.16, 1
                    Rectangle:
                        pos: self.pos
                        size: self.size

                Label:
                    text: 'Speed'
                    color: 0.55, 0.78, 1.0, 1
                    font_size: sp(10)
                    bold: True
                    size_hint_y: None
                    height: dp(14)

                BoxLayout:
                    size_hint_y: None
                    height: dp(22)
                    Label:
                        text: '0.5x'
                        color: 0.45, 0.45, 0.55, 1
                        font_size: sp(9)
                        size_hint_x: 0.07
                    Slider:
                        id: speed_slider
                        min: 0.5
                        max: 2.0
                        value: 1.0
                        step: 0.1
                        size_hint_x: 0.76
                    Label:
                        text: '2.0x'
                        color: 0.45, 0.45, 0.55, 1
                        font_size: sp(9)
                        size_hint_x: 0.07
                    Label:
                        id: speed_val
                        text: '1.0x'
                        color: 0.7, 0.9, 1.0, 1
                        font_size: sp(10)
                        bold: True
                        size_hint_x: 0.10

            # ── Echo Settings ──────────────────────────
            BoxLayout:
                size_hint_y: None
                height: dp(72)
                orientation: 'vertical'
                padding: dp(6), dp(1)
                canvas.before:
                    Color:
                        rgba: 0.09, 0.09, 0.16, 1
                    Rectangle:
                        pos: self.pos
                        size: self.size

                Label:
                    text: 'Echo Settings'
                    color: 0.55, 0.78, 1.0, 1
                    font_size: sp(10)
                    bold: True
                    size_hint_y: None
                    height: dp(14)

                BoxLayout:
                    size_hint_y: None
                    height: dp(22)
                    Label:
                        text: 'Delay'
                        color: 0.50, 0.50, 0.60, 1
                        font_size: sp(9)
                        size_hint_x: 0.10
                    Slider:
                        id: echo_delay_slider
                        min: 0.05
                        max: 0.50
                        value: 0.15
                        step: 0.01
                        size_hint_x: 0.68
                    Label:
                        id: echo_delay_val
                        text: '0.15s'
                        color: 0.7, 0.9, 1.0, 1
                        font_size: sp(9)
                        size_hint_x: 0.22

                BoxLayout:
                    size_hint_y: None
                    height: dp(22)
                    Label:
                        text: 'Decay'
                        color: 0.50, 0.50, 0.60, 1
                        font_size: sp(9)
                        size_hint_x: 0.10
                    Slider:
                        id: echo_decay_slider
                        min: 0.10
                        max: 0.90
                        value: 0.50
                        step: 0.05
                        size_hint_x: 0.68
                    Label:
                        id: echo_decay_val
                        text: '0.50'
                        color: 0.7, 0.9, 1.0, 1
                        font_size: sp(9)
                        size_hint_x: 0.22

            # ── Spacer ─────────────────────────────────
            Widget:
                size_hint_y: None
                height: dp(4)

            # ── Action Buttons ─────────────────────────
            BoxLayout:
                size_hint_y: None
                height: dp(44)
                spacing: dp(5)

                Button:
                    id: btn_preview
                    text: 'Preview'
                    background_normal: ''
                    background_color: 0.15, 0.55, 0.85, 1
                    color: 1, 1, 1, 1
                    font_size: sp(13)
                    bold: True
                    disabled: True
                    on_press: root.preview_modulated()

                Button:
                    id: btn_save
                    text: 'Save'
                    background_normal: ''
                    background_color: 0.18, 0.60, 0.28, 1
                    color: 1, 1, 1, 1
                    font_size: sp(13)
                    bold: True
                    disabled: True
                    on_press: root.save_audio()

                Button:
                    id: btn_reset
                    text: 'Reset'
                    background_normal: ''
                    background_color: 0.55, 0.38, 0.18, 1
                    color: 1, 1, 1, 1
                    font_size: sp(13)
                    bold: True
                    disabled: True
                    on_press: root.reset_modulation()
'''


# ════════════════════════════════════════════════════════════
#  CUSTOM WAVEFORM WIDGET
# ════════════════════════════════════════════════════════════

class WaveformWidget(Widget):
    audio_data = ListProperty([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(pos=self._redraw, size=self._redraw, audio_data=self._redraw)

    def set_data(self, data):
        if data is not None and len(data) > 0:
            max_pts = 400
            if len(data) > max_pts:
                indices = np.linspace(0, len(data) - 1, max_pts).astype(int)
                display = data[indices]
            else:
                display = data
            mx = np.max(np.abs(display))
            if mx > 0:
                display = display / mx
            self.audio_data = display.tolist()
        else:
            self.audio_data = []

    def _redraw(self, *args):
        self.canvas.clear()
        with self.canvas:
            Color(0.05, 0.05, 0.09, 1)
            Rectangle(pos=self.pos, size=self.size)
            Color(0.20, 0.28, 0.38, 0.4)
            mid_y = self.y + self.height / 2
            Line(points=[self.x, mid_y, self.x + self.width, mid_y], width=0.5)

            data = self.audio_data
            if not data or len(data) < 2:
                return

            w, h, n = self.width, self.height, len(data)
            bar_w = max(w / n, 1.0)

            for i, val in enumerate(data):
                x = self.x + (i / n) * w
                half = abs(val) * h * 0.42
                intensity = min(abs(val), 1.0)
                Color(0.10 + 0.20 * intensity,
                      0.35 + 0.35 * intensity,
                      0.75 + 0.25 * intensity, 0.50)
                Rectangle(pos=(x, mid_y - half), size=(bar_w, half * 2))

            Color(0.30, 0.70, 1.0, 0.85)
            pts = []
            for i, val in enumerate(data):
                pts.extend([self.x + (i / n) * w, mid_y + val * h * 0.42])
            if len(pts) >= 4:
                Line(points=pts, width=1.2)


# ════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ════════════════════════════════════════════════════════════

class MainLayout(BoxLayout):
    is_recording = BooleanProperty(False)

    EFFECT_MAP = {
        'deep':      (apply_deep_voice,  False),
        'chipmunk':  (apply_chipmunk,     False),
        'robot':     (apply_robot,        False),
        'echo':      (apply_echo,         True),
        'reverb':    (apply_reverb,       False),
        'reverse':   (apply_reverse,      False),
        'tremolo':   (apply_tremolo,      False),
        'telephone': (apply_telephone,    False),
        'whisper':   (apply_whisper,      False),
    }

    TARGET_RMS = 0.08

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = 44100
        self.recorded_data = None
        self.modulated_data = None
        self.active_effect = None
        self.record_buffer = []
        self.recording_start = 0
        self.current_input_level = 0.0
        self._timer_event = None
        self._level_event = None
        Clock.schedule_once(self._bind_sliders, 0)

    def _bind_sliders(self, dt):
        self.ids.volume_slider.bind(value=self._on_volume)
        self.ids.pitch_slider.bind(value=self._on_pitch)
        self.ids.speed_slider.bind(value=self._on_speed)
        self.ids.echo_delay_slider.bind(value=self._on_echo_delay)
        self.ids.echo_decay_slider.bind(value=self._on_echo_decay)
        self.ids.lfo_rate.bind(value=self._on_lfo_rate)
        self.ids.lfo_depth.bind(value=self._on_lfo_depth)
        self.ids.lfo_target.bind(text=self._on_lfo_change)
        self.ids.lfo_waveform.bind(text=self._on_lfo_change)
        self.ids.lfo_enable.bind(state=self._on_lfo_change)

        self._on_volume(None, 0)
        self._on_pitch(None, 0)
        self._on_speed(None, 1.0)
        self._on_echo_delay(None, 0.15)
        self._on_echo_decay(None, 0.50)
        self._on_lfo_rate(None, 5.0)
        self._on_lfo_depth(None, 70)
        self._on_lfo_change(None, None)

    def _on_volume(self, inst, val):
        v = int(val)
        self.ids.volume_db_label.text = f'{v:+d} dB' if v else '0 dB'
        if v <= -15:
            h, c = 'Very quiet', (0.8, 0.3, 0.3, 1)
        elif v <= -5:
            h, c = 'Soft', (0.6, 0.7, 0.4, 1)
        elif v <= 5:
            h, c = 'Normal', (0.3, 0.8, 0.4, 1)
        elif v <= 12:
            h, c = 'Loud', (0.9, 0.7, 0.2, 1)
        else:
            h, c = 'Very loud', (0.9, 0.3, 0.3, 1)
        self.ids.volume_hint.text = h
        self.ids.volume_hint.color = c
        if self.modulated_data is not None:
            self._update_output_meter(self.modulated_data)

    def _on_pitch(self, inst, val):
        v = int(val)
        self.ids.pitch_val.text = f'{v:+d} st' if v else '0 st'

    def _on_speed(self, inst, val):
        self.ids.speed_val.text = f'{val:.1f}x'

    def _on_echo_delay(self, inst, val):
        self.ids.echo_delay_val.text = f'{val:.2f}s'

    def _on_echo_decay(self, inst, val):
        self.ids.echo_decay_val.text = f'{val:.2f}'

    def _on_lfo_rate(self, inst, val):
        self.ids.lfo_rate_val.text = f'{val:.1f} Hz'

    def _on_lfo_depth(self, inst, val):
        self.ids.lfo_depth_val.text = f'{int(val)}%'

    def _on_lfo_change(self, inst, val):
        if self.ids.lfo_enable.state == 'down':
            target = self.ids.lfo_target.text
            wave = self.ids.lfo_waveform.text
            self.ids.lfo_hint.text = f'{wave} -> {target}'
        else:
            self.ids.lfo_hint.text = 'Off'

    def _set_status(self, text, color=(0.5, 0.75, 0.5, 1)):
        self.ids.lbl_status.text = text
        self.ids.lbl_status.color = color

    def _update_output_meter(self, data):
        if data is None or len(data) == 0:
            self.ids.output_meter.value = 0
            self.ids.output_db.text = '-inf'
            return
        vol_db = int(self.ids.volume_slider.value)
        _, final_rms, final_peak = full_process_pipeline(
            data, self.sample_rate, volume_db=vol_db, target_rms=self.TARGET_RMS
        )
        self.ids.output_meter.value = final_peak
        if final_rms > 1e-7:
            db_val = 20.0 * np.log10(final_rms)
            self.ids.output_db.text = f'{db_val:.0f}dB'
            if final_peak < 0.85:
                self.ids.output_db.color = (0.3, 0.8, 0.4, 1)
            elif final_peak < 0.95:
                self.ids.output_db.color = (0.9, 0.8, 0.2, 1)
            else:
                self.ids.output_db.color = (0.9, 0.3, 0.3, 1)
        else:
            self.ids.output_db.text = '-inf'
            self.ids.output_db.color = (0.5, 0.5, 0.5, 1)

    def _show_popup(self, title, message):
        box = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(15))
        box.add_widget(Label(
            text=message, color=(0.85, 0.85, 0.92, 1),
            font_size=sp(14), halign='center'
        ))
        btn = Button(
            text='OK', size_hint_y=None, height=dp(42),
            background_normal='', background_color=(0.20, 0.50, 0.85, 1),
            color=(1, 1, 1, 1), font_size=sp(14), bold=True
        )
        box.add_widget(btn)
        popup = Popup(
            title=title, content=box, size_hint=(0.72, 0.32),
            background_color=(0.10, 0.10, 0.18, 0.96),
            separator_color=(0.20, 0.50, 0.85, 1),
            title_color=(0.85, 0.85, 0.95, 1)
        )
        btn.bind(on_press=popup.dismiss)
        popup.open()

    # ── Recording ──────────────────────────────────────
    def start_recording(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.record_buffer = []
        self.current_input_level = 0.0
        self.recording_start = time.time()

        self.ids.btn_record.disabled = True
        self.ids.btn_stop.disabled = False
        self.ids.btn_play_orig.disabled = True
        self.ids.btn_preview.disabled = True
        self.ids.btn_save.disabled = True
        self.ids.btn_reset.disabled = True
        self._set_status('Recording...', (0.90, 0.30, 0.30, 1))
        self.ids.waveform_placeholder.opacity = 0

        threading.Thread(target=self._record_thread, daemon=True).start()
        self._timer_event = Clock.schedule_interval(self._tick_timer, 0.1)
        self._level_event = Clock.schedule_interval(self._tick_level, 0.05)

    def _record_callback(self, indata, frames, time_info, status):
        self.record_buffer.append(indata.copy())
        self.current_input_level = float(np.max(np.abs(indata)))

    def _record_thread(self):
        try:
            with sd.InputStream(
                samplerate=self.sample_rate, channels=1,
                dtype='float32', callback=self._record_callback
            ):
                while self.is_recording:
                    time.sleep(0.05)
        except Exception as e:
            Clock.schedule_once(
                lambda dt: self._set_status(f'Rec error: {e}', (0.9, 0.2, 0.2, 1))
            )
            self.is_recording = False

    def _tick_timer(self, dt):
        m, s = divmod(int(time.time() - self.recording_start), 60)
        self.ids.lbl_duration.text = f'{m:02d}:{s:02d}'

    def _tick_level(self, dt):
        level = self.current_input_level
        self.ids.input_meter.value = level
        if level > 1e-7:
            self.ids.input_db.text = f'{20 * np.log10(level):.0f}dB'
        else:
            self.ids.input_db.text = '-inf'

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False

        if self._timer_event:
            self._timer_event.cancel()
        if self._level_event:
            self._level_event.cancel()
        self.ids.input_meter.value = 0
        self.current_input_level = 0.0

        if self.record_buffer:
            raw = np.concatenate(self.record_buffer, axis=0).flatten()
            self.recorded_data, _, _ = full_process_pipeline(
                raw, self.sample_rate, volume_db=0, target_rms=self.TARGET_RMS
            )
            self.modulated_data = None
            self.active_effect = None
            duration = len(self.recorded_data) / self.sample_rate
            self._set_status(f'Recorded {duration:.1f}s', (0.25, 0.80, 0.35, 1))
            self.ids.waveform.set_data(self.recorded_data)
            self.ids.waveform_placeholder.opacity = 0
            self._update_output_meter(self.recorded_data)
        else:
            self._set_status('No data captured', (0.90, 0.70, 0.20, 1))

        self.ids.btn_record.disabled = False
        self.ids.btn_stop.disabled = True
        self.ids.btn_play_orig.disabled = self.recorded_data is None
        self.ids.btn_preview.disabled = True
        self.ids.btn_save.disabled = self.recorded_data is None
        self.ids.btn_reset.disabled = self.recorded_data is None

    # ── Effect Selection ───────────────────────────────
    def select_effect(self, name):
        self.active_effect = name
        self.ids.btn_preview.disabled = self.recorded_data is None
        self.ids.btn_save.disabled = self.recorded_data is None
        if self.recorded_data is not None:
            mod = self._build_modulated_raw()
            if mod is not None:
                self.ids.waveform.set_data(mod)
                self._update_output_meter(mod)
        self._set_status(f'Effect: {name.title()}', (0.40, 0.70, 0.95, 1))

    # ── Build Modulated Audio ──────────────────────────
    def _build_modulated_raw(self):
        if self.recorded_data is None:
            return None
        data = self.recorded_data.copy()
        sr = self.sample_rate

        # 1. Preset effect
        if self.active_effect and self.active_effect in self.EFFECT_MAP:
            func, needs_echo = self.EFFECT_MAP[self.active_effect]
            if needs_echo:
                data = func(
                    data, sr,
                    delay=self.ids.echo_delay_slider.value,
                    decay=self.ids.echo_decay_slider.value
                )
            else:
                data = func(data, sr)

        # 2. LFO
        if self.ids.lfo_enable.state == 'down':
            target = self.ids.lfo_target.text
            rate = self.ids.lfo_rate.value
            depth = self.ids.lfo_depth.value
            waveform = self.ids.lfo_waveform.text

            if target == 'Tremolo':
                data = apply_lfo_tremolo(data, sr, rate, depth, waveform)
            elif target == 'Vibrato':
                data = apply_lfo_vibrato(data, sr, rate, depth, waveform)
            elif target == 'Wah-Wah':
                data = apply_lfo_wah(data, sr, rate, depth, waveform)

        # 3. Pitch shift
        semitones = int(self.ids.pitch_slider.value)
        if semitones != 0:
            data = apply_pitch_shift(data, sr, semitones)

        # 4. Speed change
        speed = self.ids.speed_slider.value
        if abs(speed - 1.0) >= 0.05:
            data = apply_speed_change(data, sr, speed)

        return data.astype(np.float32)

    def _build_final_audio(self):
        raw_mod = self._build_modulated_raw()
        if raw_mod is None:
            return None
        if self.ids.btn_mute.state == 'down':
            return np.zeros_like(raw_mod)
        vol_db = int(self.ids.volume_slider.value)
        final, _, _ = full_process_pipeline(
            raw_mod, self.sample_rate, volume_db=vol_db, target_rms=self.TARGET_RMS
        )
        return final

    # ── Playback ───────────────────────────────────────
    def _play_audio(self, data, label=''):
        if data is None or len(data) == 0:
            return
        data = np.clip(data.astype(np.float32), -1.0, 1.0)
        self._set_status(f'Playing {label}...', (0.30, 0.70, 1.0, 1))

        def _thread():
            try:
                sd.play(data, self.sample_rate)
                sd.wait()
            except Exception as e:
                print(f'Playback error: {e}')
            finally:
                Clock.schedule_once(
                    lambda dt: self._set_status('Done', (0.25, 0.80, 0.35, 1))
                )

        threading.Thread(target=_thread, daemon=True).start()

    def play_original(self):
        if self.recorded_data is None:
            self._show_popup('No Recording', 'Record some audio first!')
            return
        self._play_audio(self.recorded_data, 'original')

    def preview_modulated(self):
        final = self._build_final_audio()
        if final is None:
            self._show_popup('No Audio', 'Record some audio first!')
            return
        self.modulated_data = self._build_modulated_raw()
        self.ids.waveform.set_data(self.modulated_data)
        self._update_output_meter(self.modulated_data)

        fx = self.active_effect or 'modified'
        lfo = ''
        if self.ids.lfo_enable.state == 'down':
            lfo = f'+LFO({self.ids.lfo_target.text})'
        self._play_audio(final, f'{fx}{lfo}')

    def stop_playback(self):
        sd.stop()
        self._set_status('Stopped', (0.70, 0.55, 0.30, 1))

    # ── Save ───────────────────────────────────────────
    def save_audio(self):
        final = self._build_final_audio()
        if final is None:
            if self.recorded_data is not None:
                final = self.recorded_data
            else:
                self._show_popup('No Audio', 'Nothing to save!')
                return

        save_dir = Path('recordings')
        save_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix = f'_{self.active_effect}' if self.active_effect else '_original'
        if self.ids.lfo_enable.state == 'down':
            suffix += f'_lfo{self.ids.lfo_target.text}'
        vol = int(self.ids.volume_slider.value)
        if vol != 0:
            suffix += f'_{vol:+d}dB'
        path = save_dir / f'recording_{ts}{suffix}.wav'

        try:
            audio = final.copy().astype(np.float32)
            mx = np.max(np.abs(audio))
            if mx > 0:
                audio = audio / mx * 0.92
            wavfile.write(str(path), self.sample_rate, audio)
            rms_v, peak_v = rms_level(audio), peak_level(audio)
            rms_db = 20 * np.log10(rms_v) if rms_v > 1e-7 else -np.inf
            peak_db = 20 * np.log10(peak_v) if peak_v > 1e-7 else -np.inf
            self._set_status(f'Saved: {path.name}', (0.20, 0.80, 0.40, 1))
            self._show_popup(
                'Saved!',
                f'Audio saved to:\n{path}\n\n'
                f'RMS: {rms_db:.1f} dB | Peak: {peak_db:.1f} dB\n'
                f'Duration: {len(audio) / self.sample_rate:.1f}s'
            )
        except Exception as e:
            self._set_status('Save failed', (0.90, 0.25, 0.25, 1))
            self._show_popup('Error', f'Could not save:\n{e}')

    # ── Reset ──────────────────────────────────────────
    def reset_modulation(self):
        self.modulated_data = None
        self.active_effect = None
        self.ids.pitch_slider.value = 0
        self.ids.speed_slider.value = 1.0
        self.ids.volume_slider.value = 0
        self.ids.btn_mute.state = 'normal'

        self.ids.lfo_enable.state = 'normal'
        self.ids.lfo_target.text = 'Tremolo'
        self.ids.lfo_waveform.text = 'Sine'
        self.ids.lfo_rate.value = 5.0
        self.ids.lfo_depth.value = 70

        for widget in self.walk():
            if isinstance(widget, ToggleButton) and widget.group == 'fx':
                widget.state = 'normal'

        if self.recorded_data is not None:
            self.ids.waveform.set_data(self.recorded_data)
            self._update_output_meter(self.recorded_data)

        self.ids.btn_preview.disabled = True
        self._set_status('Reset', (0.55, 0.70, 0.55, 1))


# ════════════════════════════════════════════════════════════
#  APP CLASS
# ════════════════════════════════════════════════════════════

class VoiceRecorderApp(App):
    def build(self):
        Builder.load_string(KV)
        self.title = 'Voice Recorder & Modulator'
        return MainLayout()

    def on_stop(self):
        sd.stop()


if __name__ == '__main__':
    VoiceRecorderApp().run()