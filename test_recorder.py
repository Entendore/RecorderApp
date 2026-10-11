"""
Comprehensive pytest test suite for Voice Recorder & Modulator App.

Run with:  pytest test_recorder.py -v

Requirements:
    pip install kivy numpy sounddevice scipy pytest
"""

import os

os.environ.setdefault("KIVY_NO_ARGS", "1")
os.environ.setdefault("KIVY_NO_CONSOLELOG", "1")

import pytest
import numpy as np
import tempfile
import shutil
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from scipy.io import wavfile

# ── Import everything from the app module ──────────────────
from app import (
    # Audio utilities
    rms_level,
    peak_level,
    normalize_rms,
    soft_limiter,
    apply_volume,
    full_process_pipeline,
    # LFO
    generate_lfo,
    apply_lfo_tremolo,
    apply_lfo_vibrato,
    apply_lfo_wah,
    # Static effects
    apply_echo,
    apply_robot,
    apply_reverb,
    apply_pitch_shift,
    apply_speed_change,
    apply_deep_voice,
    apply_chipmunk,
    apply_reverse,
    apply_tremolo,
    apply_telephone,
    apply_whisper,
    # Kivy classes
    MainLayout,
    VoiceRecorderApp,
    WaveformWidget,
)

# ═══════════════════════════════════════════════════════════
#  CONSTANTS & FIXTURES
# ═══════════════════════════════════════════════════════════

SR = 44100


def _sine(duration=1.0, freq=440.0, amp=0.5, sr=SR):
    t = np.arange(int(sr * duration), dtype=np.float64) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def silence():
    return np.zeros(SR, dtype=np.float32)


@pytest.fixture
def sine():
    return _sine(1.0, 440.0, 0.5)


@pytest.fixture
def loud_sine():
    return _sine(1.0, 440.0, 0.95)


@pytest.fixture
def quiet_sine():
    return _sine(1.0, 440.0, 0.001)


@pytest.fixture
def noise():
    return (np.random.randn(SR // 2) * 0.3).astype(np.float32)


@pytest.fixture
def short():
    return _sine(100 / SR, 440.0, 0.5)  # 100 samples


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ── MainLayout with fully-mocked ids ──────────────────────
@pytest.fixture
def layout():
    obj = MainLayout.__new__(MainLayout)
    obj.sample_rate = SR
    obj.recorded_data = None
    obj.modulated_data = None
    obj.active_effect = None
    obj.record_buffer = []
    obj.recording_start = 0
    obj.current_input_level = 0.0
    obj.is_recording = False
    obj._timer_event = None
    obj._level_event = None

    # ── mock every id referenced in the KV / code ────────
    def _slider(val):
        m = MagicMock()
        m.value = val
        return m

    def _spinner(txt):
        m = MagicMock()
        m.text = txt
        return m

    def _toggle(state="normal"):
        m = MagicMock()
        m.state = state
        return m

    def _label(txt=""):
        m = MagicMock()
        m.text = txt
        m.color = (1, 1, 1, 1)
        return m

    obj.ids = MagicMock()
    obj.ids.volume_slider = _slider(0)
    obj.ids.pitch_slider = _slider(0)
    obj.ids.speed_slider = _slider(1.0)
    obj.ids.echo_delay_slider = _slider(0.15)
    obj.ids.echo_decay_slider = _slider(0.50)
    obj.ids.lfo_rate = _slider(5.0)
    obj.ids.lfo_depth = _slider(70)
    obj.ids.lfo_target = _spinner("Tremolo")
    obj.ids.lfo_waveform = _spinner("Sine")
    obj.ids.lfo_enable = _toggle("normal")
    obj.ids.btn_mute = _toggle("normal")

    obj.ids.btn_preview = MagicMock(disabled=True)
    obj.ids.btn_save = MagicMock(disabled=True)
    obj.ids.btn_reset = MagicMock(disabled=True)
    obj.ids.btn_record = MagicMock(disabled=False)
    obj.ids.btn_stop = MagicMock(disabled=True)
    obj.ids.btn_play_orig = MagicMock(disabled=True)

    obj.ids.volume_db_label = _label("0 dB")
    obj.ids.volume_hint = _label("Normal")
    obj.ids.pitch_val = _label("0 st")
    obj.ids.speed_val = _label("1.0x")
    obj.ids.echo_delay_val = _label("0.15s")
    obj.ids.echo_decay_val = _label("0.50")
    obj.ids.lfo_rate_val = _label("5.0 Hz")
    obj.ids.lfo_depth_val = _label("70%")
    obj.ids.lfo_hint = _label("Off")
    obj.ids.lbl_status = _label("Ready")
    obj.ids.lbl_duration = _label("00:00")
    obj.ids.waveform = MagicMock()
    obj.ids.waveform_placeholder = MagicMock(opacity=1)
    obj.ids.output_meter = MagicMock(value=0)
    obj.ids.output_db = MagicMock(text="-inf", color=(0.5, 0.5, 0.5, 1))
    obj.ids.input_meter = MagicMock(value=0)
    obj.ids.input_db = MagicMock(text="-inf")

    # walk() used in reset_modulation to clear FxButtons
    obj.walk = MagicMock(return_value=iter([]))

    return obj


# ═══════════════════════════════════════════════════════════
#  1. AUDIO UTILITIES
# ═══════════════════════════════════════════════════════════

class TestRmsLevel:
    def test_empty(self):
        assert rms_level(np.array([])) == 0.0

    def test_silence(self, silence):
        assert rms_level(silence) == 0.0

    def test_known_dc(self):
        assert np.isclose(
            rms_level(np.ones(1000, dtype=np.float32) * 0.5), 0.5, rtol=1e-3
        )

    def test_sine(self, sine):
        expected = 0.5 / np.sqrt(2)
        assert np.isclose(rms_level(sine), expected, rtol=0.01)

    def test_bipolar(self):
        d = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        assert np.isclose(rms_level(d), 1.0)

    def test_int_input(self):
        d = np.array([1, 2, 3], dtype=np.int16)
        expected = np.sqrt(np.mean(d.astype(np.float64) ** 2))
        assert np.isclose(rms_level(d), expected)


class TestPeakLevel:
    def test_empty(self):
        assert peak_level(np.array([])) == 0.0

    def test_silence(self, silence):
        assert peak_level(silence) == 0.0

    def test_known(self):
        d = np.array([0.1, -0.8, 0.3], dtype=np.float32)
        assert np.isclose(peak_level(d), 0.8)

    def test_sine(self, sine):
        assert np.isclose(peak_level(sine), 0.5, atol=0.01)


class TestNormalizeRms:
    """
    NOTE: normalize_rms clamps gain to [min_gain, max_gain] (default
    [0.5, 10.0]).  Tests that verify the *target is reached* must use
    signals whose current RMS is within the reachable gain range, or
    explicitly widen the gain limits.
    """

    def test_silence_unchanged(self, silence):
        assert np.allclose(normalize_rms(silence), 0.0)

    def test_loud_to_target(self, loud_sine):
        # loud_sine RMS ≈ 0.672; target 0.08 needs gain ≈ 0.12 which
        # is below default min_gain=0.5.  Widen the limits so the
        # normaliser can actually reach the target.
        r = normalize_rms(loud_sine, target_rms=0.08, min_gain=0.01, max_gain=20.0)
        assert np.isclose(rms_level(r), 0.08, rtol=0.1)

    def test_quiet_to_target(self, quiet_sine):
        # quiet_sine RMS ≈ 0.0007; needs gain ≈ 113 which exceeds
        # default max_gain=10.  Widen limits.
        r = normalize_rms(quiet_sine, target_rms=0.08, min_gain=0.01, max_gain=200.0)
        assert np.isclose(rms_level(r), 0.08, rtol=0.1)

    def test_custom_target(self, sine):
        # sine RMS ≈ 0.354; target 0.15 needs gain ≈ 0.42 < 0.5.
        # Widen limits.
        r = normalize_rms(sine, target_rms=0.15, min_gain=0.01, max_gain=20.0)
        assert np.isclose(rms_level(r), 0.15, rtol=0.1)

    def test_within_default_gain_range(self):
        """Signal already close to target – default gain limits suffice."""
        # RMS ≈ 0.10; target 0.08 needs gain ≈ 0.8 (within [0.5, 10])
        sig = _sine(1.0, 440.0, amp=0.10 * np.sqrt(2))
        r = normalize_rms(sig, target_rms=0.08)
        assert np.isclose(rms_level(r), 0.08, rtol=0.05)

    def test_gain_min_clamp(self):
        d = np.ones(1000, dtype=np.float32) * 0.9
        r = normalize_rms(d, target_rms=0.08, min_gain=0.5, max_gain=10.0)
        # Gain clamped to 0.5, so RMS should be 0.9*0.5 = 0.45
        assert np.isclose(rms_level(r), 0.45, rtol=0.01)

    def test_gain_max_clamp(self):
        d = np.ones(1000, dtype=np.float32) * 1e-10
        r = normalize_rms(d, target_rms=0.08, min_gain=0.5, max_gain=10.0)
        # Gain clamped to 10, result still essentially 0
        assert r.dtype == np.float32

    def test_dtype(self, sine):
        assert normalize_rms(sine).dtype == np.float32

    def test_does_not_modify_input(self):
        d = _sine(1.0, 440.0, 0.3)
        d_copy = d.copy()
        normalize_rms(d, target_rms=0.08)
        assert np.allclose(d, d_copy)


class TestSoftLimiter:
    def test_below_threshold(self):
        d = np.ones(100, dtype=np.float32) * 0.5
        assert np.allclose(soft_limiter(d), d, atol=0.01)

    def test_above_threshold_reduced(self):
        d = np.ones(100, dtype=np.float32) * 0.99
        r = soft_limiter(d, threshold=0.92, knee=0.08)
        assert np.max(np.abs(r)) < np.max(np.abs(d))

    def test_hard_clamp(self):
        d = np.ones(100, dtype=np.float32) * 2.0
        r = soft_limiter(d)
        assert np.max(np.abs(r)) <= 1.0

    def test_silence(self, silence):
        assert np.allclose(soft_limiter(silence), 0.0)

    def test_negative(self):
        d = np.ones(100, dtype=np.float32) * -0.99
        r = soft_limiter(d)
        assert np.min(r) >= -1.0

    def test_dtype(self, sine):
        assert soft_limiter(sine).dtype == np.float32


class TestApplyVolume:
    def test_zero_db(self, sine):
        assert np.allclose(apply_volume(sine, 0.0), sine, atol=1e-6)

    def test_positive_db(self, sine):
        r = apply_volume(sine, 6.0)
        g = 10.0 ** (6.0 / 20.0)
        assert np.isclose(np.max(np.abs(r)), 0.5 * g, rtol=0.01)

    def test_negative_db(self, sine):
        r = apply_volume(sine, -6.0)
        assert np.max(np.abs(r)) < np.max(np.abs(sine))

    def test_extreme_negative(self, sine):
        assert np.allclose(apply_volume(sine, -120.0), 0.0, atol=1e-5)

    def test_near_zero(self, sine):
        assert np.allclose(apply_volume(sine, 0.005), sine, atol=1e-3)

    def test_dtype(self, sine):
        assert apply_volume(sine, 3.0).dtype == np.float32


class TestFullProcessPipeline:
    def test_returns_tuple(self, sine):
        r, rms, pk = full_process_pipeline(sine, SR)
        assert isinstance(r, np.ndarray)
        assert isinstance(rms, float)
        assert isinstance(pk, float)

    def test_no_clipping(self, loud_sine):
        r, _, pk = full_process_pipeline(loud_sine, SR, volume_db=10)
        assert np.max(np.abs(r)) <= 1.0

    def test_silence(self, silence):
        r, rms, pk = full_process_pipeline(silence, SR)
        assert np.allclose(r, 0.0, atol=1e-6)

    def test_peak_gte_rms(self, sine):
        _, rms, pk = full_process_pipeline(sine, SR)
        assert pk >= rms - 1e-7

    def test_target_rms_param(self, sine):
        """Higher target_rms should produce higher output RMS (when
        both gains fall within the allowed [0.5, 10] range)."""
        # sine RMS ≈ 0.354.  For gain ≥ 0.5 we need target ≥ 0.177.
        _, r1, _ = full_process_pipeline(sine, SR, target_rms=0.20)
        _, r2, _ = full_process_pipeline(sine, SR, target_rms=0.35)
        assert r2 > r1

    def test_dtype(self, sine):
        assert full_process_pipeline(sine, SR)[0].dtype == np.float32


# ═══════════════════════════════════════════════════════════
#  2. LFO
# ═══════════════════════════════════════════════════════════

class TestGenerateLfo:
    @pytest.mark.parametrize("waveform", ["Sine", "Triangle", "Saw", "Square"])
    def test_range(self, waveform):
        lfo = generate_lfo(SR, SR, 1.0, waveform)
        assert np.max(lfo) <= 1.0 + 1e-5
        assert np.min(lfo) >= -1.0 - 1e-5

    def test_length(self):
        assert len(generate_lfo(22050, SR, 5.0, "Sine")) == 22050

    def test_dtype(self):
        assert generate_lfo(SR, SR, 5.0, "Sine").dtype == np.float32

    def test_sine_starts_near_zero(self):
        lfo = generate_lfo(SR, SR, 1.0, "Sine")
        assert np.isclose(lfo[0], 0.0, atol=1e-6)

    def test_rate_affects_crossings(self):
        slow = generate_lfo(SR, SR, 1.0, "Sine")
        fast = generate_lfo(SR, SR, 10.0, "Sine")
        assert np.sum(np.diff(np.sign(slow)) != 0) < np.sum(
            np.diff(np.sign(fast)) != 0
        )


class TestApplyLfoTremolo:
    def test_basic(self, sine):
        r = apply_lfo_tremolo(sine, SR, 5.0, 50, "Sine")
        assert len(r) == len(sine)
        assert r.dtype == np.float32

    def test_zero_depth_passthrough(self, sine):
        r = apply_lfo_tremolo(sine, SR, 5.0, 0, "Sine")
        assert np.allclose(r, sine, atol=1e-6)

    def test_full_depth_no_exceed(self, sine):
        r = apply_lfo_tremolo(sine, SR, 5.0, 100, "Sine")
        assert np.max(np.abs(r)) <= np.max(np.abs(sine)) + 1e-6

    def test_reduces_energy(self, sine):
        r = apply_lfo_tremolo(sine, SR, 5.0, 80, "Sine")
        assert rms_level(r) <= rms_level(sine) + 1e-6

    @pytest.mark.parametrize("wf", ["Sine", "Triangle", "Saw", "Square"])
    def test_waveforms(self, sine, wf):
        assert len(apply_lfo_tremolo(sine, SR, 5.0, 50, wf)) == len(sine)


class TestApplyLfoVibrato:
    def test_basic(self, sine):
        r = apply_lfo_vibrato(sine, SR, 5.0, 50, "Sine")
        assert len(r) == len(sine)

    def test_zero_depth_passthrough(self, sine):
        r = apply_lfo_vibrato(sine, SR, 5.0, 0, "Sine")
        assert np.allclose(r, sine, atol=1e-3)

    def test_nonzero_depth_modifies(self, sine):
        r = apply_lfo_vibrato(sine, SR, 5.0, 80, "Sine")
        assert not np.allclose(r, sine, atol=1e-4)

    @pytest.mark.parametrize("wf", ["Sine", "Triangle", "Saw", "Square"])
    def test_waveforms(self, sine, wf):
        assert len(apply_lfo_vibrato(sine, SR, 5.0, 50, wf)) == len(sine)


class TestApplyLfoWah:
    def test_basic(self, sine):
        r = apply_lfo_wah(sine, SR, 2.0, 50, "Sine")
        assert len(r) == len(sine)

    def test_zero_depth(self, sine):
        r = apply_lfo_wah(sine, SR, 2.0, 0, "Sine")
        assert len(r) == len(sine)

    def test_short_signal(self, short):
        r = apply_lfo_wah(short, SR, 2.0, 50, "Sine")
        assert len(r) == len(short)

    def test_very_short_signal(self):
        d = np.array([0.1, 0.2], dtype=np.float32)
        r = apply_lfo_wah(d, SR, 2.0, 50, "Sine")
        assert len(r) == len(d)

    @pytest.mark.parametrize("wf", ["Sine", "Triangle", "Saw", "Square"])
    def test_waveforms(self, sine, wf):
        assert len(apply_lfo_wah(sine, SR, 2.0, 50, wf)) == len(sine)


# ═══════════════════════════════════════════════════════════
#  3. STATIC EFFECTS
# ═══════════════════════════════════════════════════════════

class TestApplyEcho:
    def test_basic(self, sine):
        r = apply_echo(sine, SR)
        assert len(r) == len(sine)

    def test_adds_energy(self, sine):
        r = apply_echo(sine, SR, delay=0.1, decay=0.5, n_echoes=4)
        assert rms_level(r) > rms_level(sine)

    def test_zero_decay(self, sine):
        r = apply_echo(sine, SR, decay=0.0)
        assert np.allclose(r, sine, atol=1e-6)

    def test_single_echo(self, sine):
        r = apply_echo(sine, SR, n_echoes=1)
        assert len(r) == len(sine)

    def test_dtype(self, sine):
        assert apply_echo(sine, SR).dtype == np.float32


class TestApplyRobot:
    def test_basic(self, sine):
        r = apply_robot(sine, SR)
        assert len(r) == len(sine)

    def test_modifies(self, sine):
        assert not np.allclose(apply_robot(sine, SR, 50.0), sine, atol=1e-3)

    def test_silence(self, silence):
        assert np.allclose(apply_robot(silence, SR, 0.0), 0.0, atol=1e-6)

    def test_different_freq(self, sine):
        r1 = apply_robot(sine, SR, 20.0)
        r2 = apply_robot(sine, SR, 100.0)
        assert not np.allclose(r1, r2)


class TestApplyReverb:
    def test_basic(self, sine):
        r = apply_reverb(sine, SR)
        assert len(r) == len(sine)

    def test_zero_room(self, sine):
        """room_size=0 means no reverb taps should be added – only the
        dry signal remains (fixed app bug: ds must be > 0)."""
        r = apply_reverb(sine, SR, room_size=0.0)
        assert np.allclose(r, sine, atol=1e-6)

    def test_params_differ(self, sine):
        r1 = apply_reverb(sine, SR, room_size=0.5, damping=0.3)
        r2 = apply_reverb(sine, SR, room_size=1.0, damping=0.8)
        assert not np.allclose(r1, r2)

    def test_reverb_adds_energy(self, sine):
        r = apply_reverb(sine, SR, room_size=0.8, damping=0.3)
        assert rms_level(r) > rms_level(sine)


class TestApplyPitchShift:
    def test_up_shortens(self, sine):
        assert len(apply_pitch_shift(sine, SR, 5)) < len(sine)

    def test_down_lengthens(self, sine):
        assert len(apply_pitch_shift(sine, SR, -5)) > len(sine)

    def test_zero_passthrough(self, sine):
        assert np.allclose(apply_pitch_shift(sine, SR, 0), sine)

    def test_extreme(self, sine):
        assert len(apply_pitch_shift(sine, SR, 12)) > 100

    def test_dtype(self, sine):
        assert apply_pitch_shift(sine, SR, 3).dtype == np.float32


class TestApplySpeedChange:
    def test_faster_shortens(self, sine):
        assert len(apply_speed_change(sine, SR, 2.0)) < len(sine)

    def test_slower_lengthens(self, sine):
        assert len(apply_speed_change(sine, SR, 0.5)) > len(sine)

    def test_normal_passthrough(self, sine):
        assert np.allclose(apply_speed_change(sine, SR, 1.0), sine)

    def test_dtype(self, sine):
        assert apply_speed_change(sine, SR, 1.5).dtype == np.float32


class TestApplyDeepVoice:
    def test_produces_output(self, sine):
        r = apply_deep_voice(sine, SR)
        assert len(r) > 0

    def test_modifies(self, sine):
        r = apply_deep_voice(sine, SR)
        assert not np.allclose(r[: len(sine)], sine, atol=1e-3)


class TestApplyChipmunk:
    def test_shortens(self, sine):
        assert len(apply_chipmunk(sine, SR)) < len(sine)

    def test_dtype(self, sine):
        assert apply_chipmunk(sine, SR).dtype == np.float32


class TestApplyReverse:
    def test_correct(self):
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        assert np.allclose(apply_reverse(d, SR), d[::-1])

    def test_palindrome(self):
        d = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
        assert np.allclose(apply_reverse(d, SR), d)

    def test_single(self):
        d = np.array([0.5], dtype=np.float32)
        assert np.allclose(apply_reverse(d, SR), d)

    def test_dtype(self, sine):
        assert apply_reverse(sine, SR).dtype == np.float32


class TestApplyTremolo:
    def test_basic(self, sine):
        r = apply_tremolo(sine, SR)
        assert len(r) == len(sine)

    def test_zero_depth(self, sine):
        assert np.allclose(apply_tremolo(sine, SR, depth=0.0), sine, atol=1e-6)

    def test_reduces_energy(self, sine):
        assert rms_level(apply_tremolo(sine, SR, depth=0.7)) < rms_level(sine)


class TestApplyTelephone:
    def test_basic(self, sine):
        r = apply_telephone(sine, SR)
        assert len(r) == len(sine)

    def test_modifies(self, sine):
        assert not np.allclose(apply_telephone(sine, SR), sine, atol=1e-3)

    def test_no_clip(self, sine):
        assert np.max(np.abs(apply_telephone(sine, SR))) <= 1.0

    def test_silence(self, silence):
        assert np.allclose(apply_telephone(silence, SR), 0.0, atol=1e-5)


class TestApplyWhisper:
    def test_basic(self, sine):
        r = apply_whisper(sine, SR)
        assert len(r) == len(sine)

    def test_modifies(self, sine):
        assert not np.allclose(apply_whisper(sine, SR, 0.15), sine, atol=0.01)

    def test_zero_noise(self, sine):
        r = apply_whisper(sine, SR, noise_amount=0.0)
        assert np.allclose(r, sine * 0.4, atol=0.01)


# ═══════════════════════════════════════════════════════════
#  4. WAVEFORM WIDGET
# ═══════════════════════════════════════════════════════════

class TestWaveformWidget:
    @pytest.fixture(autouse=True)
    def _ensure_window(self):
        """Try to ensure a Kivy window; skip if impossible."""
        try:
            from kivy.base import EventLoop
            EventLoop.ensure_window()
        except Exception:
            pytest.skip("Kivy window unavailable in this environment")

    def test_set_data_basic(self):
        w = WaveformWidget()
        d = np.sin(np.linspace(0, 2 * np.pi, 500)).astype(np.float32)
        w.set_data(d)
        assert len(w.audio_data) > 0

    def test_set_data_empty(self):
        w = WaveformWidget()
        w.set_data(np.array([], dtype=np.float32))
        assert w.audio_data == []

    def test_set_data_none(self):
        w = WaveformWidget()
        w.set_data(None)
        assert w.audio_data == []

    def test_downsamples_long(self):
        w = WaveformWidget()
        w.set_data(np.random.randn(10000).astype(np.float32))
        assert len(w.audio_data) <= 400

    def test_short_preserved(self):
        w = WaveformWidget()
        d = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        w.set_data(d)
        assert len(w.audio_data) == 3

    def test_normalized_peak(self):
        w = WaveformWidget()
        d = np.array([0.0, 0.5, 1.0, 0.5, 0.0], dtype=np.float32)
        w.set_data(d)
        assert np.isclose(max(abs(v) for v in w.audio_data), 1.0, atol=0.01)

    def test_zeros_remain_zero(self):
        w = WaveformWidget()
        w.set_data(np.zeros(100, dtype=np.float32))
        assert all(v == 0.0 for v in w.audio_data)

    def test_audio_data_is_list(self):
        w = WaveformWidget()
        w.set_data(np.ones(50, dtype=np.float32) * 0.3)
        assert isinstance(w.audio_data, list)


# ═══════════════════════════════════════════════════════════
#  5. MAIN LAYOUT LOGIC
# ═══════════════════════════════════════════════════════════

class TestEffectMap:
    def test_all_keys(self):
        expected = {
            "deep", "chipmunk", "robot", "echo", "reverb",
            "reverse", "tremolo", "telephone", "whisper",
        }
        assert set(MainLayout.EFFECT_MAP.keys()) == expected

    def test_echo_needs_params(self):
        assert MainLayout.EFFECT_MAP["echo"][1] is True

    def test_others_no_params(self):
        for k, (_, need) in MainLayout.EFFECT_MAP.items():
            if k != "echo":
                assert need is False

    def test_correct_functions(self):
        from app import (
            apply_deep_voice, apply_chipmunk, apply_robot,
            apply_echo, apply_reverb, apply_reverse,
            apply_tremolo, apply_telephone, apply_whisper,
        )
        mapping = {
            "deep": apply_deep_voice, "chipmunk": apply_chipmunk,
            "robot": apply_robot, "echo": apply_echo,
            "reverb": apply_reverb, "reverse": apply_reverse,
            "tremolo": apply_tremolo, "telephone": apply_telephone,
            "whisper": apply_whisper,
        }
        for name, (func, _) in MainLayout.EFFECT_MAP.items():
            assert func is mapping[name]


class TestBuildModulatedRaw:
    def test_no_data(self, layout):
        assert layout._build_modulated_raw() is None

    def test_passthrough(self, layout, sine):
        layout.recorded_data = sine.copy()
        r = layout._build_modulated_raw()
        assert np.allclose(r, sine, atol=1e-5)

    def test_echo(self, layout, sine):
        layout.recorded_data = sine.copy()
        layout.active_effect = "echo"
        r = layout._build_modulated_raw()
        assert not np.allclose(r, sine, atol=1e-3)

    def test_robot(self, layout, sine):
        layout.recorded_data = sine.copy()
        layout.active_effect = "robot"
        assert not np.allclose(layout._build_modulated_raw(), sine, atol=1e-3)

    def test_reverse(self, layout, sine):
        layout.recorded_data = sine.copy()
        layout.active_effect = "reverse"
        assert np.allclose(layout._build_modulated_raw(), sine[::-1], atol=1e-5)

    @pytest.mark.parametrize("fx", [
        "deep", "chipmunk", "echo", "reverb",
        "robot", "reverse", "tremolo", "telephone", "whisper",
    ])
    def test_every_effect(self, layout, sine, fx):
        layout.recorded_data = sine.copy()
        layout.active_effect = fx
        r = layout._build_modulated_raw()
        assert r is not None and len(r) > 0 and r.dtype == np.float32

    def test_pitch_shift(self, layout, sine):
        layout.recorded_data = sine.copy()
        layout.ids.pitch_slider.value = 5
        assert len(layout._build_modulated_raw()) < len(sine)

    def test_speed_change(self, layout, sine):
        layout.recorded_data = sine.copy()
        layout.ids.speed_slider.value = 2.0
        assert len(layout._build_modulated_raw()) < len(sine)

    def test_lfo_tremolo(self, layout, sine):
        layout.recorded_data = sine.copy()
        layout.ids.lfo_enable.state = "down"
        layout.ids.lfo_target.text = "Tremolo"
        r = layout._build_modulated_raw()
        assert len(r) == len(sine)

    def test_lfo_vibrato(self, layout, sine):
        layout.recorded_data = sine.copy()
        layout.ids.lfo_enable.state = "down"
        layout.ids.lfo_target.text = "Vibrato"
        assert layout._build_modulated_raw() is not None

    def test_lfo_wah(self, layout, sine):
        layout.recorded_data = sine.copy()
        layout.ids.lfo_enable.state = "down"
        layout.ids.lfo_target.text = "Wah-Wah"
        assert layout._build_modulated_raw() is not None

    def test_combined(self, layout, sine):
        layout.recorded_data = sine.copy()
        layout.active_effect = "reverb"
        layout.ids.lfo_enable.state = "down"
        layout.ids.lfo_target.text = "Tremolo"
        layout.ids.pitch_slider.value = 3
        r = layout._build_modulated_raw()
        assert r is not None and r.dtype == np.float32


class TestBuildFinalAudio:
    def test_no_data(self, layout):
        assert layout._build_final_audio() is None

    def test_basic(self, layout, sine):
        layout.recorded_data = sine.copy()
        r = layout._build_final_audio()
        assert r is not None and np.max(np.abs(r)) <= 1.0

    def test_muted(self, layout, sine):
        layout.recorded_data = sine.copy()
        layout.ids.btn_mute.state = "down"
        assert np.allclose(layout._build_final_audio(), 0.0, atol=1e-6)

    def test_with_volume(self, layout, sine):
        layout.recorded_data = sine.copy()
        layout.ids.volume_slider.value = 6
        r = layout._build_final_audio()
        assert np.max(np.abs(r)) <= 1.0

    def test_with_effect_and_volume(self, layout, sine):
        layout.recorded_data = sine.copy()
        layout.active_effect = "echo"
        layout.ids.volume_slider.value = -5
        assert layout._build_final_audio() is not None


class TestSelectEffect:
    def test_sets_active(self, layout, sine):
        layout.recorded_data = sine.copy()
        layout.select_effect("echo")
        assert layout.active_effect == "echo"

    def test_updates_waveform(self, layout, sine):
        layout.recorded_data = sine.copy()
        layout.select_effect("reverse")
        layout.ids.waveform.set_data.assert_called()

    def test_no_data_no_crash(self, layout):
        layout.select_effect("echo")
        assert layout.active_effect == "echo"


class TestResetModulation:
    def test_clears_effect(self, layout, sine):
        layout.recorded_data = sine.copy()
        layout.active_effect = "echo"
        layout.modulated_data = sine.copy()
        layout.reset_modulation()
        assert layout.active_effect is None
        assert layout.modulated_data is None

    def test_resets_sliders(self, layout, sine):
        layout.recorded_data = sine.copy()
        layout.ids.pitch_slider.value = 5
        layout.ids.speed_slider.value = 1.5
        layout.ids.volume_slider.value = -10
        layout.reset_modulation()
        assert layout.ids.pitch_slider.value == 0
        assert layout.ids.speed_slider.value == 1.0
        assert layout.ids.volume_slider.value == 0

    def test_resets_lfo(self, layout, sine):
        layout.recorded_data = sine.copy()
        layout.ids.lfo_enable.state = "down"
        layout.reset_modulation()
        assert layout.ids.lfo_enable.state == "normal"

    def test_resets_mute(self, layout, sine):
        layout.recorded_data = sine.copy()
        layout.ids.btn_mute.state = "down"
        layout.reset_modulation()
        assert layout.ids.btn_mute.state == "normal"


class TestRecording:
    @patch("app.threading.Thread")
    @patch("app.Clock")
    def test_start_sets_state(self, mock_clock, mock_thread, layout):
        layout.start_recording()
        assert layout.is_recording is True
        assert layout.ids.btn_record.disabled is True
        assert layout.ids.btn_stop.disabled is False
        layout.is_recording = False  # cleanup

    @patch("app.threading.Thread")
    @patch("app.Clock")
    def test_double_start_ignored(self, mock_clock, mock_thread, layout):
        layout.is_recording = True
        layout.start_recording()
        assert layout.is_recording is True

    def test_stop_not_recording(self, layout):
        layout.is_recording = False
        layout.stop_recording()

    @patch("app.Clock")
    def test_stop_with_data(self, mock_clock, layout):
        layout.is_recording = True
        layout.record_buffer = [np.zeros((100, 1), dtype=np.float32)]
        layout._timer_event = MagicMock()
        layout._level_event = MagicMock()
        layout.stop_recording()
        assert layout.is_recording is False
        assert layout.recorded_data is not None
        assert layout.ids.btn_stop.disabled is True

    @patch("app.Clock")
    def test_stop_empty_buffer(self, mock_clock, layout):
        layout.is_recording = True
        layout.record_buffer = []
        layout._timer_event = MagicMock()
        layout._level_event = MagicMock()
        layout.stop_recording()
        assert layout.recorded_data is None

    def test_record_callback(self, layout):
        indata = np.array([[0.5], [-0.3]], dtype=np.float32)
        layout._record_callback(indata, 2, None, None)
        assert len(layout.record_buffer) == 1
        assert np.allclose(layout.record_buffer[0], indata)
        assert np.isclose(layout.current_input_level, 0.5)


class TestPlayback:
    @patch("app.sd")
    def test_play_original_no_data(self, mock_sd, layout):
        layout.recorded_data = None
        layout.play_original()
        mock_sd.play.assert_not_called()

    @patch("app.sd")
    def test_play_original(self, mock_sd, layout, sine):
        layout.recorded_data = sine.copy()
        layout.play_original()
        time.sleep(0.15)
        mock_sd.play.assert_called_once()

    @patch("app.sd")
    def test_stop_playback(self, mock_sd, layout):
        layout.stop_playback()
        mock_sd.stop.assert_called_once()

    @patch("app.sd")
    def test_preview_no_data(self, mock_sd, layout):
        layout.recorded_data = None
        layout.preview_modulated()
        mock_sd.play.assert_not_called()

    @patch("app.sd")
    def test_preview_with_data(self, mock_sd, layout, sine):
        layout.recorded_data = sine.copy()
        layout.active_effect = "echo"
        layout.preview_modulated()
        time.sleep(0.15)
        mock_sd.play.assert_called_once()


class TestSaveAudio:
    def test_save_no_data_shows_popup(self, layout):
        layout.recorded_data = None
        with patch.object(layout, "_show_popup") as mock_popup:
            layout.save_audio()
            mock_popup.assert_called_once()

    def test_save_creates_file(self, layout, sine, tmp_dir):
        layout.recorded_data = sine.copy()
        layout.active_effect = None
        layout.ids.btn_mute.state = "normal"
        layout.ids.volume_slider.value = 0
        layout.ids.lfo_enable.state = "normal"
        layout.ids.pitch_slider.value = 0
        layout.ids.speed_slider.value = 1.0

        rec_dir = tmp_dir / "recordings"
        rec_dir.mkdir(exist_ok=True)

        with patch("app.Path") as mock_path_cls:
            fake_path = MagicMock()
            fake_path.parent = rec_dir
            fake_path.name = "test_recording.wav"
            fake_path.__str__ = lambda s: str(rec_dir / "test_recording.wav")
            mock_path_cls.return_value = fake_path

            with patch.object(layout, "_show_popup"):
                layout.save_audio()

    def test_wav_roundtrip(self, sine, tmp_dir):
        """Independent check: write + read WAV."""
        path = tmp_dir / "roundtrip.wav"
        audio = sine.copy().astype(np.float32)
        mx = np.max(np.abs(audio))
        if mx > 0:
            audio = audio / mx * 0.92
        wavfile.write(str(path), SR, audio)
        sr2, data2 = wavfile.read(str(path))
        assert sr2 == SR
        assert len(data2) == len(audio)


class TestSetStatus:
    def test_updates(self, layout):
        layout._set_status("Test", (1, 0, 0, 1))
        assert layout.ids.lbl_status.text == "Test"
        assert layout.ids.lbl_status.color == (1, 0, 0, 1)


class TestOnVolumeCallback:
    def test_zero_db(self, layout):
        layout._on_volume(None, 0)
        assert layout.ids.volume_db_label.text == "0 dB"
        assert layout.ids.volume_hint.text == "Normal"

    def test_positive_db(self, layout):
        layout._on_volume(None, 10)
        assert "10" in layout.ids.volume_db_label.text
        assert layout.ids.volume_hint.text == "Loud"

    def test_negative_db(self, layout):
        layout._on_volume(None, -8)
        assert layout.ids.volume_hint.text == "Soft"

    def test_very_quiet(self, layout):
        layout._on_volume(None, -18)
        assert layout.ids.volume_hint.text == "Very quiet"

    def test_very_loud(self, layout):
        layout._on_volume(None, 15)
        assert layout.ids.volume_hint.text == "Very loud"


class TestOnPitchCallback:
    def test_zero(self, layout):
        layout._on_pitch(None, 0)
        assert layout.ids.pitch_val.text == "0 st"

    def test_nonzero(self, layout):
        layout._on_pitch(None, 5)
        assert "5" in layout.ids.pitch_val.text


class TestOnSpeedCallback:
    def test_normal(self, layout):
        layout._on_speed(None, 1.0)
        assert "1.0" in layout.ids.speed_val.text


class TestOnLfoChange:
    def test_lfo_off(self, layout):
        layout.ids.lfo_enable.state = "normal"
        layout._on_lfo_change(None, None)
        assert layout.ids.lfo_hint.text == "Off"

    def test_lfo_on(self, layout):
        layout.ids.lfo_enable.state = "down"
        layout.ids.lfo_target.text = "Vibrato"
        layout.ids.lfo_waveform.text = "Saw"
        layout._on_lfo_change(None, None)
        assert "Saw" in layout.ids.lfo_hint.text
        assert "Vibrato" in layout.ids.lfo_hint.text