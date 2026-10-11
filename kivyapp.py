import os
import time
import json
import threading
import subprocess
import zipfile
import webbrowser
import shutil
from datetime import datetime
import numpy as np  # Global import for performance
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.slider import Slider
from kivy.uix.popup import Popup
from kivy.uix.spinner import Spinner
from kivy.uix.switch import Switch
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.utils import platform
from kivy.core.audio import SoundLoader
from kivy.lang import Builder
from kivy.properties import ListProperty, NumericProperty
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture

# --- Premium UI Definition ---
KV = """
<RenamePopup>:
    title: "Rename Recording"
    size_hint: 0.8, 0.4
    BoxLayout:
        orientation: 'vertical'
        padding: 20
        spacing: 20
        TextInput:
            id: rename_input
            multiline: False
            font_size: 20
            text: root.old_name
        BoxLayout:
            orientation: 'horizontal'
            spacing: 20
            size_hint_y: 0.4
            Button:
                text: "Cancel"
                on_press: root.dismiss()
            Button:
                text: "Save"
                background_color: 0.2, 0.6, 0.2, 1
                on_press: root.confirm_rename()

<NotesPopup>:
    title: "Field Notes"
    size_hint: 0.9, 0.6
    BoxLayout:
        orientation: 'vertical'
        padding: 20
        spacing: 20
        TextInput:
            id: notes_input
            multiline: True
            font_size: 16
            text: root.old_notes
        BoxLayout:
            orientation: 'horizontal'
            spacing: 20
            size_hint_y: 0.2
            Button:
                text: "Cancel"
                on_press: root.dismiss()
            Button:
                text: "Save Notes"
                background_color: 0.2, 0.6, 0.2, 1
                on_press: root.confirm_save()

<ProcessPopup>:
    title: "Post-Process Audio"
    size_hint: 0.8, 0.5
    BoxLayout:
        orientation: 'vertical'
        padding: 20
        spacing: 20
        Label:
            text: "Apply DSP to saved recording.\\nOriginal file is kept."
            halign: 'center'
        Button:
            text: "Normalize Volume (Max Gain)"
            on_press: root.apply_process('normalize')
            background_color: 0.2, 0.6, 0.8, 1
        Button:
            text: "Sub-Bass Cut (Remove < 80Hz)"
            on_press: root.apply_process('highpass')
            background_color: 0.2, 0.6, 0.8, 1
        Button:
            text: "Cancel"
            on_press: root.dismiss()

<BatchPopup>:
    title: "Batch Process All Recordings"
    size_hint: 0.8, 0.5
    BoxLayout:
        orientation: 'vertical'
        padding: 20
        spacing: 20
        Label:
            text: "Apply effect to ALL recordings.\\nThis may take a while."
            halign: 'center'
        Button:
            text: "Batch Normalize All"
            on_press: root.run_batch('normalize')
            background_color: 0.8, 0.4, 0.2, 1
        Button:
            text: "Batch Sub-Bass Cut All"
            on_press: root.run_batch('highpass')
            background_color: 0.8, 0.4, 0.2, 1
        Button:
            text: "Cancel"
            on_press: root.dismiss()

<SpectrogramWidget>:
    canvas:
        Color:
            rgba: 0, 0, 0, 1
        Rectangle:
            pos: self.pos
            size: self.size
        Color:
            rgba: 1, 1, 1, 1
        Rectangle:
            pos: self.pos
            size: self.size
            texture: self.spec_texture

<RecordingItem>:
    orientation: 'horizontal'
    size_hint_y: None
    height: 55
    spacing: 2
    padding: [5, 5, 5, 5]
    canvas.before:
        Color:
            rgba: root.bg_color if root.bg_color else app.theme_card
        Rectangle:
            pos: self.pos
            size: self.size
            
    Button:
        id: play_btn
        text: root.display_name
        font_size: 9
        color: app.theme_text
        on_press: root.app.play_recording(root.filepath, root)
        background_normal: ''
        background_color: 0,0,0, 0
        
    Button:
        text: root.category
        size_hint_x: 0.1
        font_size: 9
        on_press: root.app.cycle_category(root.filepath)
        background_normal: ''
        background_color: 0.6, 0.4, 0.8, 1
        
    Button:
        text: "FX"
        size_hint_x: 0.08
        font_size: 10
        on_press: root.app.open_process_popup(root.filepath)
        background_normal: ''
        background_color: 0.8, 0.4, 0.2, 1
        
    Button:
        text: "Map"
        size_hint_x: 0.08
        font_size: 10
        disabled: not root.has_gps
        on_press: root.app.open_map(root.filepath)
        background_normal: ''
        background_color: 0.2, 0.5, 0.2, 1
        
    Button:
        text: "Notes"
        size_hint_x: 0.08
        font_size: 10
        on_press: root.app.open_notes_popup(root.filepath)
        background_normal: ''
        background_color: 0.2, 0.5, 0.2, 1
        
    Button:
        text: "Rename"
        size_hint_x: 0.08
        font_size: 10
        on_press: root.app.open_rename_popup(root.filepath, root.display_name)
        background_normal: ''
        background_color: 0.2, 0.5, 0.8, 1

    Button:
        text: "Delete"
        size_hint_x: 0.08
        font_size: 10
        on_press: root.app.delete_recording(root.filepath, root)
        background_normal: ''
        background_color: 0.8, 0.2, 0.2, 1

<RecorderApp>:
    orientation: 'vertical'
    padding: 20
    spacing: 8
    canvas.before:
        Color:
            rgba: app.theme_bg
        Rectangle:
            pos: self.pos
            size: self.size

    # Top Header
    BoxLayout:
        size_hint_y: 0.08
        orientation: 'horizontal'
        spacing: 5
        Label:
            text: "Bioacoustics Lab"
            font_size: 18
            bold: True
            color: app.theme_text
            size_hint_x: 0.2
        Spinner:
            id: format_spinner
            text: "Format: AAC"
            values: ["Format: AAC", "Format: FLAC"]
            size_hint_x: 0.13
            background_color: app.theme_card
            color: app.theme_text
        Spinner:
            id: channels_spinner
            text: "Mono"
            values: ["Mono", "Stereo"]
            size_hint_x: 0.1
            background_color: app.theme_card
            color: app.theme_text
        Spinner:
            id: split_spinner
            text: "Split: Off"
            values: ["Split: Off", "Split: 15 Min", "Split: 30 Min", "Split: 1 Hour"]
            size_hint_x: 0.13
            background_color: app.theme_card
            color: app.theme_text
        Button:
            text: "Batch FX"
            size_hint_x: 0.09
            on_press: app.open_batch_popup()
            background_normal: ''
            background_color: 0.8, 0.4, 0.2, 1
        Button:
            text: "Zip All"
            size_hint_x: 0.08
            on_press: app.zip_and_share_all()
            background_normal: ''
            background_color: 0.8, 0.6, 0.1, 1
        Button:
            text: "Stealth"
            size_hint_x: 0.08
            on_press: app.toggle_stealth()
            background_normal: ''
            background_color: 0, 0, 0, 1
        Button:
            text: "Theme"
            size_hint_x: 0.08
            on_press: app.toggle_theme()
            background_normal: ''
            background_color: 0.3, 0.3, 0.4, 1

    # Timer & Spectrogram
    Label:
        id: timer_label
        text: "00:00"
        font_size: 48
        size_hint_y: 0.12
        bold: True
        color: app.theme_text

    SpectrogramWidget:
        id: vu_meter
        size_hint_y: 0.12

    # Toggles & Threshold Row
    BoxLayout:
        size_hint_y: 0.08
        orientation: 'vertical'
        BoxLayout:
            orientation: 'horizontal'
            Spinner:
                id: mic_source_spinner
                text: "Mic: Default"
                values: ["Mic: Default", "Mic: Unprocessed", "Mic: Voice Rec"]
                size_hint_x: 0.25
                background_color: app.theme_card
                color: app.theme_text
            Spinner:
                id: start_delay_spinner
                text: "Start: Now"
                values: ["Start: Now", "Start: 5s", "Start: 10s", "Start: 30s"]
                size_hint_x: 0.2
                background_color: app.theme_card
                color: app.theme_text
            Label:
                text: "Pre-Rec"
                color: app.theme_subtext
                size_hint_x: 0.15
            Switch:
                id: pre_rec_switch
                size_hint_x: 0.1
            Label:
                text: "Sound Act"
                color: app.theme_subtext
                size_hint_x: 0.15
            Switch:
                id: sound_active_switch
                size_hint_x: 0.1
        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: 0.5
            Label:
                text: "Act Threshold:"
                color: app.theme_subtext
                size_hint_x: 0.2
            Slider:
                id: threshold_slider
                min: 0
                max: 500
                value: 50
                size_hint_x: 0.4
            Label:
                text: "Input Gain:"
                color: app.theme_subtext
                size_hint_x: 0.15
            Slider:
                id: gain_slider
                min: 1
                max: 40
                value: 10
                size_hint_x: 0.2
            Label:
                id: gain_label
                text: "1.0x"
                color: app.theme_subtext
                size_hint_x: 0.05

    # Main Controls
    BoxLayout:
        size_hint_y: 0.1
        orientation: 'horizontal'
        spacing: 10
        Button:
            id: record_btn
            text: "Record"
            font_size: 18
            background_normal: ''
            background_color: 0.8, 0.2, 0.2, 1
            on_press: app.toggle_recording()
        Button:
            id: pause_btn
            text: "Pause"
            font_size: 18
            disabled: True
            background_normal: ''
            background_color: 0.8, 0.6, 0.1, 1
            on_press: app.toggle_pause()
        Button:
            id: mark_btn
            text: "Mark"
            font_size: 18
            disabled: True
            background_normal: ''
            background_color: 0.6, 0.2, 0.8, 1
            on_press: app.add_bookmark()
            
    BoxLayout:
        size_hint_y: 0.06
        orientation: 'horizontal'
        spacing: 10
        Label:
            text: "Auto-Stop:"
            color: app.theme_subtext
            size_hint_x: 0.15
        Spinner:
            id: auto_stop_spinner
            text: "Off"
            values: ["Off", "1 Min", "5 Mins", "10 Mins", "30 Mins", "1 Hour", "Infinite"]
            size_hint_x: 0.25
            background_color: app.theme_card
            color: app.theme_text
        Label:
            id: status_label
            text: "Ready"
            font_size: 14
            color: app.theme_subtext
            size_hint_x: 0.6

    # Search & Category Filter Bar
    BoxLayout:
        size_hint_y: 0.06
        orientation: 'horizontal'
        spacing: 10
        TextInput:
            id: search_input
            hint_text: "Search recordings..."
            multiline: False
            on_text: app.refresh_recordings_list()
        Spinner:
            id: cat_filter_spinner
            text: "All Categories"
            values: ["All Categories", "General", "Birds", "Insects", "Mammals", "Weather"]
            size_hint_x: 0.4
            background_color: app.theme_card
            color: app.theme_text
            on_text: app.refresh_recordings_list()

    Label:
        text: "Saved Recordings"
        size_hint_y: 0.04
        font_size: 18
        bold: True
        color: app.theme_text

    ScrollView:
        size_hint_y: 0.15
        canvas.before:
            Color:
                rgba: app.theme_card
            Rectangle:
                pos: self.pos
                size: self.size
        BoxLayout:
            id: recordings_list
            orientation: 'vertical'
            size_hint_y: None
            height: self.minimum_height
            spacing: 8
            padding: [10, 10, 10, 10]

    # Sticky Mini-Player (Bottom Bar)
    BoxLayout:
        id: player_bar
        size_hint_y: 0
        opacity: 0
        disabled: True
        orientation: 'vertical'
        padding: [10, 5, 10, 5]
        spacing: 5
        canvas.before:
            Color:
                rgba: 0.1, 0.1, 0.15, 1
            Rectangle:
                pos: self.pos
                size: self.size
        
        Label:
            id: player_title
            text: ""
            font_size: 14
            bold: True
            color: 1, 1, 1, 1
            size_hint_y: 0.2
            halign: 'left'
        
        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: 0.3
            spacing: 10
            Label:
                id: current_pos
                text: "00:00"
                size_hint_x: 0.15
                color: 0.8, 0.8, 0.8, 1
            Slider:
                id: seek_slider
                min: 0
                max: 100
                value: 0
                size_hint_x: 0.7
                on_touch_up: if args[1].type == 'up': app.seek_audio(self.value)
            Label:
                id: total_duration
                text: "00:00"
                size_hint_x: 0.15
                color: 0.8, 0.8, 0.8, 1
                
        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: 0.3
            spacing: 5
            Button:
                text: "Set A"
                on_press: app.set_loop_point('a')
                background_normal: ''
                background_color: 0.3, 0.5, 0.3, 1
            Button:
                text: "Set B"
                on_press: app.set_loop_point('b')
                background_normal: ''
                background_color: 0.3, 0.5, 0.3, 1
            Button:
                text: "Save Seg"
                on_press: app.save_segment()
                background_normal: ''
                background_color: 0.8, 0.6, 0.1, 1
            Button:
                text: "Play/Pause"
                on_press: app.toggle_mini_player_playback()
                background_normal: ''
                background_color: 0.2, 0.7, 0.3, 1
            Button:
                text: "Stop"
                on_press: app.stop_mini_player()
                background_normal: ''
                background_color: 0.8, 0.3, 0.2, 1
                
        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: 0.2
            spacing: 5
            Label:
                text: "Speed:"
                color: 0.8, 0.8, 0.8, 1
                size_hint_x: 0.2
            Spinner:
                id: speed_spinner
                text: "1.0x"
                values: ["0.5x", "0.75x", "1.0x", "1.5x", "2.0x"]
                size_hint_x: 0.3
                on_text: app.change_playback_speed()
                background_color: app.theme_card
                color: app.theme_text
            Button:
                text: "Clear Loop"
                on_press: app.clear_loop()
                size_hint_x: 0.5
                background_normal: ''
                background_color: 0.5, 0.3, 0.3, 1
                
    # Stealth Mode Overlay
    Widget:
        id: stealth_overlay
        opacity: 0
        disabled: True
        canvas:
            Color:
                rgba: 0, 0, 0, 1
            Rectangle:
                pos: self.pos
                size: self.size
        on_touch_down: app.toggle_stealth()
"""

class RenamePopup(Popup):
    def __init__(self, app, filepath, old_name, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.filepath = filepath
        self.old_name = old_name
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        self.ids.rename_input.text = base_name

    def confirm_rename(self):
        new_name = self.ids.rename_input.text.strip()
        if not new_name: return
        ext = os.path.splitext(self.filepath)[1]
        dirname = os.path.dirname(self.filepath)
        new_filepath = os.path.join(dirname, new_name + ext)
        try:
            os.rename(self.filepath, new_filepath)
            old_base = os.path.basename(self.filepath)
            new_base = os.path.basename(new_filepath)
            if old_base in self.app.metadata:
                self.app.metadata[new_base] = self.app.metadata.pop(old_base)
                self.app.save_metadata()
            self.app.refresh_recordings_list()
        except: pass
        self.dismiss()

class NotesPopup(Popup):
    def __init__(self, app, filepath, old_notes, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.filepath = filepath
        self.old_notes = old_notes
        self.ids.notes_input.text = old_notes

    def confirm_save(self):
        notes = self.ids.notes_input.text.strip()
        filename = os.path.basename(self.filepath)
        if filename not in self.app.metadata:
            self.app.metadata[filename] = {"bookmarks": [], "starred": False}
        self.app.metadata[filename]["notes"] = notes
        self.app.save_metadata()
        self.app.refresh_recordings_list()
        self.dismiss()

class ProcessPopup(Popup):
    def __init__(self, app, filepath, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.filepath = filepath

    def apply_process(self, process_type):
        self.dismiss()
        self.app.root.ids.status_label.text = "Processing audio..."
        threading.Thread(target=self.app.process_audio_file, args=(self.filepath, process_type)).start()

class BatchPopup(Popup):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app

    def run_batch(self, process_type):
        self.dismiss()
        self.app.root.ids.status_label.text = "Batch processing started..."
        threading.Thread(target=self.app.batch_process_all, args=(process_type,)).start()

# --- Vectorized Spectrogram Widget for Performance ---
class SpectrogramWidget(Widget):
    spec_texture = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spec_texture = Texture.create(size=(64, 10), colorfmt='rgba')
        self.spec_texture.wrap = 'clamp_to_edge'
        self.history = np.zeros((64, 10))

    def update_bars(self, bars):
        # Shift history left using np.roll (fast C-level operation)
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = bars
        
        # Vectorized RGBA mapping (No Python loops!)
        val = self.history
        r = np.where(val < 0.5, 0, (val - 0.5) * 2 * 255).clip(0, 255).astype(np.uint8)
        g = np.where(val < 0.5, val * 2 * 255, 255).clip(0, 255).astype(np.uint8)
        b = np.zeros_like(val, dtype=np.uint8)
        a = np.full_like(val, 255, dtype=np.uint8)
        
        # Stack to get (64, 10, 4) and flip Y so low freq is at bottom
        buf = np.stack((r, g, b, a), axis=-1)
        buf = np.flip(buf, axis=1)
        
        self.spec_texture.blit_buffer(buf.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
        self.canvas.ask_update()

class RecordingItem(BoxLayout):
    bg_color = ListProperty(None)
    has_gps = False
    def __init__(self, filepath, display_name, app, category="General", has_gps=False, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.app = app
        self.display_name = display_name
        self.category = category
        self.has_gps = has_gps

class RecorderApp(App, BoxLayout):
    theme_bg = ListProperty((0.96, 0.96, 0.96, 1))
    theme_card = ListProperty((1, 1, 1, 1))
    theme_text = ListProperty((0.1, 0.1, 0.1, 1))
    theme_subtext = ListProperty((0.4, 0.4, 0.4, 1))
    
    def build(self):
        self.is_recording = False
        self.is_paused = False
        self.recorder_thread = None
        self.current_sound = None
        self.currently_playing_item = None
        self.record_start_time = 0
        self.accumulated_time = 0
        self.timer_event = None
        self.vu_event = None
        self.playback_event = None
        self.is_mini_player_playing = False
        self.auto_stop_limit = 0
        self.auto_split_limit = 0
        self.split_part_number = 1
        self.loop_a = None
        self.loop_b = None
        self.delay_start_time = 0
        self.wake_lock = None
        
        self.storage_dir = self.user_data_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            
        self.metadata_file = os.path.join(self.storage_dir, "metadata.json")
        self.metadata = self.load_metadata()

        return Builder.load_string(KV)

    def on_start(self):
        if platform == 'android':
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.RECORD_AUDIO, Permission.ACCESS_FINE_LOCATION])
        self.refresh_recordings_list()
        self.root.ids.threshold_label.text = str(int(self.root.ids.threshold_slider.value))
        
        self.root.ids.gain_slider.bind(value=self.update_gain_label)
        self.update_gain_label(self.root.ids.gain_slider, self.root.ids.gain_slider.value)

    def update_gain_label(self, instance, value):
        self.root.ids.gain_label.text = f"{value/10.0:.1f}x"

    def load_metadata(self):
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f: return json.load(f)
            except: return {}
        return {}

    def save_metadata(self):
        try:
            with open(self.metadata_file, 'w') as f: json.dump(self.metadata, f)
        except: pass

    def toggle_theme(self):
        if self.theme_bg[0] > 0.5:
            self.theme_bg = (0.1, 0.1, 0.12, 1)
            self.theme_card = (0.2, 0.2, 0.22, 1)
            self.theme_text = (1, 1, 1, 1)
            self.theme_subtext = (0.7, 0.7, 0.7, 1)
        else:
            self.theme_bg = (0.96, 0.96, 0.96, 1)
            self.theme_card = (1, 1, 1, 1)
            self.theme_text = (0.1, 0.1, 0.1, 1)
            self.theme_subtext = (0.4, 0.4, 0.4, 1)

    def toggle_stealth(self):
        ov = self.root.ids.stealth_overlay
        if ov.opacity == 0:
            ov.opacity = 1
            ov.disabled = False
        else:
            ov.opacity = 0
            ov.disabled = True

    def on_pause(self): return True

    def acquire_wakelock(self):
        if platform == 'android':
            try:
                from jnius import autoclass
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
                Context = autoclass('android.content.Context')
                PowerManager = autoclass('android.os.PowerManager')
                pm = PythonActivity.mActivity.getSystemService(Context.POWER_SERVICE)
                self.wake_lock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "BioacousticsRecorderLock")
                self.wake_lock.acquire()
            except: pass

    def release_wakelock(self):
        if self.wake_lock:
            try: self.wake_lock.release()
            except: pass
            self.wake_lock = None

    def check_storage(self):
        try:
            usage = shutil.disk_usage(self.storage_dir)
            if usage.free < 100 * 1024 * 1024: return False
            return True
        except:
            return True 

    def format_time(self, seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

    def update_timer(self, dt):
        if self.is_paused: return
        elapsed = self.accumulated_time + (time.time() - self.record_start_time)
        self.root.ids.timer_label.text = self.format_time(elapsed)
        
        if self.auto_split_limit > 0 and elapsed >= self.auto_split_limit:
            self.root.ids.status_label.text = "Auto-splitting file..."
            self.auto_split_chunk()
            
        if self.auto_stop_limit > 0 and elapsed >= self.auto_stop_limit:
            self.root.ids.status_label.text = "Auto-stop triggered"
            self.stop_recording()

    def auto_split_chunk(self):
        self.split_part_number += 1
        self.accumulated_time = 0
        self.record_start_time = time.time()
        
        if platform == 'android':
            self.stop_recording_android()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_file = os.path.join(self.storage_dir, f"rec_{timestamp}_part{self.split_part_number}.m4a")
            self.start_recording_android(self.current_file, self.root.ids.mic_source_spinner.text, self.root.ids.format_spinner.text, self.root.ids.channels_spinner.text)
        else:
            if self.recorder_thread:
                self.recorder_thread.stop()
                self.recorder_thread.join()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_file = os.path.join(self.storage_dir, f"rec_{timestamp}_part{self.split_part_number}.wav")
            selected_filter = self.root.ids.filter_spinner.text
            is_sound_active = self.root.ids.sound_active_switch.active
            is_auto_trim = self.root.ids.auto_trim_switch.active
            channels = 2 if self.root.ids.channels_spinner.text == "Stereo" else 1
            gain = self.root.ids.gain_slider.value / 10.0
            self.recorder_thread = DesktopRecorder(self.current_file, selected_filter, is_sound_active, is_auto_trim, channels, int(self.root.ids.threshold_slider.value), gain)
            self.recorder_thread.start()
            
        self.root.ids.status_label.text = f"Recording (Part {self.split_part_number})..."

    def update_vu_desktop(self, dt):
        if self.recorder_thread and self.recorder_thread.is_running:
            self.root.ids.vu_meter.update_bars(self.recorder_thread.spectrum_bars)
            
            if self.recorder_thread.is_clipping:
                self.root.ids.timer_label.color = (1, 0, 0, 1)
            else:
                self.root.ids.timer_label.color = self.theme_text
                
            if self.recorder_thread.is_silent and self.root.ids.sound_active_switch.active:
                self.root.ids.status_label.text = "Listening... (Silent)"
            else:
                self.root.ids.status_label.text = f"Recording (Part {self.split_part_number})..."

    def update_vu_android(self, dt):
        if self.recorder and not self.is_paused:
            try: 
                amp = self.recorder.getMaxAmplitude()
                val = min(1.0, amp / 32767.0)
                self.root.ids.vu_meter.update_bars([val]*10)
                if amp > 32000:
                    self.root.ids.timer_label.color = (1, 0, 0, 1)
                else:
                    self.root.ids.timer_label.color = self.theme_text
            except: pass

    def get_android_gps(self):
        try:
            from jnius import autoclass
            LocationManager = autoclass('android.location.LocationManager')
            Context = autoclass('android.content.Context')
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            mActivity = PythonActivity.mActivity
            lm = mActivity.getSystemService(Context.LOCATION_SERVICE)
            location = lm.getLastKnownLocation(LocationManager.GPS_PROVIDER)
            if location:
                return location.getLatitude(), location.getLongitude()
        except: pass
        return None, None

    def toggle_recording(self):
        if not self.is_recording:
            if not self.check_storage():
                self.root.ids.status_label.text = "Storage Full! Free up space."
                return
                
            self.stop_mini_player()
            self.split_part_number = 1
            
            delay_text = self.root.ids.start_delay_spinner.text
            if "Now" not in delay_text:
                delay_secs = int(delay_text.replace("Start: ", "").replace("s", ""))
                self.delay_start_time = delay_secs
                self.root.ids.timer_label.text = f"Starting in {delay_secs}..."
                self.root.ids.status_label.text = "Delay Active..."
                Clock.schedule_once(self.trigger_start_recording, delay_secs)
                return
                
            self.start_recording()
        else:
            self.stop_recording()

    def trigger_start_recording(self, dt):
        self.root.ids.status_label.text = "Recording..."
        self.start_recording()

    def add_bookmark(self):
        if self.is_recording and not self.is_paused:
            elapsed = self.accumulated_time + (time.time() - self.record_start_time)
        elif self.is_mini_player_playing and self.current_sound:
            elapsed = self.current_sound.get_pos()
        else: return
            
        filename = os.path.basename(self.current_file if self.is_recording else self.current_sound.source)
        if filename not in self.metadata: self.metadata[filename] = {"bookmarks": [], "starred": False}
        if "bookmarks" not in self.metadata[filename]: self.metadata[filename]["bookmarks"] = []
            
        self.metadata[filename]["bookmarks"].append(round(elapsed, 2))
        self.save_metadata()
        self.root.ids.status_label.text = f"Marked at {self.format_time(elapsed)}"

    def toggle_pause(self):
        if not self.is_recording: return
        if not self.is_paused:
            self.is_paused = True
            self.accumulated_time += (time.time() - self.record_start_time)
            if platform == 'android':
                try: self.recorder.pause()
                except: pass
            else:
                if self.recorder_thread: self.recorder_thread.is_paused = True
            self.root.ids.pause_btn.text = "Resume"
            self.root.ids.status_label.text = "Paused"
        else:
            self.is_paused = False
            self.record_start_time = time.time()
            if platform == 'android':
                try: self.recorder.resume()
                except: pass
            else:
                if self.recorder_thread: self.recorder_thread.is_paused = False
            self.root.ids.pause_btn.text = "Pause"
            self.root.ids.status_label.text = f"Recording (Part {self.split_part_number})..."

    def start_recording(self):
        self.is_recording = True
        self.is_paused = False
        self.accumulated_time = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_part{self.split_part_number}" if self.split_part_number > 1 else ""
        
        lat, lon = None, None
        if platform == 'android':
            lat, lon = self.get_android_gps()
        
        if platform == 'android':
            fmt = self.root.ids.format_spinner.text
            ext = ".flac" if "FLAC" in fmt else ".m4a"
            self.current_file = os.path.join(self.storage_dir, f"rec_{timestamp}{suffix}{ext}")
            mic_src = self.root.ids.mic_source_spinner.text
            channels_str = self.root.ids.channels_spinner.text
            self.start_recording_android(self.current_file, mic_src, fmt, channels_str)
            self.vu_event = Clock.schedule_interval(self.update_vu_android, 0.1)
            self.acquire_wakelock()
        else:
            self.current_file = os.path.join(self.storage_dir, f"rec_{timestamp}{suffix}.wav")
            selected_filter = self.root.ids.filter_spinner.text
            is_sound_active = self.root.ids.sound_active_switch.active
            is_auto_trim = self.root.ids.auto_trim_switch.active
            channels = 2 if self.root.ids.channels_spinner.text == "Stereo" else 1
            threshold = int(self.root.ids.threshold_slider.value)
            gain = self.root.ids.gain_slider.value / 10.0
            is_pre_rec = self.root.ids.pre_rec_switch.active
            
            self.recorder_thread = DesktopRecorder(self.current_file, selected_filter, is_sound_active, is_auto_trim, channels, threshold, gain, is_pre_rec)
            self.recorder_thread.start()
            self.vu_event = Clock.schedule_interval(self.update_vu_desktop, 0.05)

        filename = os.path.basename(self.current_file)
        if filename not in self.metadata: self.metadata[filename] = {"bookmarks": [], "starred": False, "category": "General"}
        if lat and lon:
            self.metadata[filename]["gps"] = {"lat": lat, "lon": lon}
            self.root.ids.status_label.text = f"Recording... GPS Tagged. (Part {self.split_part_number})"
        else:
            self.root.ids.status_label.text = f"Recording... (Part {self.split_part_number})"
        self.save_metadata()

        self.root.ids.record_btn.text = "Stop"
        self.root.ids.record_btn.background_color = (0.5, 0.5, 0.5, 1)
        self.root.ids.pause_btn.disabled = False
        self.root.ids.mark_btn.disabled = False
        self.record_start_time = time.time()
        self.timer_event = Clock.schedule_interval(self.update_timer, 1)

    def stop_recording(self):
        self.is_recording = False
        if self.is_paused: self.is_paused = False
        if platform == 'android':
            self.stop_recording_android()
            self.release_wakelock()
        else:
            if self.recorder_thread:
                self.recorder_thread.stop()
                self.recorder_thread.join()

        self.root.ids.record_btn.text = "Record"
        self.root.ids.record_btn.background_color = (0.8, 0.2, 0.2, 1)
        self.root.ids.pause_btn.disabled = True
        self.root.ids.mark_btn.disabled = True
        self.root.ids.status_label.text = "Recording Saved!"
        self.root.ids.timer_label.text = "00:00"
        self.root.ids.timer_label.color = self.theme_text
        self.root.ids.vu_meter.update_bars(np.zeros(10))
        self.auto_stop_limit = 0
        self.auto_split_limit = 0
        self.split_part_number = 1
        
        if self.timer_event: self.timer_event.cancel()
        if self.vu_event: self.vu_event.cancel()
        self.refresh_recordings_list()

    # --- Mini Player & FX Logic ---
    def play_recording(self, filepath, item_widget):
        if self.currently_playing_item == item_widget:
            self.toggle_mini_player_playback()
            return

        if self.currently_playing_item:
            self.currently_playing_item.bg_color = self.theme_card
        
        item_widget.bg_color = (1.0, 0.8, 0.6, 1)
        self.currently_playing_item = item_widget
        self.clear_loop()

        if self.current_sound:
            self.current_sound.stop()
            self.current_sound.unload()

        self.current_sound = SoundLoader.load(filepath)
        if self.current_sound:
            self.show_mini_player(os.path.basename(filepath))
            self.change_playback_speed()
            self.current_sound.play()
            self.is_mini_player_playing = True
            length = self.current_sound.length
            self.root.ids.seek_slider.max = length
            self.root.ids.total_duration.text = self.format_time(length)
            
            marks = self.metadata.get(os.path.basename(filepath), {}).get("bookmarks", [])
            self.root.ids.prev_mark_btn.disabled = len(marks) == 0
            self.root.ids.next_mark_btn.disabled = len(marks) == 0
            
            if self.playback_event: self.playback_event.cancel()
            self.playback_event = Clock.schedule_interval(self.update_playback, 0.1)
            self.current_sound.bind(on_stop=self.on_sound_stop)
        else:
            self.root.ids.status_label.text = "Error loading audio"

    def show_mini_player(self, title):
        self.root.ids.player_title.text = title
        self.root.ids.player_bar.size_hint_y = 0.25
        self.root.ids.player_bar.opacity = 1
        self.root.ids.player_bar.disabled = False

    def hide_mini_player(self):
        self.root.ids.player_bar.size_hint_y = 0
        self.root.ids.player_bar.opacity = 0
        self.root.ids.player_bar.disabled = True
        if self.currently_playing_item:
            self.currently_playing_item.bg_color = self.theme_card
            self.currently_playing_item = None

    def toggle_mini_player_playback(self):
        if not self.current_sound: return
        if self.is_mini_player_playing:
            self.current_sound.stop()
            self.is_mini_player_playing = False
        else:
            self.current_sound.play()
            self.is_mini_player_playing = True
            self.playback_event = Clock.schedule_interval(self.update_playback, 0.1)

    def stop_mini_player(self):
        if self.current_sound: self.current_sound.stop()
        self.on_sound_stop(None)

    def change_playback_speed(self):
        if not self.current_sound: return
        speed_str = self.root.ids.speed_spinner.text
        try:
            speed = float(speed_str.replace('x', ''))
            self.current_sound.pitch = speed 
        except: pass

    def set_loop_point(self, point):
        if not self.current_sound: return
        pos = self.current_sound.get_pos()
        if point == 'a':
            self.loop_a = pos
            self.root.ids.status_label.text = f"Loop A set: {self.format_time(pos)}"
        else:
            self.loop_b = pos
            self.root.ids.status_label.text = f"Loop B set: {self.format_time(pos)}"

    def clear_loop(self):
        self.loop_a = None
        self.loop_b = None

    def save_segment(self):
        if not self.current_sound: return
        if self.loop_a is None or self.loop_b is None or self.loop_b <= self.loop_a:
            self.root.ids.status_label.text = "Set A and B points first!"
            return
            
        filepath = self.current_sound.source
        self.root.ids.status_label.text = "Extracting segment..."
        threading.Thread(target=self.process_extract_segment, args=(filepath, self.loop_a, self.loop_b)).start()

    def process_extract_segment(self, filepath, start_s, end_s):
        if platform != 'android':
            import wave
            try:
                with wave.open(filepath, 'rb') as wf:
                    fs = wf.getframerate()
                    channels = wf.getnchannels()
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).copy()

                start_idx = int(start_s * fs)
                end_idx = int(end_s * fs)
                segment = audio[start_idx:end_idx]

                base, ext = os.path.splitext(filepath)
                new_filepath = f"{base}_seg{int(start_s)}-{int(end_s)}{ext}"
                with wave.open(new_filepath, 'wb') as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(2)
                    wf.setframerate(fs)
                    wf.writeframes(segment.astype('int16').tobytes())
                    
                Clock.schedule_once(lambda dt: self.refresh_recordings_list(), 0)
                self.root.ids.status_label.text = "Segment saved!"
            except:
                self.root.ids.status_label.text = "Trim failed"

    def skip_bookmark(self, direction):
        if not self.current_sound: return
        marks = self.metadata.get(os.path.basename(self.current_sound.source), {}).get("bookmarks", [])
        if not marks: return
        current_pos = self.current_sound.get_pos()
        marks.sort()
        if direction == 1:
            for m in marks:
                if m > current_pos + 0.5:
                    self.current_sound.seek(m); return
            self.current_sound.seek(marks[-1])
        else:
            for m in reversed(marks):
                if m < current_pos - 0.5:
                    self.current_sound.seek(m); return
            self.current_sound.seek(0)

    def update_playback(self, dt):
        if self.current_sound and self.is_mini_player_playing:
            pos = self.current_sound.get_pos()
            if self.loop_a is not None and self.loop_b is not None and self.loop_b > self.loop_a:
                if pos >= self.loop_b:
                    self.current_sound.seek(self.loop_a)
                    pos = self.loop_a
            self.root.ids.seek_slider.value = pos
            self.root.ids.current_pos.text = self.format_time(pos)

    def seek_audio(self, value):
        if self.current_sound: self.current_sound.seek(value)

    def on_sound_stop(self, instance):
        if self.playback_event: self.playback_event.cancel()
        self.is_mini_player_playing = False
        self.root.ids.seek_slider.value = 0
        self.root.ids.current_pos.text = "00:00"
        self.hide_mini_player()

    def cycle_category(self, filepath):
        filename = os.path.basename(filepath)
        cats = ["General", "Birds", "Insects", "Mammals", "Weather"]
        if filename not in self.metadata:
            self.metadata[filename] = {"category": "General"}
        current_cat = self.metadata[filename].get("category", "General")
        next_idx = (cats.index(current_cat) + 1) % len(cats)
        self.metadata[filename]["category"] = cats[next_idx]
        self.save_metadata()
        self.refresh_recordings_list()

    def share_file(self, filepath):
        if platform == 'android': self.share_file_android(filepath)
        else: self.open_file_desktop(filepath)

    def share_file_android(self, filepath):
        try:
            from jnius import autoclass, cast
            Intent = autoclass('android.content.Intent')
            Uri = autoclass('android.net.Uri')
            File = autoclass('java.io.File')
            intent = Intent()
            intent.setAction(Intent.ACTION_SEND)
            intent.setType("audio/*")
            intent.putExtra(Intent.EXTRA_STREAM, cast('android.os.Parcelable', Uri.fromFile(File(filepath))))
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            PythonActivity.mActivity.startActivity(Intent.createChooser(intent, "Share Recording"))
        except: pass

    def open_map(self, filepath):
        filename = os.path.basename(filepath)
        gps = self.metadata.get(filename, {}).get("gps")
        if not gps: return
        lat, lon = gps["lat"], gps["lon"]
        if platform == 'android':
            try:
                from jnius import autoclass
                Intent = autoclass('android.content.Intent')
                Uri = autoclass('android.net.Uri')
                intent = Intent(Intent.ACTION_VIEW, Uri.parse(f"geo:{lat},{lon}?q={lat},{lon}(Recording Location)"))
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
                PythonActivity.mActivity.startActivity(intent)
            except: pass
        else:
            webbrowser.open(f"https://www.google.com/maps?q={lat},{lon}")

    def zip_and_share_all(self):
        self.root.ids.status_label.text = "Zipping files..."
        zip_path = os.path.join(self.storage_dir, "field_recordings_backup.zip")
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if os.path.exists(self.metadata_file):
                    zipf.write(self.metadata_file, "metadata.json")
                for f in os.listdir(self.storage_dir):
                    if f.endswith((".wav", ".mp4", ".m4a", ".flac")):
                        zipf.write(os.path.join(self.storage_dir, f), f)
            self.root.ids.status_label.text = "Backup created!"
            self.share_file(zip_path)
        except:
            self.root.ids.status_label.text = "Zip failed"

    def open_file_desktop(self, filepath):
        try:
            if platform == 'win': subprocess.Popen(f'explorer /select,"{filepath}"')
            elif platform == 'mac': subprocess.Popen(['open', '-R', filepath])
            else: subprocess.Popen(['xdg-open', os.path.dirname(filepath)])
        except: pass

    def open_rename_popup(self, filepath, old_name):
        RenamePopup(self, filepath, old_name).open()

    def open_notes_popup(self, filepath):
        filename = os.path.basename(filepath)
        old_notes = self.metadata.get(filename, {}).get("notes", "")
        NotesPopup(self, filepath, old_notes).open()

    def open_process_popup(self, filepath):
        ProcessPopup(self, filepath).open()

    def open_batch_popup(self):
        BatchPopup(self).open()

    # --- Optimized Audio Processing ---
    def process_audio_file(self, filepath, process_type):
        if platform != 'android':
            import wave
            try:
                with wave.open(filepath, 'rb') as wf:
                    fs = wf.getframerate()
                    channels = wf.getnchannels()
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).copy()

                if process_type == 'normalize':
                    max_val = np.max(np.abs(audio))
                    if max_val > 0:
                        audio = audio * (32767 / max_val)
                        audio = np.clip(audio, -32767, 32767)
                        
                elif process_type == 'highpass':
                    # O(N) Moving average highpass instead of O(N^2) np.convolve
                    window_size = int(fs / 80)
                    cumsum = np.cumsum(np.insert(audio, 0, 0))
                    lowpass = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
                    lowpass = np.pad(lowpass, (window_size//2, window_size - window_size//2), mode='edge')
                    audio = audio - lowpass
                    audio = np.clip(audio, -32767, 32767)

                base, ext = os.path.splitext(filepath)
                new_filepath = f"{base}_{process_type}{ext}"
                with wave.open(new_filepath, 'wb') as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(2)
                    wf.setframerate(fs)
                    wf.writeframes(audio.astype('int16').tobytes())
                    
                Clock.schedule_once(lambda dt: self.refresh_recordings_list(), 0)
                self.root.ids.status_label.text = f"Processed file saved!"
            except:
                self.root.ids.status_label.text = "Processing failed"

    def batch_process_all(self, process_type):
        if platform != 'android':
            for f in os.listdir(self.storage_dir):
                if f.endswith(".wav"):
                    self.process_audio_file(os.path.join(self.storage_dir, f), process_type)
            self.root.ids.status_label.text = "Batch Processing Complete!"

    def get_duration(self, filepath):
        try:
            sound = SoundLoader.load(filepath)
            if sound and sound.length > 0: return sound.length
        except: pass
        return 0

    def refresh_recordings_list(self, *args):
        list_layout = self.root.ids.recordings_list
        list_layout.clear_widgets()
        search_text = self.root.ids.search_input.text.lower()
        filter_cat = self.root.ids.cat_filter_spinner.text
        
        files = [f for f in os.listdir(self.storage_dir) if f.endswith((".wav", ".mp4", ".m4a", ".flac"))]
        files = [f for f in files if search_text in f.lower()]
        
        filtered_files = []
        for f in files:
            file_cat = self.metadata.get(f, {}).get("category", "General")
            if filter_cat == "All Categories" or file_cat == filter_cat:
                filtered_files.append(f)
        filtered_files.sort(reverse=True)

        if not filtered_files:
            list_layout.add_widget(Label(text="No recordings found", size_hint_y=None, height=40, color=self.theme_subtext))
            return

        for f in filtered_files:
            self.add_list_item(list_layout, f)

    def add_list_item(self, list_layout, filename):
        filepath = os.path.join(self.storage_dir, filename)
        duration = self.get_duration(filepath)
        marks = self.metadata.get(filename, {}).get("bookmarks", [])
        has_gps = "gps" in self.metadata.get(filename, {})
        has_notes = "notes" in self.metadata.get(filename, {}) and self.metadata[filename]["notes"].strip()
        category = self.metadata.get(filename, {}).get("category", "General")
        
        mark_str = f" | 🔖 {len(marks)}" if marks else ""
        gps_str = " | 📍" if has_gps else ""
        notes_str = " | 📝" if has_notes else ""
        
        display_name = f"{filename} ({self.format_time(duration)}){mark_str}{gps_str}{notes_str}"
        list_layout.add_widget(RecordingItem(filepath, display_name, self, category=category, has_gps=has_gps))

    def delete_recording(self, filepath, item_widget):
        if self.currently_playing_item == item_widget: self.stop_mini_player()
        if os.path.exists(filepath):
            os.remove(filepath)
            filename = os.path.basename(filepath)
            if filename in self.metadata:
                del self.metadata[filename]; self.save_metadata()
            self.root.ids.status_label.text = "Deleted recording"
            self.refresh_recordings_list()

    def start_recording_android(self, filepath, mic_src_str, format_str, channels_str):
        from jnius import autoclass
        self.MediaRecorder = autoclass('android.media.MediaRecorder')
        self.recorder = self.MediaRecorder()
        
        source = 1 
        if "Unprocessed" in mic_src_str:
            try: source = self.MediaRecorder.AudioSource.UNPROCESSED
            except: source = 9 
        elif "Voice Rec" in mic_src_str:
            source = 6
            
        self.recorder.setAudioSource(source)
        
        if "FLAC" in format_str:
            self.recorder.setOutputFormat(self.MediaRecorder.OutputFormat.FLAC)
            self.recorder.setAudioEncoder(self.MediaRecorder.AudioEncoder.FLAC)
            self.recorder.setAudioEncodingBitRate(320000)
        else:
            self.recorder.setOutputFormat(self.MediaRecorder.OutputFormat.MPEG_4)
            self.recorder.setAudioEncoder(self.MediaRecorder.AudioEncoder.AAC)
            self.recorder.setAudioEncodingBitRate(320000)
            
        self.recorder.setAudioSamplingRate(48000)
        
        if channels_str == "Stereo":
            self.recorder.setAudioChannels(2)
        else:
            self.recorder.setAudioChannels(1)
            
        self.recorder.setOutputFile(filepath)
        self.recorder.prepare()
        self.recorder.start()

    def stop_recording_android(self):
        if self.recorder:
            try: self.recorder.stop()
            except: pass
            self.recorder.release()
            self.recorder = None

# --- Optimized Desktop Threading & DSP Engine ---
class DesktopRecorder(threading.Thread):
    def __init__(self, filename, audio_filter="No Filter", sound_activation=False, auto_trim=False, channels=1, threshold=50, gain=1.0, pre_rec=False):
        super().__init__()
        self.filename = filename
        self.audio_filter = audio_filter
        self.sound_activation = sound_activation
        self.auto_trim = auto_trim
        self.channels = channels
        self.is_running = True
        self.is_paused = False
        self.current_volume = 0
        self.is_silent = False
        self.is_clipping = False
        self.silence_threshold = threshold 
        self.gain = gain
        self.pre_rec_enabled = pre_rec
        self.spectrum_bars = np.zeros(10)
        self.pre_buffer = []
        self.max_pre_buffer_chunks = int(3.0 * 48000 / 1024)

    def run(self):
        import sounddevice as sd
        import wave
        self.fs = 48000 
        self.audio_data = []

        def callback(indata, frames, time_info, status):
            if self.is_running and not self.is_paused:
                gained_data = indata * self.gain
                gained_data = np.clip(gained_data, -32767, 32767)
                
                # Flatten to 1D for faster processing if mono
                mono_data = gained_data[:, 0] if self.channels == 2 else gained_data.flatten()
                rms = np.sqrt(np.mean(np.square(mono_data)))
                self.current_volume = int(rms * 3)
                
                self.is_clipping = np.max(np.abs(mono_data)) > 30000
                
                try:
                    fft_vals = np.abs(np.fft.rfft(mono_data))
                    bands = np.logspace(np.log10(20), np.log10(20000), 11)
                    bars = []
                    for i in range(10):
                        start = int(bands[i] / (self.fs/2) * len(fft_vals))
                        end = int(bands[i+1] / (self.fs/2) * len(fft_vals))
                        if start == end: end = start + 1
                        val = np.max(fft_vals[start:end]) / 5000.0 if start < len(fft_vals) else 0
                        bars.append(min(1.0, val))
                    self.spectrum_bars = np.array(bars)
                except: pass
                
                if self.sound_activation:
                    if rms < self.silence_threshold:
                        self.is_silent = True
                        return 
                    else:
                        self.is_silent = False
                
                self.audio_data.append(mono_data.copy())
            
            if self.pre_rec_enabled and not self.audio_data:
                self.pre_buffer.append(indata.copy())
                if len(self.pre_buffer) > self.max_pre_buffer_chunks:
                    self.pre_buffer.pop(0)

        self.stream = sd.InputStream(callback=callback, channels=self.channels, samplerate=self.fs, dtype='int16')
        self.stream.start()
        while self.is_running: sd.sleep(50)
        self.stream.stop()
        self.stream.close()

        if self.pre_rec_enabled and self.pre_buffer:
            self.audio_data = [np.array(p).flatten() for p in self.pre_buffer] + self.audio_data

        if self.audio_data:
            audio = np.concatenate(self.audio_data)
            
            if self.auto_trim and len(audio) > 0:
                abs_audio = np.abs(audio)
                indices = np.where(abs_audio > 150)[0] 
                if len(indices) > 0:
                    start_idx = max(0, indices[0] - 4410) 
                    end_idx = min(len(audio), indices[-1] + 4410) 
                    audio = audio[start_idx:end_idx]
            
            if self.audio_filter == "Wind Reduction" or self.audio_filter == "highpass":
                # O(N) Moving average highpass
                window_size = int(self.fs / 150) if self.audio_filter == "Wind Reduction" else int(self.fs / 80)
                cumsum = np.cumsum(np.insert(audio, 0, 0))
                lowpass = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
                lowpass = np.pad(lowpass, (window_size//2, window_size - window_size//2), mode='edge')
                audio = audio - lowpass
                audio = np.clip(audio, -32767, 32767)
                
            elif self.audio_filter == "Telephone":
                window_size = 10
                cumsum = np.cumsum(np.insert(audio, 0, 0))
                lowpass = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
                lowpass = np.pad(lowpass, (window_size//2, window_size - window_size//2), mode='edge')
                audio = audio - lowpass
                audio = np.clip(audio, -32767, 32767)
                
            elif self.audio_filter == "Echo":
                delay_samples = int(0.3 * self.fs)
                echo = np.zeros_like(audio)
                echo[delay_samples:] = audio[:-delay_samples] * 0.6
                audio = audio + echo
                audio = np.clip(audio, -32767, 32767)

            wf = wave.open(self.filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.fs)
            wf.writeframes(audio.astype('int16').tobytes())
            wf.close()

    def stop(self):
        self.is_running = False

if __name__ == '__main__':
    RecorderApp().run()