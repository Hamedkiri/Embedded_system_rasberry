# main.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAIN — Wiring UI ↔ CORE
- Configure environnement (Qt scaling, threads)
- Crée QApplication, MainWindow (ui_app.py), SessionManager (core_app.py)
- Contrôleur : relie les signaux UI aux actions core, et inversement.
"""

import os
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

# Threads (Pi)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ui_app import MainWindow, EmailRequest
from core_app import SessionManager, send_email_smtp
from PySide6.QtCore import Qt, QTimer


class AppController:
    """
    Contrôleur :
    - reçoit les intentions UI
    - configure le core
    - pousse frames/pixmap & messages vers l’UI
    """

    def __init__(self, ui: MainWindow, core: SessionManager):
        self.ui = ui
        self.core = core

        # Timer UI → tick core
        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._tick)

        self._connect_signals()

        # Bootstrap lists
        self.reload_models()
        self.reload_cameras()

    def _connect_signals(self):
        # UI → Controller
        self.ui.modelsReloadRequested.connect(self.reload_models)
        self.ui.camerasReloadRequested.connect(self.reload_cameras)

        self.ui.modelSelectionChanged.connect(self.on_model_changed)
        self.ui.tasksSelectionChanged.connect(self.on_tasks_changed)

        self.ui.startRequested.connect(self.start_session)
        self.ui.stopRequested.connect(self.stop_session)
        self.ui.recordToggled.connect(self.toggle_recording)
        self.ui.fullscreenRequested.connect(self.ui._toggle_fullscreen)

        self.ui.classesPathChanged.connect(self.on_classes_path_changed)
        self.ui.outputDirChanged.connect(self.on_output_dir_changed)

        self.ui.openRecordingRequested.connect(self.open_recording)
        self.ui.transferRequested.connect(self.transfer_files)
        self.ui.emailRequested.connect(self.send_email)

        # Core → Controller/UI
        self.core.tasksLoaded.connect(lambda tasks, order: self.ui.set_tasks(tasks, order))
        self.core.deviceChanged.connect(lambda d: self.ui.set_device_text(f"Device: {d}"))
        self.core.frameReady.connect(self.ui.show_frame_pixmap)
        self.core.error.connect(lambda msg: self.ui.show_error("Erreur", msg))
        self.core.info.connect(lambda msg: self.ui.show_info("Info", msg))

    # ───────────────────────── Bootstrap lists ─────────────────────────

    def reload_models(self):
        entries = self.core.list_models()
        self.ui.set_models(entries)

    def reload_cameras(self):
        entries = self.core.list_cameras()
        self.ui.set_cameras(entries)

    # ───────────────────────── UI changes ─────────────────────────

    def on_model_changed(self, model_path: str):
        # Charger tasks du modèle pour UI (sans démarrer la session)
        self.core.all_tasks = self.core.load_tasks_for_model(model_path)
        self.core.selected_tasks = list(self.core.all_tasks.keys())
        self.ui.set_tasks(self.core.all_tasks, self.core.selected_tasks)

    def on_tasks_changed(self, selected: list):
        self.core.selected_tasks = list(selected)
        # Mise à jour UI
        self.ui.set_tasks(self.core.all_tasks, self.core.selected_tasks)

    def on_classes_path_changed(self, path: str):
        self.core.classes_json_path = path
        # Recalculer tasks pour le modèle courant
        model_path = self.ui.modelCombo.currentData() or self.ui.modelCombo.currentText()
        self.on_model_changed(str(model_path))

    def on_output_dir_changed(self, path: str):
        self.core.output_dir = Path(path).resolve()
        self.core.output_dir.mkdir(parents=True, exist_ok=True)

    # ───────────────────────── Start/Stop ─────────────────────────

    def start_session(self):
        model_path = self.ui.modelCombo.currentData() or self.ui.modelCombo.currentText()
        camera_data = self.ui.cameraCombo.currentData()

        # Settings runtime depuis UI
        self.core.target_fps = int(self.ui.fpsSpin.value())
        self.core.prob_threshold = float(self.ui.threshSpin.value())
        self.core.infer_stride = int(self.ui.inferEverySpin.value())
        self.core.show_speed = bool(self.ui.speedCheck.isChecked())
        self.core.export_format = self.ui.formatCombo.currentText()
        self.core.session_name = self.ui.sessionEdit.text().strip()
        if not self.core.session_name:
            # session name sera déterminé au moment record/export si vide
            self.core.session_name = ""

        try:
            self.core.start_session(str(model_path), camera_data)
        except Exception as e:
            self.ui.show_error("Démarrage", str(e))
            return

        # timer d’affichage
        self.timer.start(int(1000 / max(1, self.core.target_fps)))
        self.ui.set_running_state(True)

    def stop_session(self):
        self.timer.stop()
        try:
            self.core.stop_session()
        except Exception as e:
            self.ui.show_error("Stop", str(e))
        self.ui.set_running_state(False)
        self.ui.set_recording_state(False)

    def toggle_recording(self, checked: bool):
        if checked:
            self.core.start_recording()
        else:
            self.core.stop_recording()
        self.ui.set_recording_state(checked)

    # ───────────────────────── Tick display ─────────────────────────

    def _tick(self):
        # FAST_SCALE env (comme ton code)
        fast = os.getenv("FAST_SCALE", "1") == "1"
        self.core.tick(target_qsize=self.ui.videoLabel.size(), fast_scale=fast)

    # ───────────────────────── Extras ─────────────────────────

    def open_recording(self, video_path: str):
        # (Optionnel) Tu peux ré-intégrer ton mode playback ici,
        # ou le laisser comme fonctionnalité future.
        self.ui.show_info("Lecture", f"Ouverture demandée: {video_path}\n(Playback à rebrancher si souhaité.)")

    def transfer_files(self, files: list, dest: str):
        n = self.core.copy_files(files, dest)
        self.ui.show_info("Transfert", f"{n} fichier(s) copié(s) vers\n{dest}")

    def send_email(self, req: EmailRequest):
        # si tu veux des pièces jointes auto (dernière session), tu peux ici:
        # atts = req.attachments or self.core.collect_latest_session_files()
        ok, msg = send_email_smtp(req.to_list, req.subject, req.body, req.attachments)
        if ok:
            self.ui.show_info("E-mail", msg)
        else:
            self.ui.show_error("E-mail", msg)


def main():
    app = QApplication(sys.argv)

    root = Path(__file__).resolve().parent
    ui = MainWindow()
    core = SessionManager(root_dir=root)

    ctrl = AppController(ui, core)

    ui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
