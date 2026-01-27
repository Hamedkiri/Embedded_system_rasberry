# ui_app.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI ONLY — PySide6 (Responsive / no-overflow)

✅ Fixes:
- Plus de widgets "vides" (plus de partage de widgets entre Desktop et Drawer)
- Drawer construit 1 seule fois, zone "tâches" seule est rafraîchie
- Anti overflow horizontal dans le drawer (clamp du contenu)
- Bouton = AGRANDIR / RESTAURER (showMaximized/showNormal)
- Vrai plein écran = F11 (showFullScreen)
- Panneau options ne peut plus s’écraser à 0 (min width controls)
- Re-application robuste des tailles du splitter après changement d’état (timers)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

from PySide6.QtCore import (
    Qt, QTimer, Signal, QRect, QPropertyAnimation, QEasingCurve, QEvent
)
from PySide6.QtGui import QPixmap, QFont, QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QComboBox, QLineEdit,
    QFileDialog, QCheckBox, QSpinBox, QDoubleSpinBox, QHBoxLayout, QVBoxLayout,
    QGroupBox, QMessageBox, QDialog, QFormLayout, QDialogButtonBox,
    QTextEdit, QScrollArea, QFrame, QLayout, QToolButton,
    QGraphicsDropShadowEffect, QSizePolicy, QSplitter
)

# ──────────────────────────────────────────────────────────────────────────────
# Data structures UI → Controller
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EmailRequest:
    to_list: List[str]
    subject: str
    body: str
    attachments: List[Path]


# ──────────────────────────────────────────────────────────────────────────────
# Small UI widgets
# ──────────────────────────────────────────────────────────────────────────────

class ClickableScrim(QFrame):
    """Voile semi-transparent cliquable pour fermer le drawer."""
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)

    def mousePressEvent(self, e):
        self.clicked.emit()
        e.accept()


class DrawerSection(QGroupBox):
    """Section groupée dans le drawer."""
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(10)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)


class SideDrawer(QFrame):
    """
    Drawer latéral (côté gauche).
    Anti overflow horizontal:
    - ScrollArea widgetResizable
    - Clamp maxWidth du contenu au viewport
    """
    closed = Signal()

    def __init__(self, parent, side: str = "left", title: str = "Options"):
        super().__init__(parent)
        assert side in ("left", "right")
        self.side = side
        self.setObjectName("drawerLeft" if side == "left" else "drawerRight")
        self.setVisible(False)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(22)
        shadow.setOffset(-2 if side == "right" else 2, 0)
        shadow.setColor(Qt.black)
        self.setGraphicsEffect(shadow)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setObjectName("drawerHeader")
        h = QHBoxLayout(header)
        h.setContentsMargins(12, 12, 12, 12)
        h.setSpacing(8)

        self.titleLbl = QLabel(title)
        self.titleLbl.setObjectName("drawerTitle")
        self.subtitleLbl = QLabel("Configuration")
        self.subtitleLbl.setObjectName("drawerSubtitle")
        self.subtitleLbl.setWordWrap(True)
        self.subtitleLbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        titleWrap = QVBoxLayout()
        titleWrap.setContentsMargins(0, 0, 0, 0)
        titleWrap.setSpacing(0)
        titleWrap.addWidget(self.titleLbl)
        titleWrap.addWidget(self.subtitleLbl)

        closeBtn = QToolButton()
        closeBtn.setText("✕")
        closeBtn.setCursor(Qt.PointingHandCursor)
        closeBtn.clicked.connect(self.close_drawer)

        h.addLayout(titleWrap)
        h.addStretch()
        h.addWidget(closeBtn)
        root.addWidget(header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.viewport().installEventFilter(self)

        self.inner = QWidget()
        self.inner.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.inner.setLayout(QVBoxLayout())
        self.inner.layout().setContentsMargins(12, 12, 12, 12)
        self.inner.layout().setSpacing(12)
        self.scroll.setWidget(self.inner)
        root.addWidget(self.scroll)

        footer = QFrame()
        footer.setObjectName("drawerFooter")
        f = QHBoxLayout(footer)
        f.setContentsMargins(12, 10, 12, 12)
        f.setSpacing(8)

        self.primaryBtn = QPushButton("Fermer")
        self.primaryBtn.setProperty("kind", "primary")
        self.primaryBtn.clicked.connect(self.close_drawer)
        f.addStretch()
        f.addWidget(self.primaryBtn)
        root.addWidget(footer)

        self.anim = QPropertyAnimation(self, b"geometry", self)
        self.anim.setDuration(220)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        self._geom_show = QRect()
        self._geom_hide = QRect()

        self._closing_hook_connected = False

    def eventFilter(self, obj, event):
        if obj is self.scroll.viewport() and event.type() in (QEvent.Resize, QEvent.Show):
            self._clamp_inner_width()
        return super().eventFilter(obj, event)

    def _clamp_inner_width(self):
        vpw = self.scroll.viewport().width()
        if vpw <= 0:
            return
        maxw = max(220, vpw - 2)
        self.inner.setMaximumWidth(maxw)
        lay = self.inner.layout()
        for i in range(lay.count()):
            it = lay.itemAt(i)
            w = it.widget()
            if w:
                w.setMaximumWidth(maxw)

    def setContent(self, widget_or_layout):
        lay = self.inner.layout()
        while lay.count():
            child = lay.takeAt(0)
            w = child.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

        if isinstance(widget_or_layout, QLayout):
            wrap = QWidget()
            wrap.setLayout(widget_or_layout)
            wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            lay.addWidget(wrap)
        else:
            widget_or_layout.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            lay.addWidget(widget_or_layout)

        lay.addStretch(1)
        QTimer.singleShot(0, self._clamp_inner_width)

    def _target_rects(self):
        central = self.parent()
        dw = max(320, min(560, int(central.width() * 0.82)))
        ch = central.height()
        if self.side == "right":
            show = QRect(central.width() - dw, 0, dw, ch)
            hide = QRect(central.width(), 0, dw, ch)
        else:
            show = QRect(0, 0, dw, ch)
            hide = QRect(-dw, 0, dw, ch)
        return show, hide

    def layout_now(self, initial=False):
        show, hide = self._target_rects()
        self._geom_show = show
        self._geom_hide = hide
        if initial or not self.isVisible():
            self.setGeometry(hide)

    def open_drawer(self):
        self.layout_now()
        self.setVisible(True)
        self.raise_()
        self.anim.stop()
        self.anim.setStartValue(self._geom_hide)
        self.anim.setEndValue(self._geom_show)
        self.anim.start()
        QTimer.singleShot(0, self._clamp_inner_width)

    def close_drawer(self):
        self.layout_now()
        self.anim.stop()
        self.anim.setStartValue(self._geom_show)
        self.anim.setEndValue(self._geom_hide)
        self.anim.start()
        if not self._closing_hook_connected:
            self.anim.finished.connect(self._after_close)
            self._closing_hook_connected = True

    def _after_close(self):
        self.setVisible(False)
        self.closed.emit()


class TaskSelectionDialog(QDialog):
    """Dialog desktop simple pour choisir les tâches (checkbox)."""
    def __init__(self, parent, all_tasks: Dict[str, List[str]], selected: List[str]):
        super().__init__(parent)
        self.setWindowTitle("Sélection des tâches")

        self.all_tasks = all_tasks
        self.checks: Dict[str, QCheckBox] = {}

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        for t in all_tasks.keys():
            cb = QCheckBox(t)
            cb.setChecked(t in selected)
            self.checks[t] = cb
            v.addWidget(cb)

        row = QHBoxLayout()
        btnAll = QPushButton("Tout")
        btnNone = QPushButton("Aucun")
        btnAll.setProperty("kind", "secondary")
        btnNone.setProperty("kind", "secondary")
        btnAll.clicked.connect(lambda: [c.setChecked(True) for c in self.checks.values()])
        btnNone.clicked.connect(lambda: [c.setChecked(False) for c in self.checks.values()])
        row.addWidget(btnAll)
        row.addWidget(btnNone)
        row.addStretch(1)
        v.addLayout(row)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        v.addWidget(bb)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)

    def selected_tasks(self) -> List[str]:
        return [t for t, cb in self.checks.items() if cb.isChecked()]


class EmailDialog(QDialog):
    def __init__(self, parent=None, default_attachments: Optional[List[Path]] = None):
        super().__init__(parent)
        self.setWindowTitle("Envoyer par e-mail")
        self.resize(560, 380)

        self.attachments: List[Path] = list(default_attachments or [])

        self.toEdit = QLineEdit()
        self.subjEdit = QLineEdit("Fichiers de détection")
        self.bodyEdit = QTextEdit("Veuillez trouver ci-joint les fichiers de détection.")

        self.attLabel = QLabel(self._att_text())
        self.attLabel.setWordWrap(True)
        self.attLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        addBtn = QPushButton("Ajouter pièces jointes…")
        clearBtn = QPushButton("Vider la liste")
        addBtn.setProperty("kind", "secondary")
        clearBtn.setProperty("kind", "secondary")
        addBtn.clicked.connect(self._add_attachments)
        clearBtn.clicked.connect(self._clear_attachments)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.addRow("À (séparés par ,) :", self.toEdit)
        form.addRow("Sujet :", self.subjEdit)
        form.addRow("Message :", self.bodyEdit)

        attRow = QHBoxLayout()
        attRow.addWidget(addBtn)
        attRow.addWidget(clearBtn)
        attRow.addStretch(1)

        v = QVBoxLayout(self)
        v.addLayout(form)
        v.addWidget(QLabel("Pièces jointes :"))
        v.addWidget(self.attLabel)
        v.addLayout(attRow)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        v.addWidget(self.buttons)

    def _att_text(self) -> str:
        return "(aucune)" if not self.attachments else "\n".join(str(p) for p in self.attachments)

    def _add_attachments(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Choisir des fichiers", "", "Tous (*.*)")
        for f in files:
            p = Path(f)
            if p.exists():
                self.attachments.append(p)
        self.attLabel.setText(self._att_text())

    def _clear_attachments(self):
        self.attachments.clear()
        self.attLabel.setText(self._att_text())

    def build_request(self) -> EmailRequest:
        to_list = [t.strip() for t in self.toEdit.text().split(",") if t.strip()]
        subject = self.subjEdit.text().strip()
        body = self.bodyEdit.toPlainText()
        return EmailRequest(to_list=to_list, subject=subject, body=body, attachments=self.attachments.copy())


# ──────────────────────────────────────────────────────────────────────────────
# Main Window (UI ONLY)
# ──────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    # --- Signals UI → Controller ---
    startRequested = Signal()
    stopRequested = Signal()
    recordToggled = Signal(bool)
    fullscreenRequested = Signal()

    modelsReloadRequested = Signal()
    camerasReloadRequested = Signal()
    modelSelectionChanged = Signal(str)
    cameraSelectionChanged = Signal(object)
    classesPathChanged = Signal(str)
    outputDirChanged = Signal(str)

    tasksSelectionChanged = Signal(list)
    openRecordingRequested = Signal(str)
    transferRequested = Signal(list, str)
    emailRequested = Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("S.T.I Innovation — Real-time weather detection")
        self.setMinimumSize(920, 560)

        self._ui_mode: Optional[str] = None
        self._open_drawer: bool = False
        self._adv_anchor: Optional[QWidget] = None

        self.all_tasks: Dict[str, List[str]] = {}
        self.selected_tasks: List[str] = []

        self._base_qss = ""
        self._last_scale = 1.0

        self._desktop_built = False
        self._drawer_built = False

        self._taskChecks: Dict[str, QCheckBox] = {}

        # build
        self._build_shared_widgets_desktop()
        self._build_shared_widgets_drawer()
        self._wire_sync_desktop_drawer()
        self._build_ui()

        # F11 = vrai plein écran
        QShortcut(QKeySequence("F11"), self, activated=self._toggle_fullscreen)

    # ───────────────────────── shared helpers ─────────────────────────

    def _tag_secondary(self, btn: QPushButton):
        btn.setProperty("kind", "secondary")
        btn.setCursor(Qt.PointingHandCursor)

    def _sync_pair(self, src, dst, getter: Callable, setter: Callable, signal_name: str):
        """
        Generic safe sync:
          - connect src.<signal_name> to set dst
          - connect dst.<signal_name> to set src
        """
        sig_src = getattr(src, signal_name)
        sig_dst = getattr(dst, signal_name)

        def apply_src_to_dst(*_):
            v = getter(src)
            dst.blockSignals(True)
            setter(dst, v)
            dst.blockSignals(False)

        def apply_dst_to_src(*_):
            v = getter(dst)
            src.blockSignals(True)
            setter(src, v)
            src.blockSignals(False)

        sig_src.connect(apply_src_to_dst)
        sig_dst.connect(apply_dst_to_src)

    # ───────────────────────── Widgets (DESKTOP) ─────────────────────────

    def _build_shared_widgets_desktop(self):
        # Header
        self.logo = QLabel()
        self.logo.setObjectName("logo")
        self.logo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.logoPath = str(Path(__file__).parent / "images" / "logo_cerema.png")

        self.appTitle = QLabel("S.T.I-WeatherMeasure")
        self.appTitle.setObjectName("appTitle")

        # Video
        self.videoLabel = QLabel("Prévisualisation vidéo")
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.videoLabel.setStyleSheet("background:#000; color:#aaa; border-radius:14px;")

        # Controls desktop (source of truth)
        self.modelCombo = QComboBox()
        self.modelCombo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.modelCombo.setMinimumContentsLength(18)
        self.modelCombo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.modelCombo.currentIndexChanged.connect(self._emit_model_changed)

        self.reloadBtn = QPushButton("Recharger")
        self._tag_secondary(self.reloadBtn)
        self.reloadBtn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.reloadBtn.clicked.connect(self.modelsReloadRequested)

        self.deviceLabel = QLabel("Device: CPU")
        self.deviceLabel.setWordWrap(True)
        self.deviceLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.taskDialogBtn = QPushButton("Sélectionner tâches…")
        self._tag_secondary(self.taskDialogBtn)
        self.taskDialogBtn.clicked.connect(self._open_task_dialog)

        self.classesEdit = QLineEdit()
        self.classesEdit.setPlaceholderText("classes JSON (facultatif)")
        self.classesEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.classesEdit.textChanged.connect(lambda t: self.classesPathChanged.emit(t) if t.strip() else None)

        self.chooseClasses = QPushButton("Parcourir…")
        self._tag_secondary(self.chooseClasses)
        self.chooseClasses.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.chooseClasses.clicked.connect(self._choose_classes)

        self.cameraCombo = QComboBox()
        self.cameraCombo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cameraCombo.setMinimumContentsLength(18)
        self.cameraCombo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cameraCombo.currentIndexChanged.connect(self._emit_camera_changed)

        self.rescanCamBtn = QPushButton("Rafraîchir")
        self._tag_secondary(self.rescanCamBtn)
        self.rescanCamBtn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.rescanCamBtn.clicked.connect(self.camerasReloadRequested)

        self.fpsSpin = QSpinBox()
        self.fpsSpin.setRange(5, 60)
        self.fpsSpin.setValue(20)
        self.fpsSpin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.threshSpin = QDoubleSpinBox()
        self.threshSpin.setRange(0.0, 1.0)
        self.threshSpin.setSingleStep(0.05)
        self.threshSpin.setValue(0.5)
        self.threshSpin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.inferEverySpin = QSpinBox()
        self.inferEverySpin.setRange(1, 8)
        self.inferEverySpin.setValue(1)
        self.inferEverySpin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.formatCombo = QComboBox()
        self.formatCombo.addItems(["json", "csv", "xlsx", "txt"])
        self.formatCombo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.sessionEdit = QLineEdit()
        self.sessionEdit.setPlaceholderText("Nom de session (défaut: <modèle>_<timestamp>)")
        self.sessionEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.dirEdit = QLineEdit("runs")
        self.dirEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.dirEdit.textChanged.connect(lambda t: self.outputDirChanged.emit(t) if t.strip() else None)

        self.chooseDir = QPushButton("Dossier…")
        self._tag_secondary(self.chooseDir)
        self.chooseDir.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.chooseDir.clicked.connect(self._choose_dir)

        self.speedCheck = QCheckBox("Afficher vitesse (IA/affichage)")
        self.speedCheck.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Mobile bar actions
        self.burgerBtn = QToolButton()
        self.burgerBtn.setText("☰")
        self.burgerBtn.setObjectName("burgerBtn")
        self.burgerBtn.setCursor(Qt.PointingHandCursor)
        self.burgerBtn.clicked.connect(self._toggle_drawer)

        self.gearBtn = QToolButton()
        self.gearBtn.setText("⚙")
        self.gearBtn.setObjectName("gearBtn")
        self.gearBtn.setCursor(Qt.PointingHandCursor)
        self.gearBtn.clicked.connect(lambda: self._toggle_drawer(scroll_to_adv=True))

        # Main actions (colored)
        self.startBtn = QPushButton("▶ Démarrer")
        self.startBtn.setObjectName("startBtn")
        self.stopBtn = QPushButton("■ Stop")
        self.stopBtn.setObjectName("stopBtn")
        self.stopBtn.setEnabled(False)
        self.recBtn = QPushButton("● Enregistrer")
        self.recBtn.setObjectName("recBtn")
        self.recBtn.setCheckable(True)

        # bouton = agrandir/restaurer
        self.fullBtn = QPushButton("⤢ Plein écran")
        self.fullBtn.setObjectName("fullBtn")

        self.startBtn.clicked.connect(self.startRequested)
        self.stopBtn.clicked.connect(self.stopRequested)
        self.recBtn.toggled.connect(self.recordToggled)

        self.fullBtn.clicked.connect(self._toggle_maximize)
        self.fullBtn.clicked.connect(self.fullscreenRequested)

        # Desktop mirror buttons
        self.startBtnDesk = QPushButton("▶ Démarrer"); self.startBtnDesk.setObjectName("startBtnDesk")
        self.stopBtnDesk = QPushButton("■ Stop"); self.stopBtnDesk.setObjectName("stopBtnDesk")
        self.recBtnDesk = QPushButton("● Enregistrer"); self.recBtnDesk.setObjectName("recBtnDesk")
        self.recBtnDesk.setCheckable(True)
        self.fullBtnDesk = QPushButton("⤢ Plein écran"); self.fullBtnDesk.setObjectName("fullBtnDesk")

        self.startBtnDesk.clicked.connect(self.startRequested)
        self.stopBtnDesk.clicked.connect(self.stopRequested)
        self.recBtnDesk.toggled.connect(self.recordToggled)

        self.fullBtnDesk.clicked.connect(self._toggle_maximize)
        self.fullBtnDesk.clicked.connect(self.fullscreenRequested)

        # Playback/share
        self.playBtn = QPushButton("Lire un enregistrement…")
        self._tag_secondary(self.playBtn)
        self.playBtn.clicked.connect(self._open_recording_dialog)

        self.transferBtn = QPushButton("Transférer fichiers…")
        self._tag_secondary(self.transferBtn)
        self.transferBtn.clicked.connect(self._transfer_dialog)

        self.emailBtn = QPushButton("Envoyer par e-mail…")
        self._tag_secondary(self.emailBtn)
        self.emailBtn.clicked.connect(self._email_dialog)

    # ───────────────────────── Widgets (DRAWER) ─────────────────────────

    def _build_shared_widgets_drawer(self):
        # clones: mêmes rôles, widgets séparés
        self.d_modelCombo = QComboBox()
        self.d_modelCombo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.d_modelCombo.setMinimumContentsLength(18)
        self.d_modelCombo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.d_modelCombo.currentIndexChanged.connect(self._emit_model_changed_from_drawer)

        self.d_reloadBtn = QPushButton("Recharger")
        self._tag_secondary(self.d_reloadBtn)
        self.d_reloadBtn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.d_reloadBtn.clicked.connect(self.modelsReloadRequested)

        self.d_deviceLabel = QLabel("Device: CPU")
        self.d_deviceLabel.setWordWrap(True)
        self.d_deviceLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.d_classesEdit = QLineEdit()
        self.d_classesEdit.setPlaceholderText("classes JSON (facultatif)")
        self.d_classesEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.d_classesEdit.textChanged.connect(lambda t: self.classesPathChanged.emit(t) if t.strip() else None)

        self.d_chooseClasses = QPushButton("Parcourir…")
        self._tag_secondary(self.d_chooseClasses)
        self.d_chooseClasses.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.d_chooseClasses.clicked.connect(self._choose_classes_from_drawer)

        self.d_cameraCombo = QComboBox()
        self.d_cameraCombo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.d_cameraCombo.setMinimumContentsLength(18)
        self.d_cameraCombo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.d_cameraCombo.currentIndexChanged.connect(self._emit_camera_changed_from_drawer)

        self.d_rescanCamBtn = QPushButton("Rafraîchir")
        self._tag_secondary(self.d_rescanCamBtn)
        self.d_rescanCamBtn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.d_rescanCamBtn.clicked.connect(self.camerasReloadRequested)

        self.d_fpsSpin = QSpinBox()
        self.d_fpsSpin.setRange(5, 60)
        self.d_fpsSpin.setValue(20)
        self.d_fpsSpin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.d_threshSpin = QDoubleSpinBox()
        self.d_threshSpin.setRange(0.0, 1.0)
        self.d_threshSpin.setSingleStep(0.05)
        self.d_threshSpin.setValue(0.5)
        self.d_threshSpin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.d_inferEverySpin = QSpinBox()
        self.d_inferEverySpin.setRange(1, 8)
        self.d_inferEverySpin.setValue(1)
        self.d_inferEverySpin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.d_formatCombo = QComboBox()
        self.d_formatCombo.addItems(["json", "csv", "xlsx", "txt"])
        self.d_formatCombo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.d_sessionEdit = QLineEdit()
        self.d_sessionEdit.setPlaceholderText("Nom de session (défaut: <modèle>_<timestamp>)")
        self.d_sessionEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.d_dirEdit = QLineEdit("runs")
        self.d_dirEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.d_dirEdit.textChanged.connect(lambda t: self.outputDirChanged.emit(t) if t.strip() else None)

        self.d_chooseDir = QPushButton("Dossier…")
        self._tag_secondary(self.d_chooseDir)
        self.d_chooseDir.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.d_chooseDir.clicked.connect(self._choose_dir_from_drawer)

        self.d_speedCheck = QCheckBox("Afficher vitesse (IA/affichage)")
        self.d_speedCheck.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # share buttons in drawer (clones)
        self.d_playBtn = QPushButton("Lire un enregistrement…")
        self._tag_secondary(self.d_playBtn)
        self.d_playBtn.clicked.connect(self._open_recording_dialog)

        self.d_transferBtn = QPushButton("Transférer fichiers…")
        self._tag_secondary(self.d_transferBtn)
        self.d_transferBtn.clicked.connect(self._transfer_dialog)

        self.d_emailBtn = QPushButton("Envoyer par e-mail…")
        self._tag_secondary(self.d_emailBtn)
        self.d_emailBtn.clicked.connect(self._email_dialog)

    def _wire_sync_desktop_drawer(self):
        # combos: sync index
        self._sync_pair(
            self.modelCombo, self.d_modelCombo,
            getter=lambda w: w.currentIndex(),
            setter=lambda w, v: w.setCurrentIndex(int(v)),
            signal_name="currentIndexChanged"
        )
        self._sync_pair(
            self.cameraCombo, self.d_cameraCombo,
            getter=lambda w: w.currentIndex(),
            setter=lambda w, v: w.setCurrentIndex(int(v)),
            signal_name="currentIndexChanged"
        )

        # edits
        self._sync_pair(
            self.classesEdit, self.d_classesEdit,
            getter=lambda w: w.text(),
            setter=lambda w, v: w.setText(str(v)),
            signal_name="textChanged"
        )
        self._sync_pair(
            self.dirEdit, self.d_dirEdit,
            getter=lambda w: w.text(),
            setter=lambda w, v: w.setText(str(v)),
            signal_name="textChanged"
        )
        self._sync_pair(
            self.sessionEdit, self.d_sessionEdit,
            getter=lambda w: w.text(),
            setter=lambda w, v: w.setText(str(v)),
            signal_name="textChanged"
        )

        # spins
        self._sync_pair(
            self.fpsSpin, self.d_fpsSpin,
            getter=lambda w: w.value(),
            setter=lambda w, v: w.setValue(int(v)),
            signal_name="valueChanged"
        )
        self._sync_pair(
            self.inferEverySpin, self.d_inferEverySpin,
            getter=lambda w: w.value(),
            setter=lambda w, v: w.setValue(int(v)),
            signal_name="valueChanged"
        )
        self._sync_pair(
            self.threshSpin, self.d_threshSpin,
            getter=lambda w: w.value(),
            setter=lambda w, v: w.setValue(float(v)),
            signal_name="valueChanged"
        )

        # format combo
        self._sync_pair(
            self.formatCombo, self.d_formatCombo,
            getter=lambda w: w.currentIndex(),
            setter=lambda w, v: w.setCurrentIndex(int(v)),
            signal_name="currentIndexChanged"
        )

        # checkbox
        self._sync_pair(
            self.speedCheck, self.d_speedCheck,
            getter=lambda w: w.isChecked(),
            setter=lambda w, v: w.setChecked(bool(v)),
            signal_name="toggled"
        )

    # ───────────────────────── Helpers layout ─────────────────────────

    def _make_form(self) -> QFormLayout:
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        return form

    def _hrow(self, *widgets: QWidget, stretch_index: Optional[int] = 0) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        for i, wi in enumerate(widgets):
            if stretch_index is not None and i == stretch_index:
                lay.addWidget(wi, 1)
            else:
                lay.addWidget(wi)
        return w

    # ───────────────────────── Build UI ─────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # Topbar
        top = QFrame()
        top.setObjectName("topbar")
        ht = QHBoxLayout(top)
        ht.setContentsMargins(8, 8, 8, 8)
        ht.setSpacing(10)
        ht.addWidget(self.logo)
        ht.addWidget(self.appTitle)
        ht.addStretch()
        root.addWidget(top)

        # Desktop: splitter video / panel
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(8)

        self.controlsScroll = QScrollArea()
        self.controlsScroll.setWidgetResizable(True)
        self.controlsScroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.controlsScroll.setMinimumWidth(320)

        self.controlsPanelDesktop = QWidget()
        self.controlsScroll.setWidget(self.controlsPanelDesktop)

        self._mount_controls_desktop()  # once

        self.splitter.addWidget(self.videoLabel)
        self.splitter.addWidget(self.controlsScroll)
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 2)

        # Mobile bar
        self.mobilePanel = QWidget()
        self._mount_controls_mobile()

        root.addWidget(self.splitter, 1)
        root.addWidget(self.mobilePanel, 0)

        # Menu minimal
        actHelp = QAction("Aide", self)
        actHelp.triggered.connect(lambda: QMessageBox.information(self, "Aide", "Voir la documentation du projet."))
        self.menuBar().addAction(actHelp)

        # Drawer + scrim
        self._build_drawer()

        # Styles + responsive
        self._rebuild_base_styles()
        self._update_logo_pixmap(1.0)
        self._apply_ui_scale()
        self._apply_responsive_layout(force=True)
        self._sync_full_button_text()

    # ───────────────────────── Desktop controls ─────────────────────────

    def _mount_controls_desktop(self):
        if self._desktop_built:
            return
        self._desktop_built = True

        v = QVBoxLayout(self.controlsPanelDesktop)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(10)

        boxModel = QGroupBox("Modèle")
        l = QVBoxLayout(boxModel)
        l.setSpacing(8)
        l.addWidget(self._hrow(self.modelCombo, self.reloadBtn, stretch_index=0))
        l.addWidget(self.deviceLabel)
        l.addWidget(self.taskDialogBtn)
        v.addWidget(boxModel)

        boxCls = QGroupBox("Classes")
        l2 = QVBoxLayout(boxCls)
        l2.setSpacing(8)
        l2.addWidget(self._hrow(self.classesEdit, self.chooseClasses, stretch_index=0))
        v.addWidget(boxCls)

        boxCam = QGroupBox("Caméra & runtime")
        form = self._make_form()
        form.addRow("Caméra :", self._hrow(self.cameraCombo, self.rescanCamBtn, stretch_index=0))
        form.addRow("FPS affichage :", self.fpsSpin)
        form.addRow("Seuil proba :", self.threshSpin)
        form.addRow("Inférence 1 image sur :", self.inferEverySpin)
        boxCam.setLayout(form)
        v.addWidget(boxCam)
        v.addWidget(self.speedCheck)

        boxExp = QGroupBox("Export résumé")
        form2 = self._make_form()
        form2.addRow("Format :", self.formatCombo)
        form2.addRow("Nom session :", self.sessionEdit)
        form2.addRow("Dossier :", self._hrow(self.dirEdit, self.chooseDir, stretch_index=0))
        boxExp.setLayout(form2)
        v.addWidget(boxExp)

        actionBox = QGroupBox("Actions")
        ha = QHBoxLayout(actionBox)
        ha.setSpacing(8)
        ha.addWidget(self.startBtnDesk)
        ha.addWidget(self.stopBtnDesk)
        ha.addWidget(self.recBtnDesk)
        ha.addWidget(self.fullBtnDesk)
        v.addWidget(actionBox)

        share = QGroupBox("Enregistrements & partage")
        ls = QVBoxLayout(share)
        ls.setSpacing(8)
        ls.addWidget(self.playBtn)
        ls.addWidget(self.transferBtn)
        ls.addWidget(self.emailBtn)
        v.addWidget(share)

        v.addStretch(1)

    # ───────────────────────── Mobile bar ─────────────────────────

    def _mount_controls_mobile(self):
        v = QVBoxLayout(self.mobilePanel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        bar = QFrame()
        bar.setObjectName("mobileBar")
        hb = QHBoxLayout(bar)
        hb.setContentsMargins(8, 8, 8, 8)
        hb.setSpacing(8)

        hb.addWidget(self.burgerBtn)
        hb.addWidget(self.startBtn, 1)
        hb.addWidget(self.stopBtn, 1)
        hb.addWidget(self.recBtn, 1)
        hb.addWidget(self.fullBtn, 1)
        hb.addWidget(self.gearBtn)

        v.addWidget(bar)

    # ───────────────────────── Drawer ─────────────────────────

    def _build_drawer(self):
        central = self.centralWidget()

        self.scrim = ClickableScrim(parent=central)
        self.scrim.setVisible(False)
        self.scrim.setStyleSheet("background: rgba(0,0,0,160);")
        self.scrim.clicked.connect(self._close_drawer)

        self.drawer = SideDrawer(central, side="left", title="Options")
        self.drawer.closed.connect(lambda: self._show_scrim(False))

        self._layout_drawer(initial=True)
        self._build_drawer_content_once()

    def _build_drawer_content_once(self):
        if self._drawer_built:
            return
        self._drawer_built = True

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)

        navW = QWidget()
        nav = QHBoxLayout(navW)
        nav.setContentsMargins(0, 0, 0, 0)
        nav.setSpacing(8)
        btnAdv = QPushButton("Aller à Avancé")
        btnAdv.setProperty("kind", "secondary")
        btnAdv.clicked.connect(self._scroll_to_advanced)
        nav.addWidget(QLabel("Options"))
        nav.addStretch(1)
        nav.addWidget(btnAdv)
        outer.addWidget(navW)

        secModel = DrawerSection("Modèle & tâches")
        secModel.layout().addWidget(self._hrow(self.d_modelCombo, self.d_reloadBtn, stretch_index=0))
        secModel.layout().addWidget(self.d_deviceLabel)

        self._drawer_tasks_container = QWidget()
        self._drawer_tasks_layout = QVBoxLayout(self._drawer_tasks_container)
        self._drawer_tasks_layout.setContentsMargins(0, 0, 0, 0)
        self._drawer_tasks_layout.setSpacing(8)
        secModel.layout().addWidget(self._drawer_tasks_container)
        outer.addWidget(secModel)

        secCam = DrawerSection("Caméra & runtime")
        formc = self._make_form()
        formc.addRow("Caméra :", self._hrow(self.d_cameraCombo, self.d_rescanCamBtn, stretch_index=0))
        formc.addRow("FPS :", self.d_fpsSpin)
        formc.addRow("Seuil :", self.d_threshSpin)
        formc.addRow("Inférence 1/N :", self.d_inferEverySpin)
        secCam.layout().addLayout(formc)
        secCam.layout().addWidget(self.d_speedCheck)
        outer.addWidget(secCam)

        self._adv_anchor = QLabel("")
        self._adv_anchor.setFixedHeight(1)
        outer.addWidget(self._adv_anchor)

        secCls = DrawerSection("Classes")
        secCls.layout().addWidget(self._hrow(self.d_classesEdit, self.d_chooseClasses, stretch_index=0))
        outer.addWidget(secCls)

        secExp = DrawerSection("Export / Session")
        form2 = self._make_form()
        form2.addRow("Format :", self.d_formatCombo)
        form2.addRow("Nom session :", self.d_sessionEdit)
        form2.addRow("Dossier :", self._hrow(self.d_dirEdit, self.d_chooseDir, stretch_index=0))
        secExp.layout().addLayout(form2)
        outer.addWidget(secExp)

        secShare = DrawerSection("Enregistrements & partage")
        secShare.layout().addWidget(self.d_playBtn)
        secShare.layout().addWidget(self.d_transferBtn)
        secShare.layout().addWidget(self.d_emailBtn)
        outer.addWidget(secShare)

        self.drawer.setContent(root)
        self._refresh_drawer_tasks()

    def _refresh_drawer_tasks(self):
        if not hasattr(self, "_drawer_tasks_layout"):
            return

        while self._drawer_tasks_layout.count():
            it = self._drawer_tasks_layout.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()

        self._taskChecks = {}

        if not self.all_tasks:
            lbl = QLabel("Aucune tâche (sélectionnez un modèle).")
            lbl.setWordWrap(True)
            self._drawer_tasks_layout.addWidget(lbl)
            return

        info = QLabel("Tâches (affichage + prune au démarrage) :")
        info.setWordWrap(True)
        info.setStyleSheet("color:#cbd5e1;")
        self._drawer_tasks_layout.addWidget(info)

        for t in self.all_tasks.keys():
            cb = QCheckBox(t)
            cb.setChecked(t in self.selected_tasks)
            self._taskChecks[t] = cb
            self._drawer_tasks_layout.addWidget(cb)

        rowW = QWidget()
        row = QHBoxLayout(rowW)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        btnAll = QPushButton("Tout");   btnAll.setProperty("kind", "secondary")
        btnNone = QPushButton("Aucun"); btnNone.setProperty("kind", "secondary")
        btnApply = QPushButton("Appliquer"); btnApply.setProperty("kind", "primary")

        btnAll.clicked.connect(lambda: [c.setChecked(True) for c in self._taskChecks.values()])
        btnNone.clicked.connect(lambda: [c.setChecked(False) for c in self._taskChecks.values()])
        btnApply.clicked.connect(self._apply_task_selection_from_drawer)

        row.addWidget(btnAll)
        row.addWidget(btnNone)
        row.addStretch(1)
        row.addWidget(btnApply)

        self._drawer_tasks_layout.addWidget(rowW)

    def _layout_drawer(self, initial=False):
        if getattr(self, "drawer", None):
            self.drawer.layout_now(initial=initial)
        if getattr(self, "scrim", None) and self.centralWidget():
            self.scrim.setGeometry(self.centralWidget().rect())

    def _scroll_to_advanced(self):
        if self.drawer and self._adv_anchor:
            self.drawer.scroll.ensureWidgetVisible(self._adv_anchor, 0, 24)

    def _toggle_drawer(self, scroll_to_adv: bool = False):
        if not self._is_mobile():
            return
        if self._open_drawer:
            if scroll_to_adv:
                self._scroll_to_advanced()
            else:
                self._close_drawer()
            return

        self._show_scrim(True)
        self.drawer.open_drawer()
        self._open_drawer = True
        if scroll_to_adv:
            QTimer.singleShot(240, self._scroll_to_advanced)

    def _close_drawer(self):
        if getattr(self, "drawer", None) and self._open_drawer:
            self.drawer.close_drawer()
        self._open_drawer = False
        self._show_scrim(False)

    def _show_scrim(self, show: bool):
        if not getattr(self, "scrim", None):
            return
        self.scrim.setVisible(show)
        if show:
            self.scrim.raise_()

    # ───────────────────────── Responsive ─────────────────────────

    def showEvent(self, e):
        super().showEvent(e)
        QTimer.singleShot(0, lambda: self._apply_responsive_layout(force=True))
        QTimer.singleShot(0, self._sync_full_button_text)

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() == QEvent.WindowStateChange:
            QTimer.singleShot(0, self._apply_responsive_layout)
            QTimer.singleShot(0, self._sync_full_button_text)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._apply_responsive_layout()
        self._apply_ui_scale()
        self._layout_drawer()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape and self._open_drawer:
            self._close_drawer()
            e.accept()
            return
        super().keyPressEvent(e)

    def _is_mobile(self) -> bool:
        # force desktop en full screen (F11) et maximisé (bouton)
        if self.isFullScreen() or self.isMaximized():
            return False
        w, h = self.width(), self.height()
        return (w < 980) or (h < 620)

    def _apply_responsive_layout(self, force=False):
        mobile = self._is_mobile()
        mode = self._ui_mode

        if mobile and mode != "mobile":
            self.logo.setVisible(False)
            self.appTitle.setVisible(False)
            self.controlsScroll.setVisible(False)
            self.splitter.setVisible(True)
            self.mobilePanel.setVisible(True)

            self.splitter.widget(0).setVisible(True)
            self.splitter.widget(1).setVisible(False)

            self._ui_mode = "mobile"

        elif (not mobile) and mode != "desktop":
            self.logo.setVisible(True)
            self.appTitle.setVisible(True)
            self.controlsScroll.setVisible(True)
            self.mobilePanel.setVisible(False)

            self.splitter.widget(0).setVisible(True)
            self.splitter.widget(1).setVisible(True)

            QTimer.singleShot(0, lambda: self._ensure_splitter_sizes(force=True))
            self._close_drawer()
            self._ui_mode = "desktop"

        if (not mobile) and self._open_drawer:
            self._close_drawer()

        if (not mobile):
            QTimer.singleShot(0, self._ensure_splitter_sizes)

    def _ensure_splitter_sizes(self, force: bool = False):
        if not self.splitter.isVisible():
            return
        if self.splitter.count() < 2:
            return
        w = self.splitter.width()
        if w <= 0:
            return

        min_panel = 320
        sizes = self.splitter.sizes()

        if force:
            self.splitter.setSizes([int(w * 0.66), int(w * 0.34)])
            return

        if len(sizes) >= 2 and sizes[1] < min_panel:
            self.splitter.setSizes([int(w * 0.66), int(w * 0.34)])

    # ───────────────────────── Styles / scale ─────────────────────────

    def _rebuild_base_styles(self):
        self._base_qss = """
        /* ─────────────────────────
           Palette (dark + cool accent)
           bg0  : #0b1020 (main)
           bg1  : #0f172a (surface)
           bg2  : #111c33 (surface alt)
           line : #233047 / #2b3a55
           text : #e6edf7
           mut  : #a9b4c7
           acc  : #60a5fa (blue)
           pri  : #7c3aed (violet)
           ok   : #22c55e
           warn : #f59e0b
           bad  : #ef4444
        ───────────────────────── */

        QMainWindow {
            background:#0b1020;
            color:#e6edf7;
        }

        /* Top bar */
        QFrame#topbar {
            background:#0f172a;
            border:1px solid #233047;
            border-radius:14px;
        }
        QLabel#appTitle {
            font-size:18px;
            font-weight:900;
            color:#e6edf7;
        }

        /* Group boxes */
        QGroupBox {
            border:1px solid #233047;
            border-radius:12px;
            margin-top:12px;
            padding-top:10px;
            background: #0f172a;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left:10px;
            padding: 0 8px;
            color:#60a5fa;
            font-weight:800;
        }

        QScrollArea { border:none; }
        QLabel { color:#e6edf7; }
        QLabel[muted="true"] { color:#a9b4c7; }

        /* Inputs */
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
            background:#111c33;
            border:1px solid #2b3a55;
            border-radius:10px;
            color:#e6edf7;
            selection-background-color: rgba(96,165,250,0.35);
        }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {
            border:1px solid #60a5fa;
        }
        QLineEdit::placeholder, QTextEdit::placeholder {
            color: rgba(169,180,199,0.55);
        }

        /* Checkboxes */
        QCheckBox { color:#e6edf7; }
        QCheckBox::indicator { width:16px; height:16px; }
        QCheckBox::indicator:unchecked {
            border:1px solid #2b3a55;
            border-radius:4px;
            background:#0f172a;
        }
        QCheckBox::indicator:checked {
            border:1px solid #60a5fa;
            border-radius:4px;
            background: rgba(96,165,250,0.35);
        }

        /* Buttons (base) */
        QPushButton {
            background:#111c33;
            border:1px solid #2b3a55;
            border-radius:12px;
            color:#e6edf7;
            font-weight:700;
        }
        QPushButton:hover {
            background:#152243;
            border-color:#3a4c70;
        }
        QPushButton:disabled {
            color: rgba(230,237,247,0.45);
            background:#0f172a;
            border-color:#1f2a3f;
        }

        /* Secondary = accent blue */
        QPushButton[kind="secondary"] {
            background: rgba(96,165,250,0.18);
            border:1px solid rgba(96,165,250,0.55);
            color:#e6edf7;
            font-weight:800;
        }
        QPushButton[kind="secondary"]:hover {
            background: rgba(96,165,250,0.28);
            border-color: rgba(96,165,250,0.75);
        }

        /* Primary = violet */
        QPushButton[kind="primary"] {
            background: rgba(124,58,237,0.22);
            border:1px solid rgba(124,58,237,0.65);
            color:#e6edf7;
            font-weight:900;
        }
        QPushButton[kind="primary"]:hover {
            background: rgba(124,58,237,0.32);
            border-color: rgba(124,58,237,0.85);
        }

        /* Drawer */
        QFrame#drawerLeft {
            background: #0f172a;
            color: #e6edf7;
            border-top-left-radius:16px;
            border-bottom-left-radius:16px;
            border:1px solid #233047;
        }
        #drawerHeader {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #60a5fa, stop:1 #7c3aed);
            border-top-left-radius:16px;
            border-top-right-radius:16px;
        }
        #drawerTitle { font-size:18px; font-weight:900; color:#ffffff; }
        #drawerSubtitle { font-size:12px; color: rgba(255,255,255,0.85); }
        #drawerFooter {
            background:#0b1327;
            border-bottom-left-radius:16px;
            border-bottom-right-radius:16px;
            border-top:1px solid #233047;
        }

        /* Mobile bar + tool buttons */
        QFrame#mobileBar {
            background:#0f172a;
            border:1px solid #233047;
            border-radius:14px;
        }
        QToolButton#burgerBtn, QToolButton#gearBtn {
            background:#111c33;
            border:1px solid #2b3a55;
            border-radius:14px;
            color:#e6edf7;
            font-weight:900;
        }
        QToolButton#burgerBtn:hover, QToolButton#gearBtn:hover {
            background:#152243;
            border-color:#3a4c70;
        }
        """

    def _apply_ui_scale(self):
        w, h = max(640, self.width()), max(480, self.height())
        ui_scale = max(0.85, min(1.25, min(w / 1280.0, h / 720.0)))
        self._last_scale = ui_scale

        f = QFont(self.font())
        f.setPointSizeF(11.0 * ui_scale)
        self.setFont(f)

        pad = max(6, int(8 * ui_scale))
        tool = max(42, int(44 * ui_scale))
        icon_fs = int(18 * ui_scale)

        scaled_qss = self._base_qss + f"""
        QPushButton {{ padding:{pad}px {pad+4}px; }}

        QToolButton#burgerBtn, QToolButton#gearBtn {{
            min-width:{tool}px; min-height:{tool}px; font-size:{icon_fs}px;
        }}

        QPushButton#startBtn, QPushButton#startBtnDesk {{
            background:#16a34a; border:1px solid #22c55e; color:white; font-weight:900; border-radius:16px;
        }}
        QPushButton#stopBtn, QPushButton#stopBtnDesk {{
            background:#dc2626; border:1px solid #ef4444; color:white; font-weight:900; border-radius:16px;
        }}
        QPushButton#recBtn, QPushButton#recBtnDesk {{
            background:#f43f5e; border:1px solid #fb7185; color:white; font-weight:900; border-radius:16px;
        }}
        QPushButton#fullBtn, QPushButton#fullBtnDesk {{
            background:#0ea5e9; border:1px solid #38bdf8; color:white; font-weight:900; border-radius:16px;
        }}
        """

        self.setStyleSheet(scaled_qss)
        self._update_logo_pixmap(ui_scale)

    # ───────────────────────── Fullscreen / Maximize ─────────────────────────

    def _toggle_maximize(self):
        if self.isFullScreen():
            self.showNormal()
        elif self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

        self._sync_full_button_text()

        QTimer.singleShot(0, lambda: self._apply_responsive_layout(force=True))
        QTimer.singleShot(50, lambda: self._ensure_splitter_sizes(force=True))
        QTimer.singleShot(250, lambda: self._ensure_splitter_sizes(force=True))

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

        self._sync_full_button_text()

        QTimer.singleShot(0, lambda: self._apply_responsive_layout(force=True))
        QTimer.singleShot(50, lambda: self._ensure_splitter_sizes(force=True))
        QTimer.singleShot(250, lambda: self._ensure_splitter_sizes(force=True))

    def _sync_full_button_text(self):
        if self.isFullScreen() or self.isMaximized():
            txt = "⤡ Restaurer"
        else:
            txt = "⤢ Plein écran"

        self.fullBtn.setText(txt)
        self.fullBtnDesk.setText(txt)

    def _update_logo_pixmap(self, scale: float):
        base_h = 34
        target_h = max(22, int(base_h * max(0.75, scale)))
        p = QPixmap(self.logoPath)
        if not p.isNull():
            p = p.scaledToHeight(target_h, Qt.SmoothTransformation)
            self.logo.setPixmap(p)
            self.logo.setMinimumSize(p.size())
        else:
            self.logo.setText("LOGO")
            self.logo.setStyleSheet("font-weight:600; padding:6px;")

    # ───────────────────────── UI events → signals ─────────────────────────

    def _emit_model_changed(self):
        model_path = self.modelCombo.currentData() or self.modelCombo.currentText()
        self.modelSelectionChanged.emit(str(model_path))

    def _emit_model_changed_from_drawer(self):
        # déclenche via desktop (source de vérité)
        self._emit_model_changed()

    def _emit_camera_changed(self):
        data = self.cameraCombo.currentData()
        self.cameraSelectionChanged.emit(data)

    def _emit_camera_changed_from_drawer(self):
        self._emit_camera_changed()

    def _choose_classes(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choisir classes.json", "", "JSON (*.json)")
        if path:
            self.classesEdit.setText(path)
            self.classesPathChanged.emit(path)

    def _choose_classes_from_drawer(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choisir classes.json", "", "JSON (*.json)")
        if path:
            self.d_classesEdit.setText(path)
            self.classesPathChanged.emit(path)

    def _choose_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Choisir dossier de sortie", self.dirEdit.text().strip() or "runs")
        if path:
            self.dirEdit.setText(path)
            self.outputDirChanged.emit(path)

    def _choose_dir_from_drawer(self):
        path = QFileDialog.getExistingDirectory(self, "Choisir dossier de sortie", self.d_dirEdit.text().strip() or "runs")
        if path:
            self.d_dirEdit.setText(path)
            self.outputDirChanged.emit(path)

    def _open_task_dialog(self):
        if not self.all_tasks:
            QMessageBox.information(self, "Tâches", "Aucune tâche disponible. Sélectionnez d’abord un modèle.")
            return
        dlg = TaskSelectionDialog(self, self.all_tasks, self.selected_tasks or list(self.all_tasks.keys()))
        if dlg.exec() == QDialog.Accepted:
            sel = dlg.selected_tasks()
            if not sel:
                sel = list(self.all_tasks.keys())
            self.tasksSelectionChanged.emit(sel)

    def _apply_task_selection_from_drawer(self):
        if not self._taskChecks:
            return
        sel = [t for t, cb in self._taskChecks.items() if cb.isChecked()]
        if not sel:
            first_key = next(iter(self._taskChecks.keys()), None)
            if first_key:
                self._taskChecks[first_key].setChecked(True)
                sel = [first_key]
        self.tasksSelectionChanged.emit(sel)
        QMessageBox.information(self, "Tâches", "Sélection appliquée.\nLa PRUNE effective aura lieu au démarrage.")

    def _open_recording_dialog(self):
        vid, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir un enregistrement vidéo",
            self.dirEdit.text().strip() or "runs",
            "Vidéos (*.avi *.mp4 *.mkv *.mov)"
        )
        if vid:
            self.openRecordingRequested.emit(vid)

    def _transfer_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Choisir des fichiers à transférer",
            self.dirEdit.text().strip() or "runs",
            "Tous (*.*)"
        )
        if not files:
            return
        dest = QFileDialog.getExistingDirectory(self, "Choisir le dossier de destination")
        if dest:
            self.transferRequested.emit(files, dest)

    def _email_dialog(self):
        dlg = EmailDialog(self, default_attachments=[])
        if dlg.exec() == QDialog.Accepted:
            self.emailRequested.emit(dlg.build_request())

    # ───────────────────────── Public API (Controller → UI) ─────────────────────────

    def set_models(self, entries: List[Tuple[str, str]]):
        # desktop
        self.modelCombo.blockSignals(True)
        self.modelCombo.clear()
        # drawer
        self.d_modelCombo.blockSignals(True)
        self.d_modelCombo.clear()

        if not entries:
            self.modelCombo.addItem("(aucun modèle trouvé…)", "")
            self.d_modelCombo.addItem("(aucun modèle trouvé…)", "")
            self.modelCombo.setEnabled(False)
            self.d_modelCombo.setEnabled(False)
        else:
            self.modelCombo.setEnabled(True)
            self.d_modelCombo.setEnabled(True)
            for label, path in entries:
                self.modelCombo.addItem(label, path)
                self.d_modelCombo.addItem(label, path)

        self.modelCombo.blockSignals(False)
        self.d_modelCombo.blockSignals(False)
        self._emit_model_changed()

    def set_cameras(self, entries: List[Tuple[str, object]]):
        self.cameraCombo.blockSignals(True)
        self.cameraCombo.clear()
        self.d_cameraCombo.blockSignals(True)
        self.d_cameraCombo.clear()

        for label, data in entries:
            self.cameraCombo.addItem(label, data)
            self.d_cameraCombo.addItem(label, data)

        self.cameraCombo.blockSignals(False)
        self.d_cameraCombo.blockSignals(False)
        self._emit_camera_changed()

    def set_tasks(self, all_tasks: Dict[str, List[str]], selected_tasks: List[str]):
        self.all_tasks = dict(all_tasks or {})
        self.selected_tasks = list(selected_tasks or [])
        self._refresh_drawer_tasks()

    def set_device_text(self, txt: str):
        self.deviceLabel.setText(txt)
        self.d_deviceLabel.setText(txt)

    def set_running_state(self, running: bool):
        self.startBtn.setEnabled(not running)
        self.stopBtn.setEnabled(running)
        self.startBtnDesk.setEnabled(not running)
        self.stopBtnDesk.setEnabled(running)

    def set_recording_state(self, recording: bool):
        self.recBtn.blockSignals(True)
        self.recBtnDesk.blockSignals(True)
        self.recBtn.setChecked(recording)
        self.recBtnDesk.setChecked(recording)
        self.recBtn.blockSignals(False)
        self.recBtnDesk.blockSignals(False)

    def show_frame_pixmap(self, pix: QPixmap):
        if pix is None:
            return
        self.videoLabel.setPixmap(pix)

    def show_info(self, title: str, msg: str):
        QMessageBox.information(self, title, msg)

    def show_error(self, title: str, msg: str):
        QMessageBox.critical(self, title, msg)
