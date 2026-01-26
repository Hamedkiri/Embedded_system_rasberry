# ui_app.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI ONLY — PySide6 (Responsive / no-overflow)
- Interface desktop + mobile + drawer
- Responsive: adapte la mise en page au redimensionnement (y compris après plein-écran)
- Drawer: contenu toujours contenu dans la largeur (pas de débordements horizontaux)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import (
    Qt, QTimer, Signal, QSize, QRect, QPropertyAnimation, QEasingCurve, QEvent
)
from PySide6.QtGui import QPixmap, QFont, QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QComboBox, QLineEdit,
    QFileDialog, QCheckBox, QSpinBox, QDoubleSpinBox, QHBoxLayout, QVBoxLayout,
    QGridLayout, QGroupBox, QMessageBox, QDialog, QFormLayout, QDialogButtonBox,
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
    - ScrollArea widgetResizable
    - Anti overflow horizontal: clamp maxWidth au viewport + wrap forms
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
        self.scroll.viewport().installEventFilter(self)  # pour clamp largeur

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
        self.primaryBtn.clicked.connect(self.close_drawer)
        f.addStretch()
        f.addWidget(self.primaryBtn)
        root.addWidget(footer)

        self.anim = QPropertyAnimation(self, b"geometry", self)
        self.anim.setDuration(220)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        self._geom_show = QRect()
        self._geom_hide = QRect()

    def eventFilter(self, obj, event):
        # Clamp largeur du contenu au viewport pour éviter tout débordement horizontal
        if obj is self.scroll.viewport() and event.type() in (QEvent.Resize, QEvent.Show):
            self._clamp_inner_width()
        return super().eventFilter(obj, event)

    def _clamp_inner_width(self):
        vpw = self.scroll.viewport().width()
        if vpw <= 0:
            return
        # laisser respirer un peu (marges internes + scrollbar)
        maxw = max(200, vpw - 2)
        self.inner.setMaximumWidth(maxw)
        # clamp aussi les enfants "racines"
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
        self.anim.finished.connect(self._after_close)

    def _after_close(self):
        try:
            self.anim.finished.disconnect(self._after_close)
        except Exception:
            pass
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
        btnAll.clicked.connect(lambda: [c.setChecked(True) for c in self.checks.values()])
        btnNone.clicked.connect(lambda: [c.setChecked(False) for c in self.checks.values()])
        row.addWidget(btnAll)
        row.addWidget(btnNone)
        row.addStretch(1)
        v.addLayout(row)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

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

        # State
        self._ui_mode: Optional[str] = None
        self._open_drawer: bool = False
        self._adv_anchor: Optional[QWidget] = None

        self.all_tasks: Dict[str, List[str]] = {}
        self.selected_tasks: List[str] = []

        self._build_shared_widgets()
        self._build_ui()

        QShortcut(QKeySequence("F11"), self, activated=self._toggle_fullscreen)

    # ───────────────────────── Shared widgets ─────────────────────────

    def _build_shared_widgets(self):
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

        # Controls (source of truth)
        self.modelCombo = QComboBox()
        self.modelCombo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.modelCombo.setMinimumContentsLength(18)
        self.modelCombo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.modelCombo.currentIndexChanged.connect(self._emit_model_changed)

        self.reloadBtn = QPushButton("Recharger")
        self.reloadBtn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.reloadBtn.clicked.connect(self.modelsReloadRequested)

        self.deviceLabel = QLabel("Device: CPU")
        self.deviceLabel.setWordWrap(True)
        self.deviceLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.taskDialogBtn = QPushButton("Sélectionner tâches…")
        self.taskDialogBtn.clicked.connect(self._open_task_dialog)

        self.classesEdit = QLineEdit()
        self.classesEdit.setPlaceholderText("classes JSON (facultatif)")
        self.classesEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.chooseClasses = QPushButton("Parcourir…")
        self.chooseClasses.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.chooseClasses.clicked.connect(self._choose_classes)

        self.cameraCombo = QComboBox()
        self.cameraCombo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cameraCombo.setMinimumContentsLength(18)
        self.cameraCombo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cameraCombo.currentIndexChanged.connect(self._emit_camera_changed)

        self.rescanCamBtn = QPushButton("Rafraîchir")
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
        self.chooseDir = QPushButton("Dossier…")
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

        self.startBtn = QPushButton("▶ Démarrer")
        self.startBtn.setObjectName("startBtn")
        self.stopBtn = QPushButton("■ Stop")
        self.stopBtn.setObjectName("stopBtn")
        self.stopBtn.setEnabled(False)
        self.recBtn = QPushButton("● Enregistrer")
        self.recBtn.setObjectName("recBtn")
        self.recBtn.setCheckable(True)
        self.fullBtn = QPushButton("⤢ Plein écran")
        self.fullBtn.setObjectName("fullBtn")

        self.startBtn.clicked.connect(self.startRequested)
        self.stopBtn.clicked.connect(self.stopRequested)
        self.recBtn.toggled.connect(self.recordToggled)
        self.fullBtn.clicked.connect(self.fullscreenRequested)

        # Desktop mirror buttons
        self.startBtnDesk = QPushButton("▶ Démarrer")
        self.stopBtnDesk = QPushButton("■ Stop")
        self.recBtnDesk = QPushButton("● Enregistrer")
        self.recBtnDesk.setCheckable(True)
        self.fullBtnDesk = QPushButton("⤢ Plein écran")

        self.startBtnDesk.clicked.connect(self.startRequested)
        self.stopBtnDesk.clicked.connect(self.stopRequested)
        self.recBtnDesk.toggled.connect(self.recordToggled)
        self.fullBtnDesk.clicked.connect(self.fullscreenRequested)

        # Playback/share
        self.playBtn = QPushButton("Lire un enregistrement…")
        self.playBtn.clicked.connect(self._open_recording_dialog)

        self.transferBtn = QPushButton("Transférer fichiers…")
        self.transferBtn.clicked.connect(self._transfer_dialog)

        self.emailBtn = QPushButton("Envoyer par e-mail…")
        self.emailBtn.clicked.connect(self._email_dialog)

    # ───────────────────────── Helpers layout ─────────────────────────

    def _make_form(self) -> QFormLayout:
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)  # évite débordements
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

        # Desktop: splitter vidéo / panneau (meilleur responsive)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(8)

        self.controlsScroll = QScrollArea()
        self.controlsScroll.setWidgetResizable(True)
        self.controlsScroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.controlsPanelDesktop = QWidget()
        self.controlsScroll.setWidget(self.controlsPanelDesktop)
        self._mount_controls_desktop()

        self.splitter.addWidget(self.videoLabel)
        self.splitter.addWidget(self.controlsScroll)
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 2)

        # Mobile panel (bar only; options via drawer)
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

        # Style + responsive
        self._update_logo_pixmap(1.0)
        self._apply_ui_scale()
        self._apply_responsive_layout(force=True)

    def _clear_layout(self, layout: QLayout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

    # ───────────────────────── Desktop controls ─────────────────────────

    def _mount_controls_desktop(self):
        v = self.controlsPanelDesktop.layout() or QVBoxLayout(self.controlsPanelDesktop)
        self._clear_layout(v)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(10)

        # Modèle
        boxModel = QGroupBox("Modèle")
        l = QVBoxLayout(boxModel)
        l.setSpacing(8)
        l.addWidget(self._hrow(self.modelCombo, self.reloadBtn, stretch_index=0))
        l.addWidget(self.deviceLabel)
        l.addWidget(self.taskDialogBtn)
        v.addWidget(boxModel)

        # Classes
        boxCls = QGroupBox("Classes")
        l2 = QVBoxLayout(boxCls)
        l2.setSpacing(8)
        l2.addWidget(self._hrow(self.classesEdit, self.chooseClasses, stretch_index=0))
        v.addWidget(boxCls)

        # Camera/runtime
        boxCam = QGroupBox("Caméra & runtime")
        form = self._make_form()
        form.addRow("Caméra :", self._hrow(self.cameraCombo, self.rescanCamBtn, stretch_index=0))
        form.addRow("FPS affichage :", self.fpsSpin)
        form.addRow("Seuil proba :", self.threshSpin)
        form.addRow("Inférence 1 image sur :", self.inferEverySpin)
        boxCam.setLayout(form)
        v.addWidget(boxCam)
        v.addWidget(self.speedCheck)

        # Export
        boxExp = QGroupBox("Export résumé")
        form2 = self._make_form()
        form2.addRow("Format :", self.formatCombo)
        form2.addRow("Nom session :", self.sessionEdit)
        form2.addRow("Dossier :", self._hrow(self.dirEdit, self.chooseDir, stretch_index=0))
        boxExp.setLayout(form2)
        v.addWidget(boxExp)

        # Actions
        actionBox = QGroupBox("Actions")
        ha = QHBoxLayout(actionBox)
        ha.setSpacing(8)
        ha.addWidget(self.startBtnDesk)
        ha.addWidget(self.stopBtnDesk)
        ha.addWidget(self.recBtnDesk)
        ha.addWidget(self.fullBtnDesk)
        v.addWidget(actionBox)

        # Share
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

    # ───────────────────────── Drawer content ─────────────────────────

    def _build_drawer(self):
        central = self.centralWidget()

        self.scrim = ClickableScrim(parent=central)
        self.scrim.setVisible(False)
        self.scrim.setStyleSheet("background: rgba(0,0,0,160);")
        self.scrim.clicked.connect(self._close_drawer)

        self.drawer = SideDrawer(central, side="left", title="Options")
        self.drawer.closed.connect(lambda: self._show_scrim(False))

        self._style_app()
        self._layout_drawer(initial=True)
        self._populate_drawer_content()

    def _style_app(self):
        # Styles globaux (simples, Pi-friendly)
        self.setStyleSheet("""
        QMainWindow { background:#0b1020; color:#e5e7eb; }
        QFrame#topbar { background:#0f172a; border-radius:14px; }
        QLabel#appTitle { font-size:18px; font-weight:800; }
        QGroupBox {
            border:1px solid #1f2937; border-radius:12px; margin-top:12px; padding-top:10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left:8px; padding: 0 6px; color:#93c5fd; font-weight:700;
        }
        QScrollArea { border:none; }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
            background:#0f172a; border:1px solid #1f2937; border-radius:10px;
        }
        QPushButton {
            background:#111827; border:1px solid #1f2937; border-radius:12px;
        }
        QPushButton:hover { border-color:#334155; }
        QFrame#drawerLeft {
            background: #0F172A; color: #e5e7eb;
            border-top-left-radius:16px; border-bottom-left-radius:16px;
        }
        #drawerHeader {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #0EA5E9, stop:1 #6366F1);
            border-top-left-radius:16px; border-top-right-radius:16px;
        }
        #drawerTitle { font-size:18px; font-weight:800; color:#fff; }
        #drawerSubtitle { font-size:12px; color: rgba(255,255,255,0.85); }
        #drawerFooter { background:#0b1327; border-bottom-left-radius:16px; border-bottom-right-radius:16px; }
        QFrame#mobileBar { background:#0f172a; border-radius:14px; }
        QToolButton#burgerBtn, QToolButton#gearBtn {
            background:#111827; border:1px solid #1f2937; border-radius:14px;
            font-weight:900;
        }
        """)

    def _layout_drawer(self, initial=False):
        if self.drawer:
            self.drawer.layout_now(initial=initial)
        if self.scrim and self.centralWidget():
            self.scrim.setGeometry(self.centralWidget().rect())

    def _make_tasks_widget(self) -> QWidget:
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self._taskChecks: Dict[str, QCheckBox] = {}

        if not self.all_tasks:
            lbl = QLabel("Aucune tâche (sélectionnez un modèle).")
            lbl.setWordWrap(True)
            lay.addWidget(lbl)
            return wrap

        info = QLabel("Tâches (affichage + prune au démarrage) :")
        info.setWordWrap(True)
        info.setStyleSheet("color:#cbd5e1;")
        lay.addWidget(info)

        for t in self.all_tasks.keys():
            cb = QCheckBox(t)
            cb.setChecked(t in self.selected_tasks)
            cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._taskChecks[t] = cb
            lay.addWidget(cb)

        row = QHBoxLayout()
        btnAll = QPushButton("Tout")
        btnNone = QPushButton("Aucun")
        btnApply = QPushButton("Appliquer")
        btnAll.clicked.connect(lambda: [c.setChecked(True) for c in self._taskChecks.values()])
        btnNone.clicked.connect(lambda: [c.setChecked(False) for c in self._taskChecks.values()])
        btnApply.clicked.connect(self._apply_task_selection_from_drawer)

        row.addWidget(btnAll)
        row.addWidget(btnNone)
        row.addStretch(1)
        row.addWidget(btnApply)
        lay.addLayout(row)
        return wrap

    def _populate_drawer_content(self):
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)

        # mini nav
        navW = QWidget()
        nav = QHBoxLayout(navW)
        nav.setContentsMargins(0, 0, 0, 0)
        nav.setSpacing(8)
        btnAdv = QPushButton("Aller à Avancé")
        btnAdv.clicked.connect(self._scroll_to_advanced)
        nav.addWidget(QLabel("Options"))
        nav.addStretch(1)
        nav.addWidget(btnAdv)
        outer.addWidget(navW)

        # Modèle & tâches
        secModel = DrawerSection("Modèle & tâches")
        secModel.layout().addWidget(self._hrow(self.modelCombo, self.reloadBtn, stretch_index=0))
        secModel.layout().addWidget(self.deviceLabel)
        secModel.layout().addWidget(self._make_tasks_widget())
        outer.addWidget(secModel)

        # Caméra
        secCam = DrawerSection("Caméra & runtime")
        formc = self._make_form()
        formc.addRow("Caméra :", self._hrow(self.cameraCombo, self.rescanCamBtn, stretch_index=0))
        formc.addRow("FPS :", self.fpsSpin)
        formc.addRow("Seuil :", self.threshSpin)
        formc.addRow("Inférence 1/N :", self.inferEverySpin)
        secCam.layout().addLayout(formc)
        secCam.layout().addWidget(self.speedCheck)
        outer.addWidget(secCam)

        # Anchor
        self._adv_anchor = QLabel("")
        self._adv_anchor.setFixedHeight(1)
        outer.addWidget(self._adv_anchor)

        # Classes
        secCls = DrawerSection("Classes")
        secCls.layout().addWidget(self._hrow(self.classesEdit, self.chooseClasses, stretch_index=0))
        outer.addWidget(secCls)

        # Export
        secExp = DrawerSection("Export / Session")
        form2 = self._make_form()
        form2.addRow("Format :", self.formatCombo)
        form2.addRow("Nom session :", self.sessionEdit)
        form2.addRow("Dossier :", self._hrow(self.dirEdit, self.chooseDir, stretch_index=0))
        secExp.layout().addLayout(form2)
        outer.addWidget(secExp)

        # Share
        secShare = DrawerSection("Enregistrements & partage")
        secShare.layout().addWidget(self.playBtn)
        secShare.layout().addWidget(self.transferBtn)
        secShare.layout().addWidget(self.emailBtn)
        outer.addWidget(secShare)

        self.drawer.setContent(outer)

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
        if self.drawer and self._open_drawer:
            self.drawer.close_drawer()
        self._open_drawer = False
        self._show_scrim(False)

    def _show_scrim(self, show: bool):
        if not self.scrim:
            return
        self.scrim.setVisible(show)
        if show:
            self.scrim.raise_()

    # ───────────────────────── Responsive ─────────────────────────

    def showEvent(self, e):
        super().showEvent(e)
        QTimer.singleShot(0, lambda: self._apply_responsive_layout(force=True))

    def changeEvent(self, e):
        # utile quand on sort/entre du plein écran (Qt parfois envoie WindowStateChange)
        super().changeEvent(e)
        if e.type() == QEvent.WindowStateChange:
            QTimer.singleShot(0, self._apply_responsive_layout)

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
        w, h = self.width(), self.height()
        # breakpoint plus robuste (plein écran puis réduction)
        return (w < 980) or (h < 620)

    def _apply_responsive_layout(self, force=False):
        mobile = self._is_mobile()
        mode = self._ui_mode

        if mobile and mode != "mobile":
            # Mobile mode: hide desktop controls; show mobile bar; video takes full width
            self.logo.setVisible(False)
            self.appTitle.setVisible(False)
            self.controlsScroll.setVisible(False)
            self.splitter.setVisible(True)  # on garde splitter pour le layout; mais on masque le panneau
            self.mobilePanel.setVisible(True)

            # force splitter ratio: video only
            self.splitter.widget(0).setVisible(True)
            self.splitter.widget(1).setVisible(False)
            self.splitter.setStretchFactor(0, 1)

            self._populate_drawer_content()
            self._ui_mode = "mobile"

        elif (not mobile) and mode != "desktop":
            # Desktop mode
            self.logo.setVisible(True)
            self.appTitle.setVisible(True)
            self.controlsScroll.setVisible(True)
            self.mobilePanel.setVisible(False)

            self.splitter.widget(0).setVisible(True)
            self.splitter.widget(1).setVisible(True)

            # keep a sensible default split
            QTimer.singleShot(0, lambda: self._ensure_splitter_sizes())
            self._close_drawer()
            self._mount_controls_desktop()
            self._ui_mode = "desktop"

        # If desktop, drawer must be closed
        if (not mobile) and self._open_drawer:
            self._close_drawer()

    def _ensure_splitter_sizes(self):
        # Avoid too small controls on resize
        if not self.splitter.isVisible():
            return
        w = self.splitter.width()
        if w <= 0:
            return
        # video ~65%, panel ~35%
        self.splitter.setSizes([int(w * 0.66), int(w * 0.34)])

    def _apply_ui_scale(self):
        w, h = max(640, self.width()), max(480, self.height())
        ui_scale = max(0.85, min(1.25, min(w / 1280.0, h / 720.0)))
        base_pt = 11.0 * ui_scale

        f = QFont(self.font())
        f.setPointSizeF(base_pt)
        self.setFont(f)

        pad = max(6, int(8 * ui_scale))
        # styles légers: padding/tailles de boutons, sans casser le style global
        self.setStyleSheet(self.styleSheet() + f"""
        QPushButton {{ padding:{pad}px {pad+4}px; }}
        QToolButton#burgerBtn, QToolButton#gearBtn {{
            min-width:{max(42,int(44*ui_scale))}px;
            min-height:{max(42,int(44*ui_scale))}px;
            font-size:{int(18*ui_scale)}px;
        }}
        QPushButton#startBtn {{ background:#16a34a; color:white; font-weight:800; border-radius:16px; }}
        QPushButton#stopBtn  {{ background:#dc2626; color:white; font-weight:800; border-radius:16px; }}
        QPushButton#recBtn   {{ background:#ef4444; color:white; font-weight:800; border-radius:16px; }}
        QPushButton#fullBtn  {{ background:#334155; color:white; font-weight:800; border-radius:16px; }}
        """)

        self._update_logo_pixmap(ui_scale)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

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

    def _emit_camera_changed(self):
        data = self.cameraCombo.currentData()
        self.cameraSelectionChanged.emit(data)

    def _choose_classes(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choisir classes.json", "", "JSON (*.json)")
        if path:
            self.classesEdit.setText(path)
            self.classesPathChanged.emit(path)

    def _choose_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Choisir dossier de sortie", self.dirEdit.text().strip() or "runs")
        if path:
            self.dirEdit.setText(path)
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
        if not hasattr(self, "_taskChecks"):
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
        self.modelCombo.blockSignals(True)
        self.modelCombo.clear()
        if not entries:
            self.modelCombo.addItem("(aucun modèle trouvé…)", "")
            self.modelCombo.setEnabled(False)
        else:
            self.modelCombo.setEnabled(True)
            for label, path in entries:
                self.modelCombo.addItem(label, path)
        self.modelCombo.blockSignals(False)
        self._emit_model_changed()

    def set_cameras(self, entries: List[Tuple[str, object]]):
        self.cameraCombo.blockSignals(True)
        self.cameraCombo.clear()
        for label, data in entries:
            self.cameraCombo.addItem(label, data)
        self.cameraCombo.blockSignals(False)
        self._emit_camera_changed()

    def set_tasks(self, all_tasks: Dict[str, List[str]], selected_tasks: List[str]):
        self.all_tasks = dict(all_tasks or {})
        self.selected_tasks = list(selected_tasks or [])
        if self._is_mobile():
            self._populate_drawer_content()

    def set_device_text(self, txt: str):
        self.deviceLabel.setText(txt)

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
