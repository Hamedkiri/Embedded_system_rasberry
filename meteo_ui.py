def reload_models(self):
    self.modelCombo.clear()
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    files = []
    # Recherche récursive des poids supportés
    for ext in ("**/*.pt", "**/*.pth", "**/*.tflite", "**/*.onnx"):
        files += glob.glob(str(models_dir / ext), recursive=True)
    files = sorted(files)

    if not files:
        self.modelCombo.addItem("(aucun modèle trouvé…)")
        self.modelCombo.setEnabled(False)
    else:
        self.modelCombo.setEnabled(True)
        # Afficher un chemin relatif propre
        for f in files:
            rel = str(Path(f).resolve().relative_to(models_dir.resolve()))
            self.modelCombo.addItem(str(models_dir / rel))
