import launch

if not launch.is_installed("PIL"):
    try:
        launch.run_pip("install Pillow", "requirements for PIL")
    except Exception:
        print("Can't install Pillow. Please follow the readme to install manually")