from voicemimic.app import build_app


def test_workbench_builds_without_loading_model():
    app = build_app()
    assert app.config["title"] == "VoiceMimic · Timbre Lab"
    assert len(app.config["dependencies"]) >= 2
