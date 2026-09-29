"""Trainer service: runs fine-tuning jobs queued by the web app.

It runs in its own container (``docker compose`` service ``trainer``), shares
the ``/data`` volume and the GPU with the app, and reads its work from the
``training_runs`` table of the app's SQLite database.  See ``trainer.service``.
"""
