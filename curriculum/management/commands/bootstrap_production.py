"""
Runs automatically as part of the Render build step (see render.yaml / README).
Safe to run on every deploy: creating an existing superuser is skipped,
and import_phase updates existing lessons in place rather than duplicating.

Superuser is created from env vars if DJANGO_SUPERUSER_* are all set:
  DJANGO_SUPERUSER_USERNAME, DJANGO_SUPERUSER_EMAIL, DJANGO_SUPERUSER_PASSWORD
"""

import os

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "One-shot production bootstrap: create superuser + import committed phase content."

    def handle(self, *args, **options):
        self._create_superuser_if_configured()
        self._import_committed_phases()

    def _create_superuser_if_configured(self):
        User = get_user_model()
        username = os.environ.get('DJANGO_SUPERUSER_USERNAME')
        email = os.environ.get('DJANGO_SUPERUSER_EMAIL')
        password = os.environ.get('DJANGO_SUPERUSER_PASSWORD')

        if not (username and email and password):
            self.stdout.write('Skipping superuser creation (DJANGO_SUPERUSER_* env vars not fully set).')
            return

        if User.objects.filter(username=username).exists():
            self.stdout.write(f'Superuser "{username}" already exists — skipping.')
            return

        User.objects.create_superuser(username=username, email=email, password=password)
        self.stdout.write(self.style.SUCCESS(f'Created superuser "{username}".'))

    def _import_committed_phases(self):
        """
        Looks for content/phaseN-title-with-dashes/ folders in the repo and
        imports each one. Add a new phase later by committing e.g.
        content/phase2-statistics-and-probability/ — no code changes needed.
        """
        import re
        from pathlib import Path

        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        content_dir = base_dir / 'content'

        if not content_dir.is_dir():
            self.stdout.write('No content/ directory found — skipping phase import.')
            return

        pattern = re.compile(r'^phase(\d+)-(.+)$')
        for folder in sorted(content_dir.iterdir()):
            if not folder.is_dir():
                continue
            match = pattern.match(folder.name)
            if not match:
                self.stdout.write(self.style.WARNING(
                    f'Skipping "{folder.name}" — expected folder name like "phase1-programming-foundations".'
                ))
                continue
            number = int(match.group(1))
            title = match.group(2).replace('-', ' ').replace('_', ' ').title()

            self.stdout.write(f'Importing {folder.name} as Phase {number} — {title}...')
            call_command('import_phase', str(folder), number=number, title=title)
