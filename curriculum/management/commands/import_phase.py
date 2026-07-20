"""
Usage:
    python manage.py import_phase /path/to/phase1_extracted --number 1 --title "Programming Foundations"

What it does:
  - Scans the folder for `NN-slug-name.md` files (numeric prefix = order).
  - Reads the first `# ` heading in each file as the lesson title, unless
    --title-from-filename is passed.
  - Renders markdown -> HTML (tables, fenced code, math) and stores BOTH the
    raw markdown and the rendered HTML on the Lesson.
  - Scans for `.ipynb` files in the same folder and converts each to HTML via
    `jupyter nbconvert`, storing the result on a Project ("Station").
  - Re-running this command on the same phase is safe: existing Lesson/Project
    rows (matched by slug) are updated in place, not duplicated.
"""

import re
import subprocess
import tempfile
from pathlib import Path

import markdown
from django.core.management.base import BaseCommand, CommandError
from django.utils.text import slugify

from curriculum.models import Phase, Lesson, Project

MD_EXTENSIONS = [
    'tables',
    'fenced_code',
    'toc',
    'pymdownx.arithmatex',
    'pymdownx.highlight',
    'pymdownx.superfences',
]
MD_EXTENSION_CONFIGS = {
    'pymdownx.arithmatex': {'generic': True},  # wraps $...$ / $$...$$ for KaTeX
    'pymdownx.highlight': {
        'use_pygments': True,
        'css_class': 'codehilite',
        'pygments_style': 'monokai',  # matches static/css/pygments.css — regenerate both together if changed
    },
}

FILENAME_ORDER_RE = re.compile(r'^(\d+)[-_.](.+)$')


def render_markdown(text: str) -> str:
    return markdown.markdown(
        text,
        extensions=MD_EXTENSIONS,
        extension_configs=MD_EXTENSION_CONFIGS,
    )


def extract_title(md_text: str, fallback: str) -> str:
    for line in md_text.splitlines():
        line = line.strip()
        if line.startswith('# '):
            # Handle "# Phase 1 · Lesson 1 — Python Fundamentals" -> take part after last dash/em-dash
            heading = line.lstrip('#').strip()
            for sep in (' — ', ' - ', '—'):
                if sep in heading:
                    heading = heading.split(sep)[-1].strip()
            return heading
    return fallback


def convert_notebook_to_html(ipynb_path: Path) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                'jupyter', 'nbconvert', '--to', 'html',
                '--template', 'basic',  # body-only fragment, easier to embed/style
                '--output-dir', tmp,
                str(ipynb_path),
            ],
            check=True,
            capture_output=True,
        )
        out_file = Path(tmp) / (ipynb_path.stem + '.html')
        return out_file.read_text(encoding='utf-8')


class Command(BaseCommand):
    help = "Import a phase folder of .md lessons (and .ipynb projects) into the database."

    def add_arguments(self, parser):
        parser.add_argument('folder', type=str, help='Path to the phase folder')
        parser.add_argument('--number', type=int, required=True, help='Phase number, e.g. 1')
        parser.add_argument('--title', type=str, required=True, help='Phase title, e.g. "Programming Foundations"')
        parser.add_argument('--skip-notebooks', action='store_true', help='Skip .ipynb conversion (e.g. nbconvert not installed)')

    def handle(self, *args, **options):
        folder = Path(options['folder']).resolve()
        if not folder.is_dir():
            raise CommandError(f"Not a directory: {folder}")

        phase, created = Phase.objects.update_or_create(
            number=options['number'],
            defaults={
                'title': options['title'],
                'slug': slugify(f"phase-{options['number']}-{options['title']}"),
                'folder_name': folder.name,
            },
        )
        self.stdout.write(self.style.SUCCESS(
            f"{'Created' if created else 'Updated'} {phase}"
        ))

        # --- Lessons (.md files) ---
        md_files = sorted(folder.glob('*.md'))
        lesson_count = 0
        for path in md_files:
            if path.name.lower() == 'readme.md':
                continue  # phase overview, not a lesson

            match = FILENAME_ORDER_RE.match(path.stem)
            if match:
                order = int(match.group(1))
                slug_source = match.group(2)
            else:
                order = 0
                slug_source = path.stem

            raw_md = path.read_text(encoding='utf-8')
            title = extract_title(raw_md, fallback=slug_source.replace('-', ' ').title())
            html = render_markdown(raw_md)
            slug = slugify(slug_source)

            Lesson.objects.update_or_create(
                phase=phase,
                slug=slug,
                defaults={
                    'order': order,
                    'title': title,
                    'source_filename': path.name,
                    'raw_markdown': raw_md,
                    'rendered_html': html,
                },
            )
            lesson_count += 1
            self.stdout.write(f"  Lesson {order:02d}: {title}")

        # --- Projects (.ipynb files) ---
        project_count = 0
        if not options['skip_notebooks']:
            ipynb_files = sorted(folder.glob('*.ipynb'))
            for path in ipynb_files:
                match = FILENAME_ORDER_RE.match(path.stem)
                order = int(match.group(1)) if match else 0
                slug_source = match.group(2) if match else path.stem
                title = slug_source.replace('-', ' ').replace('_', ' ').title()

                try:
                    html = convert_notebook_to_html(path)
                except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                    self.stderr.write(self.style.WARNING(
                        f"  Skipped notebook {path.name}: nbconvert failed ({exc})"
                    ))
                    continue

                Project.objects.update_or_create(
                    phase=phase,
                    slug=slugify(slug_source),
                    defaults={
                        'order': order,
                        'title': title,
                        'source_filename': path.name,
                        'rendered_html': html,
                    },
                )
                project_count += 1
                self.stdout.write(f"  Station: {title}")

        self.stdout.write(self.style.SUCCESS(
            f"Done: {lesson_count} lesson(s), {project_count} project(s) imported into {phase}."
        ))
