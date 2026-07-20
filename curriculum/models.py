from django.db import models
from django.urls import reverse


class Phase(models.Model):
    """A 'Star System' — one phase of the curriculum (e.g. Phase 1 - Programming Foundations)."""

    number = models.PositiveIntegerField(unique=True)
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    description = models.TextField(blank=True)
    folder_name = models.CharField(
        max_length=200,
        help_text="Source folder this phase was imported from, e.g. phase-1-programming-foundations",
        blank=True,
    )

    class Meta:
        ordering = ['number']

    def __str__(self):
        return f"Phase {self.number} — {self.title}"

    def get_absolute_url(self):
        return reverse('curriculum:phase_detail', kwargs={'slug': self.slug})


class Lesson(models.Model):
    """A 'Mission' — one concept taught by a single .md file."""

    phase = models.ForeignKey(Phase, related_name='lessons', on_delete=models.CASCADE)
    order = models.PositiveIntegerField(help_text="Order within the phase, from filename prefix (01-, 02-, ...)")
    title = models.CharField(max_length=250)
    slug = models.SlugField()
    source_filename = models.CharField(max_length=250)

    raw_markdown = models.TextField()
    rendered_html = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['phase__number', 'order']
        unique_together = [('phase', 'slug')]

    def __str__(self):
        return f"{self.phase.number}.{self.order} {self.title}"

    def get_absolute_url(self):
        return reverse('curriculum:lesson_detail', kwargs={
            'phase_slug': self.phase.slug,
            'lesson_slug': self.slug,
        })


class Project(models.Model):
    """A 'Station' — a hands-on project attached to a phase, sourced from a .ipynb file."""

    phase = models.ForeignKey(Phase, related_name='projects', on_delete=models.CASCADE)
    order = models.PositiveIntegerField(default=0)
    title = models.CharField(max_length=250)
    slug = models.SlugField()
    source_filename = models.CharField(max_length=250)

    rendered_html = models.TextField(
        blank=True,
        help_text="HTML produced by `jupyter nbconvert --to html` at import time.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['phase__number', 'order']
        unique_together = [('phase', 'slug')]

    def __str__(self):
        return f"{self.phase.number} Station — {self.title}"

    def get_absolute_url(self):
        return reverse('curriculum:project_detail', kwargs={
            'phase_slug': self.phase.slug,
            'project_slug': self.slug,
        })


class LessonProgress(models.Model):
    """Tracks whether a given user has completed a given lesson (lights up the planet)."""

    user = models.ForeignKey('auth.User', related_name='lesson_progress', on_delete=models.CASCADE)
    lesson = models.ForeignKey(Lesson, related_name='progress', on_delete=models.CASCADE)
    completed = models.BooleanField(default=False)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = [('user', 'lesson')]


class ProjectProgress(models.Model):
    """Tracks whether a given user has completed a given project/station."""

    user = models.ForeignKey('auth.User', related_name='project_progress', on_delete=models.CASCADE)
    project = models.ForeignKey(Project, related_name='progress', on_delete=models.CASCADE)
    completed = models.BooleanField(default=False)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = [('user', 'project')]
