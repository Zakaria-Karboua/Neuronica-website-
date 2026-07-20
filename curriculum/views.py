from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from accounts.gamification import check_and_award_achievements
from .models import Phase, Lesson, Project, LessonProgress, ProjectProgress


def home(request):
    """The landing page: what Neuronica is, and the phase roadmap."""
    phases = Phase.objects.prefetch_related('lessons', 'projects').all()
    return render(request, 'curriculum/home.html', {'phases': phases})


def star_map(request):
    """Home page: the galaxy view of all phases (star systems)."""
    phases = Phase.objects.prefetch_related('lessons', 'projects').all()

    completed_ids = set()
    if request.user.is_authenticated:
        completed_ids = set(
            LessonProgress.objects.filter(user=request.user, completed=True)
            .values_list('lesson_id', flat=True)
        )

    return render(request, 'curriculum/star_map.html', {
        'phases': phases,
        'completed_ids': completed_ids,
    })


def phase_detail(request, slug):
    """A single star system: its lessons (planets) and projects (stations)."""
    phase = get_object_or_404(Phase, slug=slug)
    completed_ids = set()
    if request.user.is_authenticated:
        completed_ids = set(
            LessonProgress.objects.filter(
                user=request.user, completed=True, lesson__phase=phase
            ).values_list('lesson_id', flat=True)
        )
    return render(request, 'curriculum/phase_detail.html', {
        'phase': phase,
        'completed_ids': completed_ids,
    })


def lesson_detail(request, phase_slug, lesson_slug):
    lesson = get_object_or_404(Lesson, phase__slug=phase_slug, slug=lesson_slug)
    is_completed = False
    if request.user.is_authenticated:
        is_completed = LessonProgress.objects.filter(
            user=request.user, lesson=lesson, completed=True
        ).exists()

    siblings = list(lesson.phase.lessons.all())
    idx = siblings.index(lesson)
    prev_lesson = siblings[idx - 1] if idx > 0 else None
    next_lesson = siblings[idx + 1] if idx < len(siblings) - 1 else None

    return render(request, 'curriculum/lesson_detail.html', {
        'lesson': lesson,
        'is_completed': is_completed,
        'prev_lesson': prev_lesson,
        'next_lesson': next_lesson,
    })


def project_detail(request, phase_slug, project_slug):
    project = get_object_or_404(Project, phase__slug=phase_slug, slug=project_slug)
    is_completed = False
    if request.user.is_authenticated:
        is_completed = ProjectProgress.objects.filter(
            user=request.user, project=project, completed=True
        ).exists()
    return render(request, 'curriculum/project_detail.html', {
        'project': project,
        'is_completed': is_completed,
    })


@login_required
@require_POST
def toggle_lesson_complete(request, lesson_id):
    lesson = get_object_or_404(Lesson, id=lesson_id)
    progress, _ = LessonProgress.objects.get_or_create(user=request.user, lesson=lesson)
    progress.completed = not progress.completed
    progress.completed_at = timezone.now() if progress.completed else None
    progress.save()
    check_and_award_achievements(request.user)
    return render(request, 'curriculum/_completion_button.html', {
        'lesson': lesson,
        'is_completed': progress.completed,
    })


@login_required
@require_POST
def toggle_project_complete(request, project_id):
    project = get_object_or_404(Project, id=project_id)
    progress, _ = ProjectProgress.objects.get_or_create(user=request.user, project=project)
    progress.completed = not progress.completed
    progress.completed_at = timezone.now() if progress.completed else None
    progress.save()
    check_and_award_achievements(request.user)
    return render(request, 'curriculum/_project_completion_button.html', {
        'project': project,
        'is_completed': progress.completed,
    })
