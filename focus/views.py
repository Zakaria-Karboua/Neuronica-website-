import json
from datetime import timedelta

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.http import require_POST

from curriculum.models import Lesson, Project
from .models import FocusSession, UserStreak


def mission_control(request):
    """The Pomodoro timer page ('Mission Control')."""
    lessons = Lesson.objects.select_related('phase').all() if request.user.is_authenticated else []
    return render(request, 'focus/mission_control.html', {'lessons': lessons})


@login_required
@require_POST
def log_session(request):
    """
    Called by the timer's JS when a focus/break block completes.
    Expects JSON: { session_type, duration_seconds, started_at, lesson_id?, project_id? }
    """
    data = json.loads(request.body)

    session = FocusSession.objects.create(
        user=request.user,
        session_type=data.get('session_type', 'focus'),
        duration_seconds=int(data['duration_seconds']),
        started_at=data['started_at'],
        lesson_id=data.get('lesson_id') or None,
        project_id=data.get('project_id') or None,
    )

    streak = _update_streak(request.user, session)
    from accounts.gamification import check_and_award_achievements
    check_and_award_achievements(request.user)

    return JsonResponse({
        'ok': True,
        'current_streak': streak.current_streak,
        'total_focus_seconds': streak.total_focus_seconds,
    })


def _update_streak(user, session):
    streak, _ = UserStreak.objects.get_or_create(user=user)
    today = timezone.localdate()

    if session.session_type == 'focus':
        streak.total_focus_seconds += session.duration_seconds

    if streak.last_active_date == today:
        pass  # already counted today
    elif streak.last_active_date == today - timedelta(days=1):
        streak.current_streak += 1
        streak.last_active_date = today
    else:
        streak.current_streak = 1
        streak.last_active_date = today

    streak.longest_streak = max(streak.longest_streak, streak.current_streak)
    streak.save()
    return streak


@login_required
def dashboard(request):
    """Shows time spent per phase/lesson, streaks, totals."""
    streak, _ = UserStreak.objects.get_or_create(user=request.user)
    sessions = (
        FocusSession.objects.filter(user=request.user, session_type='focus')
        .select_related('lesson__phase', 'project__phase')
    )

    per_lesson = {}
    for s in sessions:
        key = s.lesson.title if s.lesson else (s.project.title if s.project else 'Unassigned')
        per_lesson[key] = per_lesson.get(key, 0) + s.duration_seconds

    return render(request, 'focus/dashboard.html', {
        'streak': streak,
        'total_focus_display': _format_duration(streak.total_focus_seconds),
        'per_lesson': [
            (name, _format_duration(seconds)) for name, seconds in
            sorted(per_lesson.items(), key=lambda x: -x[1])
        ],
    })


def _format_duration(total_seconds):
    """Turns 5400 into '1h 30m', 90 into '1m 30s', etc."""
    total_seconds = int(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"
