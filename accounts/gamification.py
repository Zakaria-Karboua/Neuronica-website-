"""
Gamification logic: XP scoring, tier/title lookup, and achievement unlocking.
Kept framework-agnostic (plain functions) so it's easy to call from any view
or signal without circular-import headaches.
"""

from dataclasses import dataclass

XP_PER_LESSON = 50
XP_PER_PROJECT = 150
XP_PER_FOCUS_HOUR = 5

# Ordered low -> high. First tier whose min_xp the user has NOT exceeded wins;
# falls through to the last (highest) tier if XP exceeds everything.
TIERS = [
    (0, 'Cadet'),
    (200, 'Pilot'),
    (500, 'Navigator'),
    (1000, 'Commander'),
    (2000, 'Ace Pilot'),
    (4000, 'Legend'),
]


@dataclass
class UserStats:
    """One query pass worth of progress data, shared across XP calc and every achievement rule."""
    lessons_completed: int
    projects_completed: int
    focus_hours: float
    current_streak: int


def _gather_stats(user):
    """
    Single query per table, computed once per call site. Previously,
    check_and_award_achievements called helper functions like
    _lessons_completed_count() once per matching achievement rule (e.g. 3x
    for first/five/ten missions), re-hitting the DB each time. Now it's
    exactly 3 queries total (lessons, projects, streak) no matter how many
    achievement rules exist.
    """
    from curriculum.models import LessonProgress, ProjectProgress
    from focus.models import UserStreak

    lessons_completed = LessonProgress.objects.filter(user=user, completed=True).count()
    projects_completed = ProjectProgress.objects.filter(user=user, completed=True).count()

    try:
        streak = UserStreak.objects.get(user=user)
        focus_hours = streak.total_focus_seconds / 3600
        current_streak = streak.current_streak
    except UserStreak.DoesNotExist:
        focus_hours = 0
        current_streak = 0

    return UserStats(
        lessons_completed=lessons_completed,
        projects_completed=projects_completed,
        focus_hours=focus_hours,
        current_streak=current_streak,
    )


def compute_xp(user, stats=None):
    """Total XP for a user. Pass a pre-fetched `stats` to avoid a redundant query pass."""
    if stats is None:
        stats = _gather_stats(user)
    return int(
        stats.lessons_completed * XP_PER_LESSON
        + stats.projects_completed * XP_PER_PROJECT
        + stats.focus_hours * XP_PER_FOCUS_HOUR
    )


def get_tier(xp):
    """Returns (title, next_tier_title_or_None, xp_needed_for_next_or_None)."""
    current_title = TIERS[0][1]
    next_tier = None
    for i, (threshold, title) in enumerate(TIERS):
        if xp >= threshold:
            current_title = title
            next_tier = TIERS[i + 1] if i + 1 < len(TIERS) else None
        else:
            break
    if next_tier is None:
        return current_title, None, None
    return current_title, next_tier[1], next_tier[0] - xp


# ---------------------------------------------------------------------------
# Achievement rules — each takes a UserStats snapshot (no DB access of its own).
# Adding a new trophy = add a row here (code, title, description, icon, rule).
# ---------------------------------------------------------------------------

ACHIEVEMENT_DEFINITIONS = [
    ('first-mission', 'First Mission', 'Complete your first lesson.', '🎖️',
     lambda s: s.lessons_completed >= 1),
    ('five-missions', 'Squadron Ready', 'Complete 5 lessons.', '🥈',
     lambda s: s.lessons_completed >= 5),
    ('ten-missions', 'Veteran Pilot', 'Complete 10 lessons.', '🥇',
     lambda s: s.lessons_completed >= 10),
    ('first-station', 'First Dock', 'Complete your first project.', '🛰️',
     lambda s: s.projects_completed >= 1),
    ('streak-7', 'Week-Long Flight', 'Reach a 7-day streak.', '🔥',
     lambda s: s.current_streak >= 7),
    ('streak-30', 'Iron Will', 'Reach a 30-day streak.', '💠',
     lambda s: s.current_streak >= 30),
    ('focus-10h', 'Marathon Pilot', 'Log 10 total hours of focus time.', '⏱️',
     lambda s: s.focus_hours >= 10),
    ('focus-50h', 'Deep Space Veteran', 'Log 50 total hours of focus time.', '🌌',
     lambda s: s.focus_hours >= 50),
]


def _phases_fully_cleared(user, already_cleared_phase_numbers):
    """
    Returns list of Phase objects where every lesson is completed by this
    user, skipping phases whose "cleared" trophy this user already has —
    once a phase is cleared it stays cleared, so there's no need to
    re-query its lessons on every single call.
    """
    from curriculum.models import Phase, LessonProgress
    cleared = []
    candidates = Phase.objects.exclude(number__in=already_cleared_phase_numbers).prefetch_related('lessons')
    for phase in candidates:
        lesson_ids = list(phase.lessons.values_list('id', flat=True))
        if not lesson_ids:
            continue
        done = LessonProgress.objects.filter(
            user=user, lesson_id__in=lesson_ids, completed=True
        ).count()
        if done == len(lesson_ids):
            cleared.append(phase)
    return cleared


def check_and_award_achievements(user):
    """
    Call this after any action that could unlock a trophy
    (completing a lesson/project, logging a focus session).
    Idempotent — safe to call often.
    Returns the list of newly-earned Achievement objects (for a toast/notification).
    """
    from .models import Achievement, UserAchievement

    newly_earned = []
    already_earned_codes = set(
        UserAchievement.objects.filter(user=user).values_list('achievement__code', flat=True)
    )

    stats = _gather_stats(user)

    for code, title, description, icon, rule in ACHIEVEMENT_DEFINITIONS:
        if code in already_earned_codes:
            continue
        if rule(stats):
            achievement, _ = Achievement.objects.get_or_create(
                code=code, defaults={'title': title, 'description': description, 'icon': icon}
            )
            UserAchievement.objects.create(user=user, achievement=achievement)
            newly_earned.append(achievement)

    # Dynamic per-phase "cleared" trophies (one per phase, generated on demand).
    # Extract phase numbers already cleared so we don't re-query lessons for them.
    already_cleared_phase_numbers = {
        int(code.split('-')[1]) for code in already_earned_codes
        if code.startswith('phase-') and code.endswith('-cleared')
    }
    for phase in _phases_fully_cleared(user, already_cleared_phase_numbers):
        code = f"phase-{phase.number}-cleared"
        if code in already_earned_codes:
            continue
        achievement, _ = Achievement.objects.get_or_create(
            code=code,
            defaults={
                'title': f"Phase {phase.number} Cleared",
                'description': f"Complete every mission in {phase.title}.",
                'icon': '🌠',
            },
        )
        _, created = UserAchievement.objects.get_or_create(user=user, achievement=achievement)
        if created:
            newly_earned.append(achievement)

    return newly_earned
