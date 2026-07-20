"""
Gamification logic: XP scoring, tier/title lookup, and achievement unlocking.
Kept framework-agnostic (plain functions) so it's easy to call from any view
or signal without circular-import headaches.
"""

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


def compute_xp(user):
    """Total XP for a user, computed live from their progress records."""
    from curriculum.models import LessonProgress, ProjectProgress
    from focus.models import UserStreak

    lessons_done = LessonProgress.objects.filter(user=user, completed=True).count()
    projects_done = ProjectProgress.objects.filter(user=user, completed=True).count()

    try:
        streak = UserStreak.objects.get(user=user)
        focus_hours = streak.total_focus_seconds / 3600
    except UserStreak.DoesNotExist:
        focus_hours = 0

    return int(
        lessons_done * XP_PER_LESSON
        + projects_done * XP_PER_PROJECT
        + focus_hours * XP_PER_FOCUS_HOUR
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
# Achievement rules — each returns True if the achievement's condition is met.
# Adding a new trophy = add a row here (code, title, description, icon, rule).
# ---------------------------------------------------------------------------

def _lessons_completed_count(user):
    from curriculum.models import LessonProgress
    return LessonProgress.objects.filter(user=user, completed=True).count()


def _projects_completed_count(user):
    from curriculum.models import ProjectProgress
    return ProjectProgress.objects.filter(user=user, completed=True).count()


def _total_focus_hours(user):
    from focus.models import UserStreak
    try:
        return UserStreak.objects.get(user=user).total_focus_seconds / 3600
    except UserStreak.DoesNotExist:
        return 0


def _current_streak(user):
    from focus.models import UserStreak
    try:
        return UserStreak.objects.get(user=user).current_streak
    except UserStreak.DoesNotExist:
        return 0


def _phases_fully_cleared(user):
    """Returns list of Phase objects where every lesson is completed by this user."""
    from curriculum.models import Phase, LessonProgress
    cleared = []
    for phase in Phase.objects.prefetch_related('lessons'):
        lesson_ids = list(phase.lessons.values_list('id', flat=True))
        if not lesson_ids:
            continue
        done = LessonProgress.objects.filter(
            user=user, lesson_id__in=lesson_ids, completed=True
        ).count()
        if done == len(lesson_ids):
            cleared.append(phase)
    return cleared


ACHIEVEMENT_DEFINITIONS = [
    ('first-mission', 'First Mission', 'Complete your first lesson.', '🎖️',
     lambda u: _lessons_completed_count(u) >= 1),
    ('five-missions', 'Squadron Ready', 'Complete 5 lessons.', '🥈',
     lambda u: _lessons_completed_count(u) >= 5),
    ('ten-missions', 'Veteran Pilot', 'Complete 10 lessons.', '🥇',
     lambda u: _lessons_completed_count(u) >= 10),
    ('first-station', 'First Dock', 'Complete your first project.', '🛰️',
     lambda u: _projects_completed_count(u) >= 1),
    ('streak-7', 'Week-Long Flight', 'Reach a 7-day streak.', '🔥',
     lambda u: _current_streak(u) >= 7),
    ('streak-30', 'Iron Will', 'Reach a 30-day streak.', '💠',
     lambda u: _current_streak(u) >= 30),
    ('focus-10h', 'Marathon Pilot', 'Log 10 total hours of focus time.', '⏱️',
     lambda u: _total_focus_hours(u) >= 10),
    ('focus-50h', 'Deep Space Veteran', 'Log 50 total hours of focus time.', '🌌',
     lambda u: _total_focus_hours(u) >= 50),
]


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

    for code, title, description, icon, rule in ACHIEVEMENT_DEFINITIONS:
        if code in already_earned_codes:
            continue
        if rule(user):
            achievement, _ = Achievement.objects.get_or_create(
                code=code, defaults={'title': title, 'description': description, 'icon': icon}
            )
            UserAchievement.objects.create(user=user, achievement=achievement)
            newly_earned.append(achievement)

    # Dynamic per-phase "cleared" trophies (one per phase, generated on demand)
    for phase in _phases_fully_cleared(user):
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
