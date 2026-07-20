from django.db import models


class FocusSession(models.Model):
    """One completed Pomodoro 'flight' — a focus block a user finished on a lesson/project."""

    SESSION_TYPES = [
        ('focus', 'Focus (flight time)'),
        ('short_break', 'Short break (orbit)'),
        ('long_break', 'Long break (refuel)'),
    ]

    user = models.ForeignKey('auth.User', related_name='focus_sessions', on_delete=models.CASCADE)
    lesson = models.ForeignKey(
        'curriculum.Lesson', null=True, blank=True,
        related_name='focus_sessions', on_delete=models.SET_NULL,
    )
    project = models.ForeignKey(
        'curriculum.Project', null=True, blank=True,
        related_name='focus_sessions', on_delete=models.SET_NULL,
    )

    session_type = models.CharField(max_length=20, choices=SESSION_TYPES, default='focus')
    duration_seconds = models.PositiveIntegerField()
    started_at = models.DateTimeField()
    ended_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-ended_at']

    def __str__(self):
        return f"{self.user} — {self.get_session_type_display()} ({self.duration_seconds}s)"


class UserStreak(models.Model):
    """Tracks consecutive days with at least one completed focus session ('missions flown')."""

    user = models.OneToOneField('auth.User', related_name='streak', on_delete=models.CASCADE)
    current_streak = models.PositiveIntegerField(default=0)
    longest_streak = models.PositiveIntegerField(default=0)
    last_active_date = models.DateField(null=True, blank=True)
    total_focus_seconds = models.PositiveBigIntegerField(default=0)

    def __str__(self):
        return f"{self.user} — {self.current_streak} day streak"
