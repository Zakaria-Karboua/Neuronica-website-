from django.db import models


AVATAR_CHOICES = [
    ('🧑‍🚀', 'Astronaut'),
    ('🤖', 'Droid'),
    ('👽', 'Alien'),
    ('🦾', 'Cyborg Arm'),
    ('🛸', 'UFO'),
    ('⚡', 'Arc Reactor'),
    ('🎯', 'Targeting'),
    ('🧠', 'Neural Core'),
]

THEME_CHOICES = [
    ('dark', 'Dark'),
    ('light', 'Light'),
]


class Profile(models.Model):
    """Extra, customisable info attached to every user (auto-created on signup)."""

    user = models.OneToOneField('auth.User', related_name='profile', on_delete=models.CASCADE)
    display_name = models.CharField(max_length=60, blank=True)
    bio = models.TextField(max_length=300, blank=True)
    avatar_emoji = models.CharField(max_length=8, choices=AVATAR_CHOICES, default='🧑‍🚀')
    theme_preference = models.CharField(max_length=10, choices=THEME_CHOICES, default='dark')
    github_url = models.URLField(blank=True)
    website_url = models.URLField(blank=True)

    def __str__(self):
        return self.display_name or self.user.username

    @property
    def name(self):
        return self.display_name or self.user.username


class Achievement(models.Model):
    """A trophy definition — e.g. 'First Mission', 'Phase 1 Cleared'."""

    code = models.SlugField(unique=True)
    title = models.CharField(max_length=100)
    description = models.CharField(max_length=200)
    icon = models.CharField(max_length=8, default='🏆')

    def __str__(self):
        return self.title


class UserAchievement(models.Model):
    """A trophy a specific user has actually unlocked."""

    user = models.ForeignKey('auth.User', related_name='achievements', on_delete=models.CASCADE)
    achievement = models.ForeignKey(Achievement, related_name='earned_by', on_delete=models.CASCADE)
    earned_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [('user', 'achievement')]
        ordering = ['-earned_at']

    def __str__(self):
        return f"{self.user} — {self.achievement.title}"
