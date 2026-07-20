from django.db import models


class ChatMessage(models.Model):
    """One turn in a user's conversation with the AI tutor."""

    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]

    user = models.ForeignKey('auth.User', related_name='chat_messages', on_delete=models.CASCADE)
    lesson = models.ForeignKey(
        'curriculum.Lesson', null=True, blank=True,
        related_name='chat_messages', on_delete=models.SET_NULL,
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.user} [{self.role}]: {self.content[:40]}"
