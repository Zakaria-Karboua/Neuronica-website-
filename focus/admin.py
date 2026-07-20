from django.contrib import admin
from .models import FocusSession, UserStreak


@admin.register(FocusSession)
class FocusSessionAdmin(admin.ModelAdmin):
    list_display = ('user', 'session_type', 'duration_seconds', 'lesson', 'project', 'ended_at')
    list_filter = ('session_type',)


@admin.register(UserStreak)
class UserStreakAdmin(admin.ModelAdmin):
    list_display = ('user', 'current_streak', 'longest_streak', 'total_focus_seconds')
