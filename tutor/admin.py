from django.contrib import admin
from .models import ChatMessage


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('user', 'role', 'lesson', 'created_at')
    list_filter = ('role',)
    search_fields = ('content',)
