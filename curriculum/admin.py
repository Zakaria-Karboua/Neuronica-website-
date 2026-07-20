from django.contrib import admin
from .models import Phase, Lesson, Project, LessonProgress


@admin.register(Phase)
class PhaseAdmin(admin.ModelAdmin):
    list_display = ('number', 'title', 'slug')
    prepopulated_fields = {'slug': ('title',)}


@admin.register(Lesson)
class LessonAdmin(admin.ModelAdmin):
    list_display = ('phase', 'order', 'title', 'source_filename', 'updated_at')
    list_filter = ('phase',)
    search_fields = ('title', 'raw_markdown')


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ('phase', 'order', 'title', 'source_filename', 'updated_at')
    list_filter = ('phase',)


@admin.register(LessonProgress)
class LessonProgressAdmin(admin.ModelAdmin):
    list_display = ('user', 'lesson', 'completed', 'completed_at')
    list_filter = ('completed',)
