from django.urls import path
from . import views

app_name = 'curriculum'

urlpatterns = [
    path('', views.home, name='home'),
    path('academy/', views.star_map, name='star_map'),
    path('phase/<slug:slug>/', views.phase_detail, name='phase_detail'),
    path('phase/<slug:phase_slug>/lesson/<slug:lesson_slug>/', views.lesson_detail, name='lesson_detail'),
    path('phase/<slug:phase_slug>/project/<slug:project_slug>/', views.project_detail, name='project_detail'),
    path('lesson/<int:lesson_id>/toggle-complete/', views.toggle_lesson_complete, name='toggle_lesson_complete'),
    path('project/<int:project_id>/toggle-complete/', views.toggle_project_complete, name='toggle_project_complete'),
]
