from django.urls import path
from . import views

app_name = 'focus'

urlpatterns = [
    path('mission-control/', views.mission_control, name='mission_control'),
    path('log-session/', views.log_session, name='log_session'),
    path('dashboard/', views.dashboard, name='dashboard'),
]
