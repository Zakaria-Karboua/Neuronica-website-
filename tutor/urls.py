from django.urls import path
from . import views

app_name = 'tutor'

urlpatterns = [
    path('ask/', views.ask, name='ask'),
    path('history/', views.history, name='history'),
]
