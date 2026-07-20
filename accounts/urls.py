from django.urls import path
from . import views

app_name = 'accounts'

urlpatterns = [
    path('profile/', views.my_profile_redirect, name='my_profile'),
    path('profile/edit/', views.profile_edit, name='profile_edit'),
    path('u/<str:username>/', views.profile_view, name='profile_view'),
]
