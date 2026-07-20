from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('allauth.urls')),  # Google + GitHub login lives here
    path('', include('accounts.urls')),
    path('focus/', include('focus.urls')),
    path('tutor/', include('tutor.urls')),
    path('', include('curriculum.urls')),  # star map is the homepage
]
