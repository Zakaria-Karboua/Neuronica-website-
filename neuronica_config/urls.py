from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView
from django.templatetags.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('favicon.ico', RedirectView.as_view(url=static('favicon.ico'), permanent=True)),
    path('accounts/', include('allauth.urls')),  # Google + GitHub login lives here
    path('', include('accounts.urls')),
    path('focus/', include('focus.urls')),
    path('tutor/', include('tutor.urls')),
    path('', include('curriculum.urls')),  # star map is the homepage
]
