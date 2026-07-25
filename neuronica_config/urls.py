from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    # Plain STATIC_URL path, not the {% static %}/manifest lookup — this runs
    # at import time, so depending on the Whitenoise manifest here would crash
    # the entire app at startup if collectstatic hadn't run yet.
    path('favicon.ico', RedirectView.as_view(url='/static/favicon.ico', permanent=True)),
    path('accounts/', include('allauth.urls')),  # Google + GitHub login lives here
    path('', include('accounts.urls')),
    path('focus/', include('focus.urls')),
    path('tutor/', include('tutor.urls')),
    path('', include('curriculum.urls')),  # star map is the homepage
]
