from django import forms
from .models import Profile


class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['display_name', 'bio', 'avatar_emoji', 'theme_preference', 'github_url', 'website_url']
        widgets = {
            'bio': forms.Textarea(attrs={'rows': 3, 'maxlength': 300}),
            'avatar_emoji': forms.Select(),
            'theme_preference': forms.Select(),
        }
