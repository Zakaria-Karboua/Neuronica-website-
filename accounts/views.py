from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404, redirect, render

from curriculum.models import LessonProgress, ProjectProgress
from focus.models import UserStreak

from .forms import ProfileForm
from .gamification import compute_xp, get_tier, check_and_award_achievements
from .models import Profile, UserAchievement


def profile_view(request, username):
    """Public profile: XP, tier, trophies, stats. Anyone can view."""
    user = get_object_or_404(User, username=username)
    profile, _ = Profile.objects.get_or_create(user=user)

    xp = compute_xp(user)
    tier, next_tier, xp_to_next = get_tier(xp)
    if xp_to_next:
        tier_progress_percent = int(100 * xp / (xp + xp_to_next))
    else:
        tier_progress_percent = 100

    trophies = (
        UserAchievement.objects.filter(user=user)
        .select_related('achievement')
        .order_by('-earned_at')
    )

    try:
        streak = UserStreak.objects.get(user=user)
    except UserStreak.DoesNotExist:
        streak = None

    return render(request, 'accounts/profile_view.html', {
        'profile_user': user,
        'profile': profile,
        'xp': xp,
        'tier': tier,
        'next_tier': next_tier,
        'xp_to_next': xp_to_next,
        'tier_progress_percent': tier_progress_percent,
        'trophies': trophies,
        'streak': streak,
        'lessons_completed': LessonProgress.objects.filter(user=user, completed=True).count(),
        'projects_completed': ProjectProgress.objects.filter(user=user, completed=True).count(),
        'is_own_profile': request.user.is_authenticated and request.user == user,
    })


@login_required
def profile_edit(request):
    profile, _ = Profile.objects.get_or_create(user=request.user)

    if request.method == 'POST':
        form = ProfileForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            return redirect('accounts:profile_view', username=request.user.username)
    else:
        form = ProfileForm(instance=profile)

    return render(request, 'accounts/profile_edit.html', {'form': form})


@login_required
def my_profile_redirect(request):
    """/profile/ -> redirects to /u/<my-username>/, also triggers an achievement check."""
    check_and_award_achievements(request.user)
    return redirect('accounts:profile_view', username=request.user.username)
