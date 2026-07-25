"""
Tests for accounts.gamification: XP scoring, tier lookup, and achievement
unlocking. This module is exactly the kind of logic that silently breaks
when someone tweaks XP_PER_LESSON or reorders TIERS — so every constant
and boundary condition gets an explicit assertion here.
"""

from django.contrib.auth.models import User
from django.test import TestCase

from accounts.gamification import (
    ACHIEVEMENT_DEFINITIONS,
    TIERS,
    UserStats,
    check_and_award_achievements,
    compute_xp,
    get_tier,
)
from accounts.models import UserAchievement
from curriculum.models import Lesson, LessonProgress, Phase, Project, ProjectProgress
from focus.models import UserStreak


class ComputeXPTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='pilot', password='x')
        self.phase = Phase.objects.create(number=1, title='Test Phase', slug='test-phase')

    def _make_lesson(self, order):
        return Lesson.objects.create(
            phase=self.phase, order=order, title=f'Lesson {order}',
            slug=f'lesson-{order}', source_filename=f'{order}.md', raw_markdown='x',
        )

    def _make_project(self, order):
        return Project.objects.create(
            phase=self.phase, order=order, title=f'Project {order}',
            slug=f'project-{order}', source_filename=f'{order}.ipynb',
        )

    def test_zero_progress_gives_zero_xp(self):
        self.assertEqual(compute_xp(self.user), 0)

    def test_xp_from_lessons_only(self):
        for i in range(3):
            lesson = self._make_lesson(i)
            LessonProgress.objects.create(user=self.user, lesson=lesson, completed=True)
        self.assertEqual(compute_xp(self.user), 3 * 50)

    def test_incomplete_lessons_dont_count(self):
        lesson = self._make_lesson(1)
        LessonProgress.objects.create(user=self.user, lesson=lesson, completed=False)
        self.assertEqual(compute_xp(self.user), 0)

    def test_xp_from_projects_only(self):
        project = self._make_project(1)
        ProjectProgress.objects.create(user=self.user, project=project, completed=True)
        self.assertEqual(compute_xp(self.user), 150)

    def test_xp_from_focus_hours_only(self):
        UserStreak.objects.create(user=self.user, total_focus_seconds=2 * 3600)  # 2 hours
        self.assertEqual(compute_xp(self.user), 2 * 5)

    def test_partial_focus_hour_is_fractional_not_rounded_up(self):
        # 90 minutes = 1.5 hours -> 1.5 * 5 = 7.5 -> int() truncates to 7
        UserStreak.objects.create(user=self.user, total_focus_seconds=90 * 60)
        self.assertEqual(compute_xp(self.user), 7)

    def test_combined_sources_sum_correctly(self):
        lesson = self._make_lesson(1)
        LessonProgress.objects.create(user=self.user, lesson=lesson, completed=True)
        project = self._make_project(1)
        ProjectProgress.objects.create(user=self.user, project=project, completed=True)
        UserStreak.objects.create(user=self.user, total_focus_seconds=4 * 3600)
        # 50 (lesson) + 150 (project) + 20 (4 hrs * 5) = 220
        self.assertEqual(compute_xp(self.user), 220)

    def test_passing_precomputed_stats_matches_live_query(self):
        lesson = self._make_lesson(1)
        LessonProgress.objects.create(user=self.user, lesson=lesson, completed=True)
        stats = UserStats(lessons_completed=1, projects_completed=0, focus_hours=0, current_streak=0)
        self.assertEqual(compute_xp(self.user, stats=stats), compute_xp(self.user))


class GetTierTests(TestCase):
    def test_zero_xp_is_lowest_tier(self):
        title, next_title, xp_to_next = get_tier(0)
        self.assertEqual(title, 'Cadet')
        self.assertEqual(next_title, 'Pilot')
        self.assertEqual(xp_to_next, 200)

    def test_exact_threshold_counts_as_reaching_that_tier(self):
        # Boundary check: xp == threshold should count as IN that tier, not the one below.
        title, _, _ = get_tier(200)
        self.assertEqual(title, 'Pilot')

    def test_one_below_threshold_stays_in_lower_tier(self):
        title, _, _ = get_tier(199)
        self.assertEqual(title, 'Cadet')

    def test_between_tiers_reports_correct_next_tier_and_remaining_xp(self):
        title, next_title, xp_to_next = get_tier(350)
        self.assertEqual(title, 'Pilot')
        self.assertEqual(next_title, 'Navigator')
        self.assertEqual(xp_to_next, 150)  # 500 - 350

    def test_max_tier_has_no_next_tier(self):
        title, next_title, xp_to_next = get_tier(4000)
        self.assertEqual(title, 'Legend')
        self.assertIsNone(next_title)
        self.assertIsNone(xp_to_next)

    def test_xp_far_beyond_max_tier_still_resolves_to_max(self):
        title, next_title, xp_to_next = get_tier(999_999)
        self.assertEqual(title, 'Legend')
        self.assertIsNone(next_title)

    def test_every_tier_is_reachable_in_order(self):
        # Guards against someone reordering TIERS and silently breaking progression.
        seen_titles = []
        for threshold, _ in TIERS:
            title, _, _ = get_tier(threshold)
            seen_titles.append(title)
        expected = [t for _, t in TIERS]
        self.assertEqual(seen_titles, expected)


class CheckAndAwardAchievementsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='pilot', password='x')
        self.phase = Phase.objects.create(number=1, title='Test Phase', slug='test-phase')

    def _make_lesson(self, order):
        return Lesson.objects.create(
            phase=self.phase, order=order, title=f'Lesson {order}',
            slug=f'lesson-{order}', source_filename=f'{order}.md', raw_markdown='x',
        )

    def test_no_progress_earns_nothing(self):
        earned = check_and_award_achievements(self.user)
        self.assertEqual(earned, [])
        self.assertEqual(UserAchievement.objects.filter(user=self.user).count(), 0)

    def test_first_lesson_earns_first_mission_only(self):
        self._make_lesson(99)  # extra uncompleted lesson so the phase isn't fully cleared
        lesson = self._make_lesson(1)
        LessonProgress.objects.create(user=self.user, lesson=lesson, completed=True)
        earned = check_and_award_achievements(self.user)
        earned_codes = {a.code for a in earned}
        self.assertEqual(earned_codes, {'first-mission'})

    def test_five_lessons_earns_first_and_five_mission_trophies(self):
        self._make_lesson(99)  # extra uncompleted lesson so the phase isn't fully cleared
        for i in range(5):
            lesson = self._make_lesson(i)
            LessonProgress.objects.create(user=self.user, lesson=lesson, completed=True)
        earned = check_and_award_achievements(self.user)
        earned_codes = {a.code for a in earned}
        self.assertEqual(earned_codes, {'first-mission', 'five-missions'})

    def test_already_earned_achievements_are_not_re_awarded(self):
        self._make_lesson(99)  # extra uncompleted lesson so the phase isn't fully cleared
        lesson = self._make_lesson(1)
        LessonProgress.objects.create(user=self.user, lesson=lesson, completed=True)

        first_call = check_and_award_achievements(self.user)
        self.assertEqual(len(first_call), 1)

        # Calling again with no new progress should award nothing new (idempotent).
        second_call = check_and_award_achievements(self.user)
        self.assertEqual(second_call, [])
        self.assertEqual(UserAchievement.objects.filter(user=self.user).count(), 1)

    def test_streak_achievement_unlocks_at_threshold(self):
        UserStreak.objects.create(user=self.user, current_streak=7, total_focus_seconds=0)
        earned = check_and_award_achievements(self.user)
        earned_codes = {a.code for a in earned}
        self.assertIn('streak-7', earned_codes)
        self.assertNotIn('streak-30', earned_codes)

    def test_completing_full_phase_earns_dynamic_phase_cleared_trophy(self):
        lessons = [self._make_lesson(i) for i in range(3)]
        for lesson in lessons:
            LessonProgress.objects.create(user=self.user, lesson=lesson, completed=True)

        earned = check_and_award_achievements(self.user)
        earned_codes = {a.code for a in earned}
        self.assertIn('phase-1-cleared', earned_codes)

        achievement = UserAchievement.objects.get(
            user=self.user, achievement__code='phase-1-cleared'
        ).achievement
        self.assertEqual(achievement.title, 'Phase 1 Cleared')

    def test_partial_phase_completion_does_not_earn_phase_cleared(self):
        lessons = [self._make_lesson(i) for i in range(3)]
        LessonProgress.objects.create(user=self.user, lesson=lessons[0], completed=True)
        LessonProgress.objects.create(user=self.user, lesson=lessons[1], completed=True)
        # lessons[2] left incomplete

        earned = check_and_award_achievements(self.user)
        earned_codes = {a.code for a in earned}
        self.assertNotIn('phase-1-cleared', earned_codes)

    def test_achievement_definitions_have_unique_codes(self):
        # Guards against a copy-paste bug when adding a new trophy.
        codes = [code for code, *_ in ACHIEVEMENT_DEFINITIONS]
        self.assertEqual(len(codes), len(set(codes)))

    def test_steady_state_query_count_is_bounded_with_an_in_progress_phase(self):
        # An uncleared phase must still be re-checked every call (it might have
        # just become cleared), so this isn't zero-cost — but it should stay
        # fixed regardless of how many ACHIEVEMENT_DEFINITIONS exist, which is
        # the actual regression this guards against.
        self._make_lesson(99)  # extra uncompleted lesson so the phase isn't fully cleared
        lesson = self._make_lesson(1)
        LessonProgress.objects.create(user=self.user, lesson=lesson, completed=True)
        check_and_award_achievements(self.user)  # first call, unlocks 'first-mission'

        with self.assertNumQueries(8):
            check_and_award_achievements(self.user)

    def test_cleared_phase_is_not_requeried_on_later_calls(self):
        # This is the actual optimization: once a phase's trophy is earned,
        # subsequent calls should skip re-querying that phase's lessons
        # entirely, rather than re-checking a phase that can't un-clear.
        lesson = self._make_lesson(1)
        LessonProgress.objects.create(user=self.user, lesson=lesson, completed=True)
        first_call = check_and_award_achievements(self.user)
        self.assertIn('phase-1-cleared', {a.code for a in first_call})

        with self.assertNumQueries(5):
            # 1: already_earned_codes, 2: lessons count, 3: projects count,
            # 4: streak lookup, 5: phase list (excludes phase 1, returns empty
            # -> no lesson-counting query happens at all for it).
            second_call = check_and_award_achievements(self.user)
        self.assertEqual(second_call, [])
