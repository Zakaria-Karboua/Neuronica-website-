import json
import os

import requests
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_GET, require_POST

from curriculum.models import Lesson
from .models import ChatMessage

GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash-lite')
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

HISTORY_TURNS = 10          # how many past turns to send back to Gemini for context
LESSON_CONTEXT_CHARS = 3000  # truncate lesson markdown so we don't burn the token budget

SYSTEM_INSTRUCTION_BASE = (
    "You are the AI tutor inside Neuronica, a space-themed learning platform for "
    "statistics, AI, and software engineering. Be encouraging, concise, and clear. "
    "Prefer short explanations with a concrete example over long lectures. "
    "If the learner is looking at a specific lesson, ground your answer in that "
    "lesson's material first before bringing in outside knowledge. "
    "Use plain text or simple markdown — no need for headers."
)


def _build_system_instruction(lesson_id):
    if not lesson_id:
        return SYSTEM_INSTRUCTION_BASE
    try:
        lesson = Lesson.objects.get(id=lesson_id)
    except Lesson.DoesNotExist:
        return SYSTEM_INSTRUCTION_BASE
    excerpt = lesson.raw_markdown[:LESSON_CONTEXT_CHARS]
    return (
        f"{SYSTEM_INSTRUCTION_BASE}\n\n"
        f"The learner is currently reading the lesson '{lesson.title}'. "
        f"Here is that lesson's content for context:\n\n{excerpt}"
    )


@login_required
@require_POST
def ask(request):
    if not os.environ.get('GEMINI_API_KEY'):
        return JsonResponse({
            'error': 'The AI tutor is not configured yet — add GEMINI_API_KEY to your .env file.'
        }, status=503)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Malformed request.'}, status=400)

    question = (data.get('message') or '').strip()
    lesson_id = data.get('lesson_id') or None
    if not question:
        return JsonResponse({'error': 'Empty message.'}, status=400)

    # Save the user's turn immediately, so it's captured even if the API call fails.
    ChatMessage.objects.create(user=request.user, lesson_id=lesson_id, role='user', content=question)

    # Build conversation history (recent turns) in Gemini's {role, parts} format.
    recent = list(
        ChatMessage.objects.filter(user=request.user).order_by('-created_at')[:HISTORY_TURNS * 2]
    )[::-1]
    contents = [
        {'role': 'user' if m.role == 'user' else 'model', 'parts': [{'text': m.content}]}
        for m in recent
    ]

    payload = {
        'contents': contents,
        'system_instruction': {'parts': [{'text': _build_system_instruction(lesson_id)}]},
        'generationConfig': {'maxOutputTokens': 500, 'temperature': 0.6},
    }

    try:
        resp = requests.post(
            GEMINI_URL,
            headers={
                'x-goog-api-key': os.environ['GEMINI_API_KEY'],
                'Content-Type': 'application/json',
            },
            json=payload,
            timeout=20,
        )
    except requests.RequestException:
        return JsonResponse({'error': "Couldn't reach the AI tutor right now. Try again shortly."}, status=502)

    if resp.status_code == 429:
        return JsonResponse({
            'error': "The AI tutor is getting a lot of questions right now (free-tier limit hit). Try again in a minute."
        }, status=429)

    if resp.status_code != 200:
        return JsonResponse({'error': f"AI tutor error ({resp.status_code}). Try again shortly."}, status=502)

    reply_text = _extract_text(resp.json())
    if not reply_text:
        reply_text = "I couldn't come up with a response for that — try rephrasing your question."

    ChatMessage.objects.create(user=request.user, lesson_id=lesson_id, role='assistant', content=reply_text)

    return JsonResponse({'reply': reply_text})


def _extract_text(gemini_response):
    try:
        candidates = gemini_response.get('candidates', [])
        if not candidates:
            return None
        parts = candidates[0].get('content', {}).get('parts', [])
        return ''.join(p.get('text', '') for p in parts).strip()
    except (AttributeError, IndexError, KeyError, TypeError):
        return None


@login_required
@require_GET
def history(request):
    lesson_id = request.GET.get('lesson_id') or None
    qs = ChatMessage.objects.filter(user=request.user)
    if lesson_id:
        qs = qs.filter(lesson_id=lesson_id)
    messages = list(qs.order_by('-created_at')[:40])[::-1]
    return JsonResponse({
        'messages': [{'role': m.role, 'content': m.content} for m in messages]
    })
