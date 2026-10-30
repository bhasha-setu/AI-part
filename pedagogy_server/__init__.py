"""Pedagogy server: transcript in, reply out.

The server frames each turn, asks an LLM judge for a verdict through a narrow,
validated contract, applies guardrails, falls back to the rule-based response when
the judge's output is unusable, and lets the existing lesson code decide what happens
next. Speech-to-text and text-to-speech are adapters with stubs (``adapters``). Nothing
here imports an audio library.

    from guided_practice.course import load_course
    from pedagogy_server.judge import StubJudge
    from pedagogy_server.pipeline import PedagogyServer

    course = load_course("english_for_hindi_a1")
    server = PedagogyServer(course, "introducing_yourself", "l1", state_dir,
                            judge=StubJudge(course))
    reply = server.start()
    reply = server.handle("my name is Ravi")

Who decides what (non-negotiable):
    server  stage, time budget, hint level, item achieved, next item, whether to
            correct, session end
    judge   verdict, which known error applies, the short reply, the correction text
"""
