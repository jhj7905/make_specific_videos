#!/usr/bin/env python
"""make_videos CLI

  python cli.py templates                      # 템플릿 목록
  python cli.py inputs propose_neon_01         # 입력 슬롯 확인
  python cli.py render jobs/demo.json          # 렌더
  python cli.py render jobs/demo.json --aspect 16:9
  python cli.py render jobs/demo.json --aspect 16:9,9:16   # 가로+세로 한 번에
  python cli.py render jobs/demo.json --gpus 0,1,2,3 --no-preview
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import config
from engine.spec import list_templates, load_template, Job
from engine.render import render_job


def cmd_templates(_):
    for t in list_templates():
        req = sum(1 for i in t.inputs if i.required)
        print(f"{t.id:<24} {t.name:<14} {t.category:<8} "
              f"{t.total_duration:>5.1f}s  씬{len(t.scenes):>2}  필수입력 {req}")


def cmd_inputs(a):
    t = load_template(a.template)
    print(f"# {t.id} · {t.name} · {t.total_duration}초 · {t.width}x{t.height}")
    for i in t.inputs:
        mark = "필수" if i.required else "선택"
        extra = f" (기본: {i.default})" if i.default else ""
        print(f"  [{mark}] {i.id:<12} {i.type:<6} {i.guide}{extra}")
    print("\n# job 템플릿:")
    print(json.dumps({"template": t.id, "order_id": "TEST-0001",
                      "inputs": {i.id: i.default or f"<{i.type}>" for i in t.inputs}},
                     ensure_ascii=False, indent=2))


def cmd_demo(a):
    """템플릿 개발용: 폴더의 사진으로 이미지 슬롯을 자동으로 채운 job 을 만든다."""
    t = load_template(a.template)
    d = Path(a.photos)
    pics = sorted([p for p in d.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    vids = sorted([p for p in d.iterdir() if p.suffix.lower() in {".mp4", ".mov"}])
    if not pics:
        raise FileNotFoundError(f"{d} 에 사진이 없습니다")
    inputs, pi, vi = {}, 0, 0
    for spec in t.inputs:
        if spec.type == "image":
            inputs[spec.id] = str(pics[pi % len(pics)]); pi += 1
        elif spec.type == "video" and vids:
            inputs[spec.id] = str(vids[vi % len(vids)]); vi += 1
        elif spec.type == "text":
            inputs[spec.id] = spec.default
    out = Path(a.out or f"jobs/{t.id}_demo.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"template": t.id, "order_id": a.order_id, "preview": True, "inputs": inputs},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"작성: {out}  (사진 {pi}슬롯 / 영상 {vi}슬롯)")
    if a.render:
        res = render_job(Job.load(out))
        print(json.dumps(res, ensure_ascii=False, indent=2))


def cmd_render(a):
    job = Job.load(a.job)
    if a.no_preview:
        job.preview = False
    gpus = [int(x) for x in a.gpus.split(",")] if a.gpus else None
    aspects = [x.strip() for x in a.aspect.split(",")] if a.aspect else [None]
    out = []
    for asp in aspects:
        out.append(render_job(job, gpus=gpus, workers=a.workers, aspect=asp))
    print(json.dumps(out if len(out) > 1 else out[0], ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(prog="make_videos")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("templates").set_defaults(fn=cmd_templates)

    q = sub.add_parser("inputs"); q.add_argument("template")
    q.set_defaults(fn=cmd_inputs)

    d = sub.add_parser("demo", help="폴더의 사진으로 데모 job 생성 (템플릿 개발용)")
    d.add_argument("template")
    d.add_argument("--photos", default="storage/uploads/demo")
    d.add_argument("--out", default=None)
    d.add_argument("--order-id", default="DEMO")
    d.add_argument("--render", action="store_true")
    d.set_defaults(fn=cmd_demo)

    r = sub.add_parser("render")
    r.add_argument("job")
    r.add_argument("--gpus", default=None, help="예: 0,1,2,3")
    r.add_argument("--aspect", default=None,
                   help="출력 규격. 예: 16:9 / 9:16 / 1920x1080 / '16:9,9:16'(둘 다)")
    r.add_argument("--workers", type=int, default=None)
    r.add_argument("--no-preview", action="store_true")
    r.set_defaults(fn=cmd_render)

    a = p.parse_args()
    try:
        a.fn(a)
    except Exception as e:
        print(f"\n✖ {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
