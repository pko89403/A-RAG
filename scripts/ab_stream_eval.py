from __future__ import annotations

import argparse
import csv
import json
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


@dataclass
class CaseResult:
    index: int
    question: str
    conversation_id: str
    request_id: str
    status: str
    elapsed_sec: float
    tool_used: bool
    tool_start_count: int
    tools_used: list[str]
    tool_call_sequence: list[str]
    tool_call_counts: dict[str, int]
    reference_event_count: int
    reference_item_count: int
    skill_used: bool
    skill_paths: list[str]
    done_reference_ids: list[str]
    done_reference_count: int
    done_citation_count: int
    done_content_length: int
    done_content: str
    request_scope_probe: dict[str, Any] | None
    error_message: str


def _load_questions(path: Path, limit: int | None = None) -> list[str]:
    questions: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
            question = row.get("question")
            if not isinstance(question, str) or not question.strip():
                continue
            questions.append(question.strip())
            if limit is not None and len(questions) >= limit:
                break
    return questions


def _parse_sse_stream(resp: httpx.Response) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event_name = "message"
    data_lines: list[str] = []

    for raw_line in resp.iter_lines():
        line = raw_line.strip()
        if not line:
            if data_lines:
                data_str = "\n".join(data_lines)
                try:
                    data_obj = json.loads(data_str)
                except json.JSONDecodeError:
                    data_obj = {"raw": data_str}
                events.append({"event": event_name, "data": data_obj})
            event_name = "message"
            data_lines = []
            continue

        if line.startswith("event:"):
            event_name = line[6:].strip() or "message"
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())
            continue

    # flush tail block (if stream ended without trailing blank line)
    if data_lines:
        data_str = "\n".join(data_lines)
        try:
            data_obj = json.loads(data_str)
        except json.JSONDecodeError:
            data_obj = {"raw": data_str}
        events.append({"event": event_name, "data": data_obj})

    return events


def _run_one_case(
    *,
    client: httpx.Client,
    api_base: str,
    endpoint: str,
    question: str,
    conversation_id: str,
    request_id: str | None = None,
) -> CaseResult:
    started_at = time.perf_counter()
    url = f"{api_base.rstrip('/')}{endpoint}"
    request_id = request_id or str(uuid.uuid4())
    payload = {
        "query": question,
        "conversationId": conversation_id,
        "configurable": {"requestId": request_id},
    }

    tool_names: list[str] = []
    reference_event_count = 0
    reference_item_count = 0
    skill_paths: list[str] = []
    done_reference_ids: list[str] = []
    done_reference_count = 0
    done_citation_count = 0
    done_content_length = 0
    done_content = ""
    error_message = ""
    status = "ok"

    try:
        with client.stream(
            "POST",
            url,
            headers={
                "accept": "text/event-stream",
                "content-type": "application/json",
            },
            json=payload,
        ) as resp:
            resp.raise_for_status()
            events = _parse_sse_stream(resp)
    except Exception as e:  # noqa: BLE001
        elapsed = time.perf_counter() - started_at
        return CaseResult(
            index=-1,
            question=question,
            conversation_id=conversation_id,
            request_id=request_id,
            status="http_error",
            elapsed_sec=elapsed,
            tool_used=False,
            tool_start_count=0,
            tools_used=[],
            tool_call_sequence=[],
            tool_call_counts={},
            reference_event_count=0,
            reference_item_count=0,
            skill_used=False,
            skill_paths=[],
            done_reference_ids=[],
            done_reference_count=0,
            done_citation_count=0,
            done_content_length=0,
            done_content="",
            request_scope_probe=None,
            error_message=str(e),
        )

    for ev in events:
        event = ev.get("event")
        data = ev.get("data")
        if not isinstance(data, dict):
            continue

        if event == "update":
            inner = data.get("event")
            name = data.get("name")
            if inner == "on_tool_start" and isinstance(name, str) and name:
                tool_names.append(name)
                if name == "read_file":
                    tool_input = data.get("toolInput")
                    if isinstance(tool_input, dict):
                        file_path = tool_input.get("file_path") or tool_input.get(
                            "path"
                        )
                        if (
                            isinstance(file_path, str)
                            and "/skills/" in file_path
                            and file_path.endswith("SKILL.md")
                        ):
                            if file_path not in skill_paths:
                                skill_paths.append(file_path)
        elif event == "reference":
            reference_event_count += 1
            refs = data.get("references")
            if isinstance(refs, list):
                reference_item_count += len(refs)
        elif event == "done":
            refs = data.get("references")
            cits = data.get("citations")
            content = data.get("content")
            if isinstance(refs, list):
                out_ids: list[str] = []
                for ref in refs:
                    if not isinstance(ref, dict):
                        continue
                    doc_id = str(ref.get("id") or "").strip()
                    if doc_id and doc_id not in out_ids:
                        out_ids.append(doc_id)
                done_reference_ids = out_ids
            done_reference_count = len(refs) if isinstance(refs, list) else 0
            done_citation_count = len(cits) if isinstance(cits, list) else 0
            done_content_length = len(content) if isinstance(content, str) else 0
            done_content = content if isinstance(content, str) else ""
        elif event == "error":
            status = "stream_error"
            msg = data.get("message")
            error_message = str(msg) if msg is not None else "unknown stream error"

    elapsed = time.perf_counter() - started_at
    tool_used = len(tool_names) > 0
    unique_tools = sorted(set(tool_names))
    tool_call_counts = dict(Counter(tool_names))
    skill_used = len(skill_paths) > 0

    return CaseResult(
        index=-1,
        question=question,
        conversation_id=conversation_id,
        request_id=request_id,
        status=status,
        elapsed_sec=elapsed,
        tool_used=tool_used,
        tool_start_count=len(tool_names),
        tools_used=unique_tools,
        tool_call_sequence=tool_names,
        tool_call_counts=tool_call_counts,
        reference_event_count=reference_event_count,
        reference_item_count=reference_item_count,
        skill_used=skill_used,
        skill_paths=skill_paths,
        done_reference_ids=done_reference_ids,
        done_reference_count=done_reference_count,
        done_citation_count=done_citation_count,
        done_content_length=done_content_length,
        done_content=done_content,
        request_scope_probe=None,
        error_message=error_message,
    )


def _call_chunk_read_probe(
    *,
    client: httpx.Client,
    api_base: str,
    endpoint: str,
    conversation_id: str,
    request_id: str,
    ids: list[str],
) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}{endpoint}"
    payload = {
        "ids": ids,
        "adjacentWindow": 0,
        "conversationId": conversation_id,
        "requestId": request_id,
    }
    resp = client.post(
        url,
        headers={
            "accept": "application/json",
            "content-type": "application/json",
        },
        json=payload,
    )
    resp.raise_for_status()
    body = resp.json()
    results = body.get("results")
    if not isinstance(results, list):
        results = []

    already_read_count = 0
    found_count = 0
    for item in results:
        if not isinstance(item, dict):
            continue
        if "metadata" in item:
            found_count += 1
        if item.get("already_read") is True:
            already_read_count += 1

    return {
        "request_id": request_id,
        "already_read_count": already_read_count,
        "found_count": found_count,
        "result_count": len(results),
        "notice": body.get("notice"),
    }


def _probe_request_scoped_chunk_cache(
    *,
    client: httpx.Client,
    api_base: str,
    probe_endpoint: str,
    conversation_id: str,
    ids: list[str],
) -> dict[str, Any]:
    base = uuid.uuid4().hex
    req_a = f"probe-{base}-a"
    req_b = f"probe-{base}-b"

    first = _call_chunk_read_probe(
        client=client,
        api_base=api_base,
        endpoint=probe_endpoint,
        conversation_id=conversation_id,
        request_id=req_a,
        ids=ids,
    )
    second_same_req = _call_chunk_read_probe(
        client=client,
        api_base=api_base,
        endpoint=probe_endpoint,
        conversation_id=conversation_id,
        request_id=req_a,
        ids=ids,
    )
    third_new_req = _call_chunk_read_probe(
        client=client,
        api_base=api_base,
        endpoint=probe_endpoint,
        conversation_id=conversation_id,
        request_id=req_b,
        ids=ids,
    )

    if first["found_count"] == 0:
        return {
            "checked_ids": ids,
            "first": first,
            "second_same_request_id": second_same_req,
            "third_new_request_id": third_new_req,
            "pass": None,
            "skipped": True,
            "reason": "no_probe_ids_found_in_chunk_read",
            "rule": "same request_id -> already_read increases, new request_id -> already_read resets",
        }

    passed = (
        second_same_req["already_read_count"] > 0
        and third_new_req["already_read_count"] == 0
    )

    return {
        "checked_ids": ids,
        "first": first,
        "second_same_request_id": second_same_req,
        "third_new_request_id": third_new_req,
        "pass": passed,
        "skipped": False,
        "rule": "same request_id -> already_read increases, new request_id -> already_read resets",
    }


def _summarize(results: list[CaseResult]) -> dict[str, Any]:
    total = len(results)
    ok = sum(1 for r in results if r.status == "ok")
    errors = total - ok
    tool_used_cases = sum(1 for r in results if r.tool_used)
    skill_used_cases = sum(1 for r in results if r.skill_used)
    total_refs = sum(r.done_reference_count for r in results)
    total_cits = sum(r.done_citation_count for r in results)
    probe_collected = sum(1 for r in results if isinstance(r.request_scope_probe, dict))
    probe_evaluable = sum(
        1
        for r in results
        if isinstance(r.request_scope_probe, dict)
        and r.request_scope_probe.get("pass") is not None
    )
    probe_passed = sum(
        1
        for r in results
        if isinstance(r.request_scope_probe, dict)
        and r.request_scope_probe.get("pass") is True
    )
    return {
        "total_cases": total,
        "ok_cases": ok,
        "error_cases": errors,
        "tool_used_cases": tool_used_cases,
        "tool_used_rate": (tool_used_cases / total) if total else 0.0,
        "skill_used_cases": skill_used_cases,
        "skill_used_rate": (skill_used_cases / total) if total else 0.0,
        "total_done_references": total_refs,
        "total_done_citations": total_cits,
        "avg_done_references": (total_refs / total) if total else 0.0,
        "avg_done_citations": (total_cits / total) if total else 0.0,
        "avg_elapsed_sec": (sum(r.elapsed_sec for r in results) / total)
        if total
        else 0.0,
        "request_scope_probe_collected": probe_collected,
        "request_scope_probe_evaluable": probe_evaluable,
        "request_scope_probe_passed": probe_passed,
        "request_scope_probe_pass_rate": (probe_passed / probe_evaluable)
        if probe_evaluable
        else 0.0,
    }


def _write_csv(path: Path, results: list[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "status",
                "elapsed_sec",
                "conversation_id",
                "request_id",
                "tool_used",
                "tool_start_count",
                "tools_used",
                "tool_call_sequence",
                "tool_call_counts",
                "reference_event_count",
                "reference_item_count",
                "skill_used",
                "skill_paths",
                "done_reference_ids",
                "done_reference_count",
                "done_citation_count",
                "done_content_length",
                "done_content",
                "request_scope_probe_pass",
                "request_scope_probe",
                "error_message",
                "question",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "index": r.index,
                    "status": r.status,
                    "elapsed_sec": f"{r.elapsed_sec:.3f}",
                    "conversation_id": r.conversation_id,
                    "request_id": r.request_id,
                    "tool_used": r.tool_used,
                    "tool_start_count": r.tool_start_count,
                    "tools_used": ",".join(r.tools_used),
                    "tool_call_sequence": " > ".join(r.tool_call_sequence),
                    "tool_call_counts": json.dumps(
                        r.tool_call_counts, ensure_ascii=False
                    ),
                    "reference_event_count": r.reference_event_count,
                    "reference_item_count": r.reference_item_count,
                    "skill_used": r.skill_used,
                    "skill_paths": ",".join(r.skill_paths),
                    "done_reference_ids": ",".join(r.done_reference_ids),
                    "done_reference_count": r.done_reference_count,
                    "done_citation_count": r.done_citation_count,
                    "done_content_length": r.done_content_length,
                    "done_content": r.done_content,
                    "request_scope_probe_pass": (
                        r.request_scope_probe.get("pass")
                        if isinstance(r.request_scope_probe, dict)
                        else ""
                    ),
                    "request_scope_probe": (
                        json.dumps(r.request_scope_probe, ensure_ascii=False)
                        if isinstance(r.request_scope_probe, dict)
                        else ""
                    ),
                    "error_message": r.error_message,
                    "question": r.question,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B evaluation helper for research-paper-summary stream citations/references"
    )
    parser.add_argument(
        "--input",
        default="tasks_question_only.jsonl",
        help="Path to JSONL file containing {'question': ...}",
    )
    parser.add_argument(
        "--api-base", default="http://localhost:8000", help="API base URL"
    )
    parser.add_argument(
        "--endpoint",
        default="/agents/research-paper-summary/invoke/stream",
        help="Streaming endpoint path",
    )
    parser.add_argument("--label", default="ab", help="Run label (e.g., A or B)")
    parser.add_argument(
        "--limit", type=int, default=None, help="Max number of questions"
    )
    parser.add_argument(
        "--sleep-sec", type=float, default=0.0, help="Sleep between requests"
    )
    parser.add_argument(
        "--timeout-sec", type=float, default=180.0, help="HTTP timeout seconds"
    )
    parser.add_argument(
        "--conversation-prefix",
        default="ab-stream",
        help="Conversation ID prefix for each case",
    )
    parser.add_argument(
        "--output-dir",
        default="backend/ab_results",
        help="Directory where JSON/CSV result files are written",
    )
    parser.add_argument(
        "--verify-request-scope",
        action="store_true",
        help=(
            "For each successful case, probe chunk_read duplicate cache scope "
            "using same conversation_id with same/new requestId"
        ),
    )
    parser.add_argument(
        "--probe-endpoint",
        default="/tools/research-paper-summary/chunk-read",
        help="Chunk-read endpoint used for request_id scope probe",
    )
    parser.add_argument(
        "--probe-ids-per-case",
        type=int,
        default=1,
        help="How many reference IDs to use per request-scope probe",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    questions = _load_questions(input_path, limit=args.limit)
    if not questions:
        raise ValueError(f"No valid questions in {input_path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{args.label}_{ts}.json"
    out_csv = out_dir / f"{args.label}_{ts}.csv"

    results: list[CaseResult] = []
    timeout = httpx.Timeout(args.timeout_sec)
    with httpx.Client(timeout=timeout) as client:
        for idx, question in enumerate(questions, start=1):
            conversation_id = f"{args.conversation_prefix}-{args.label}-{idx:04d}"
            request_id = str(uuid.uuid4())
            case = _run_one_case(
                client=client,
                api_base=args.api_base,
                endpoint=args.endpoint,
                question=question,
                conversation_id=conversation_id,
                request_id=request_id,
            )
            case.index = idx

            if (
                args.verify_request_scope
                and case.status == "ok"
                and case.done_reference_ids
            ):
                probe_ids = case.done_reference_ids[: max(1, args.probe_ids_per_case)]
                try:
                    case.request_scope_probe = _probe_request_scoped_chunk_cache(
                        client=client,
                        api_base=args.api_base,
                        probe_endpoint=args.probe_endpoint,
                        conversation_id=case.conversation_id,
                        ids=probe_ids,
                    )
                except Exception as e:  # noqa: BLE001
                    case.request_scope_probe = {
                        "pass": False,
                        "error": str(e),
                        "checked_ids": probe_ids,
                    }

            results.append(case)
            print(
                f"[{idx}/{len(questions)}] status={case.status} "
                f"tools={case.tool_start_count} refs={case.done_reference_count} "
                f"cits={case.done_citation_count} q={question[:60]}"
            )
            if isinstance(case.request_scope_probe, dict):
                probe_pass = bool(case.request_scope_probe.get("pass"))
                print(
                    f"  └─ request_scope_probe pass={probe_pass} "
                    f"ids={len(case.request_scope_probe.get('checked_ids') or [])}"
                )
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

    summary = _summarize(results)
    payload = {
        "meta": {
            "label": args.label,
            "input": str(input_path),
            "api_base": args.api_base,
            "endpoint": args.endpoint,
            "verify_request_scope": args.verify_request_scope,
            "probe_endpoint": args.probe_endpoint,
            "probe_ids_per_case": args.probe_ids_per_case,
            "timestamp": ts,
        },
        "summary": summary,
        "results": [
            {
                "index": r.index,
                "question": r.question,
                "conversation_id": r.conversation_id,
                "request_id": r.request_id,
                "status": r.status,
                "elapsed_sec": round(r.elapsed_sec, 3),
                "tool_used": r.tool_used,
                "tool_start_count": r.tool_start_count,
                "tools_used": r.tools_used,
                "tool_call_sequence": r.tool_call_sequence,
                "tool_call_counts": r.tool_call_counts,
                "reference_event_count": r.reference_event_count,
                "reference_item_count": r.reference_item_count,
                "skill_used": r.skill_used,
                "skill_paths": r.skill_paths,
                "done_reference_ids": r.done_reference_ids,
                "done_reference_count": r.done_reference_count,
                "done_citation_count": r.done_citation_count,
                "done_content_length": r.done_content_length,
                "done_content": r.done_content,
                "request_scope_probe": r.request_scope_probe,
                "error_message": r.error_message,
            }
            for r in results
        ],
    }
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(out_csv, results)

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nJSON: {out_json}")
    print(f"CSV : {out_csv}")


if __name__ == "__main__":
    main()
