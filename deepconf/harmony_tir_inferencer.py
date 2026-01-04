"""HarmonyTIRInferencer: Harmony protocol TIR inferencer with max_tool_calls, verifier and trace logging.

This file provides a drop-in enhanced version of HarmonyTIRInferencer based on the user's implementation. It adds per-sample tool call counting, a lightweight verifier step (non-streaming short LLM call), and JSONL trace logging to docs/ or tir_traces by default.

NOTE: This implementation tries to reuse existing objects (python_pool, PythonTool, encoding, Conversation, Message, Role, etc.) already used in the codebase. It uses self.client.completions.create for verifier calls in a non-streaming mode. Adjustments may be required depending on exact client output formats.
"""

from __future__ import annotations
import time
import threading
import queue
import json
import os
from pathlib import Path
from typing import List, Tuple, Any

# Reuse existing imports from the project where appropriate
# e.g., OpenAI client wrapper, encoding utilities, Conversation/Message types, PythonTool, etc.
# The code assumes these names (OpenAI, Conversation, Message, Role, PythonTool, python_pool, encoding) are available in the runtime.


class HarmonyTIRInferencer:
    """Inferencer using Harmony protocol with Tool-Integrated Reasoning (TIR).

    This variant augments the original class with:
    - per-sample max_tool_calls counting and enforcement
    - a lightweight verifier step after each tool call (non-streaming LLM check)
    - structured trace logging (JSON lines) written to a configurable directory

    The implementation is intentionally conservative: it does not change the overall
    message flow or the python_tool pool usage, only adds verification, counting and
    trace persistence to help debugging and safety.
    """

    def __init__(
        self,
        model_path: str,
        max_model_len: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.95,
        min_p: float = 0.0,
        seed: int = 42,
        k: int = 4,
        use_budget: bool = True,
        max_iter: int = 100,
        # TIR-related defaults
        max_tool_calls_per_sample: int = 10,
        tir_verifier_enabled: bool = True,
        tir_verifier_timeout: float = 5.0,
        trace_dir: str = "tir_traces",
    ):
        self.model_path = model_path
        self.model = "gpt-oss"
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.top_p = top_p
        self.min_p = min_p
        self.seed = seed
        self.k = k
        self.use_budget = use_budget
        self.max_iter = max_iter
        self.base_budget = 60 * 5.5
        self.budget = 370
        self.deadline = None

        # TIR configs
        self.max_tool_calls_per_sample = int(max_tool_calls_per_sample)
        self.tir_verifier_enabled = bool(tir_verifier_enabled)
        self.tir_verifier_timeout = float(tir_verifier_timeout)
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the OpenAI-compatible client pointing to local vLLM server
        # The user's code used OpenAI wrapper; we reuse it here.
        self.client = OpenAI(
            base_url="http://127.0.0.1:8000/v1",
            api_key="sk-local",
            timeout=360,
        )
        # Encoding utilities / tokenizer as in original code
        self.stop_token_ids = encoding.stop_tokens_for_assistant_actions()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # --- Helper methods ---
    def _run_verifier(self, tool_stdout: str) -> dict:
        """Run a short non-streaming verifier LLM request.

        The verifier must return a JSON-like answer such as {"ok": true} or {"ok": false, "reason": "..."}.
        This implementation is best-effort: it asks the model to output only JSON and attempts to parse it.
        """
        prompt = (
            "Verifier: You are a strict verifier. Given the tool output below, reply with exactly one JSON object"
            " that is either {\"ok\": true} or {\"ok\": false, \"reason\": \"...\"}. Do NOT add any other text.\n\n"
            "TOOL OUTPUT:\n" + (tool_stdout or "")
        )
        try:
            # Non-streaming call for small prompt
            resp = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=64,
                temperature=0.0,
                top_p=0.0,
                stream=False,
                timeout=self.tir_verifier_timeout,
            )
            # Extract text from response - best-effort depending on client shape
            if hasattr(resp, 'choices') and len(resp.choices) > 0:
                text = getattr(resp.choices[0], 'text', '') or ''
            else:
                # Some wrappers return the text directly
                text = getattr(resp, 'text', '') or str(resp)
            text = text.strip()
            # Try parse JSON
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                # attempt to extract a JSON substring
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1 and end > start:
                    candidate = text[start:end+1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        pass
            return {"ok": False, "reason": "verifier_parse_failed", "raw": text}
        except Exception as e:
            return {"ok": False, "reason": f"verifier_error: {e}"}

    def _write_trace(self, trace_entries: List[dict], trace_id: str) -> None:
        try:
            p = self.trace_dir / f"trace-{trace_id}.jsonl"
            with p.open("a", encoding="utf-8") as f:
                for e in trace_entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
        except Exception:
            # best-effort: do not raise
            pass

    # --- Existing behavior: wait_server, get_num_samples, apply_chat_template, format_prompts, inference ---
    # For brevity we reuse the user's existing implementations; they can be copied or imported as needed.

    def wait_server(self):
        for _ in range(15 * 60):
            time.sleep(1)
            try:
                print(self.client.models.list())
                return
            except Exception:
                continue
        raise RuntimeError("vLLM server failed to start")

    def get_num_samples(self) -> int:
        if not self.use_budget:
            return self.k
        else:
            return self.k

    def apply_chat_template(self, prompt: str, python_tool: Any) -> list:
        return [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new()
                .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
                .with_tools(python_tool.tool_config)
            ),
            Message.from_role_and_content(Role.USER, prompt),
        ]

    def format_prompts(self, problem: str) -> list:
        num_samples = self.get_num_samples()
        prompts = []
        for i in range(num_samples):
            tir_prompt = TIR_PROMPTS[i % len(TIR_PROMPTS)]
            prompts.append(problem + "\n\n" + tir_prompt)
        return prompts

    def inference(self, problem: str, deadline: float) -> Tuple[int, float]:
        self.deadline = deadline
        start_time = time.time()
        prompts = self.format_prompts(problem)
        responses = self._inference_parallel(prompts)
        duration = time.time() - start_time
        saved_time = max(0.0, deadline - time.time())
        return self.parse_responses(responses), saved_time

    # --- Modified single_generate_tir with counting, verifier, and trace logging ---
    def single_generate_tir(self, prompt: str, stop_event: threading.Event, seed_offset: int = 0) -> str:
        python_tool = None
        tool_calls = 0
        trace_entries: List[dict] = []
        trace_id = f"{int(time.time()*1000)}-{threading.get_ident()}"

        def _compute_req_timeout() -> float:
            CUSHION = 0.5
            MAX_REQ_TIMEOUT = 30.0
            MIN_ALLOW = 0.2
            if not getattr(self, "deadline", None):
                return MAX_REQ_TIMEOUT
            remaining = self.deadline - time.time()
            if remaining <= 0:
                return 0.0
            t = remaining - CUSHION
            if t <= 0:
                return 0.0
            return min(MAX_REQ_TIMEOUT, max(MIN_ALLOW, t))

        def _compute_py_timeout() -> float:
            PY_CUSHION = 1.0
            MAX_PY_TIMEOUT = 15.0
            MIN_ALLOW = 0.2
            if not getattr(self, "deadline", None):
                return MAX_PY_TIMEOUT
            remaining = self.deadline - time.time()
            t = remaining - PY_CUSHION
            if t <= 0:
                return 0.0
            return min(MAX_PY_TIMEOUT, max(MIN_ALLOW, t))

        try:
            try:
                python_tool = python_pool.get(timeout=30.0)
            except queue.Empty:
                python_tool = PythonTool(execution_backend="jupyter")
                try:
                    python_tool._ensure_session()
                except Exception as e:
                    if python_tool is not None:
                        try:
                            python_tool.close()
                        except Exception:
                            pass
                    return ""
            else:
                try:
                    if python_tool._jupyter_session is None:
                        python_tool._ensure_session()
                    test_output = python_tool._jupyter_session.execute("1+1", timeout=2.0)
                    if "[ERROR]" in test_output or "Traceback" in test_output:
                        try:
                            python_tool.close()
                        except Exception:
                            pass
                        python_tool._jupyter_session = None
                        python_tool._ensure_session()
                except Exception:
                    try:
                        python_tool.close()
                    except Exception:
                        pass
                    python_tool._jupyter_session = None
                    try:
                        python_tool._ensure_session()
                    except Exception:
                        try:
                            python_pool.put(python_tool, block=False)
                        except Exception:
                            pass
                        return ""

            messages = self.apply_chat_template(prompt, python_tool)
            final_answer_found = ""

            for iteration in range(self.max_iter):
                if stop_event and stop_event.is_set():
                    break
                if getattr(self, "deadline", None) and time.time() >= self.deadline:
                    break
                if final_answer_found:
                    break

                prompt_ids = encoding.render_conversation_for_completion(
                    Conversation.from_messages(messages), Role.ASSISTANT
                )
                max_tokens = self.max_model_len - len(prompt_ids)
                if max_tokens < 1:
                    break

                req_timeout = _compute_req_timeout()
                if req_timeout <= 0:
                    break

                token_buffer: list[int] = []
                token_buffer_str = ""
                breaking = False

                stream = None
                try:
                    stream = self.client.completions.create(
                        model=self.model,
                        prompt=prompt_ids,
                        max_tokens=max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        seed=self.seed + seed_offset,
                        stream=True,
                        extra_body=dict(
                            min_p=self.min_p,
                            stop_token_ids=self.stop_token_ids,
                            return_token_ids=True,
                        ),
                        timeout=req_timeout,
                    )

                    for chunk in stream:
                        if stop_event and stop_event.is_set():
                            breaking = True
                            break
                        if getattr(self, "deadline", None) and time.time() >= self.deadline:
                            breaking = True
                            break

                        if not chunk.choices or len(chunk.choices) == 0:
                            continue
                        choice = chunk.choices[0]
                        token_chunk = getattr(choice, 'token_ids', None) or []
                        text_chunk = getattr(choice, 'text', '') or ''

                        if token_chunk:
                            token_buffer.extend(token_chunk)
                            token_buffer_str += text_chunk

                        if len(token_buffer) > 60_000:
                            breaking = True
                            break

                        if "}" in text_chunk and self.extract_boxed_text(token_buffer_str) is not None:
                            final_answer_found = token_buffer_str
                            breaking = True
                            break

                except Exception:
                    breaking = True
                finally:
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception:
                            pass
                        try:
                            del stream
                        except Exception:
                            pass

                if breaking:
                    break

                if not token_buffer:
                    continue

                try:
                    new_messages = encoding.parse_messages_from_completion_tokens(
                        token_buffer, Role.ASSISTANT
                    )
                except Exception:
                    break

                messages.extend(new_messages)
                last_message = messages[-1]

                if last_message.channel == "final" or token_buffer[-1] == 200002:
                    break

                if last_message.recipient == "python":
                    # Enforce per-sample max tool calls
                    if tool_calls >= self.max_tool_calls_per_sample:
                        trace_entries.append({"ts": time.time(), "event": "max_tool_calls_exceeded", "tool_calls": tool_calls})
                        break

                    if stop_event and stop_event.is_set():
                        break
                    if getattr(self, "deadline", None) and time.time() >= self.deadline:
                        break

                    py_timeout = _compute_py_timeout()
                    if py_timeout <= 0 or py_timeout < 0.5:
                        trace_entries.append({"ts": time.time(), "event": "py_timeout_insufficient", "py_timeout": py_timeout})
                        break

                    # Count this tool call
                    tool_calls += 1

                    # Record tool call request
                    trace_entries.append({
                        "ts": time.time(),
                        "event": "tool_call_request",
                        "tool": "python",
                        "code": getattr(last_message, 'content', None),
                        "iteration": iteration,
                        "tool_call_index": tool_calls,
                    })

                    # Execute python code via python_tool
                    try:
                        response_msgs = python_tool.process_sync_plus(last_message, timeout=py_timeout)
                    except Exception as e:
                        trace_entries.append({"ts": time.time(), "event": "tool_error", "error": str(e)})
                        break

                    # Attempt to extract stdout/stderr from response messages
                    tool_stdout = ""
                    tool_stderr = ""
                    try:
                        for rm in response_msgs:
                            # best-effort: many Message-like objects have .content
                            txt = getattr(rm, 'content', None) or getattr(rm, 'text', None) or ''
                            tool_stdout += str(txt)
                    except Exception:
                        pass

                    trace_entries.append({
                        "ts": time.time(),
                        "event": "tool_call_result",
                        "stdout": tool_stdout,
                        "stderr": tool_stderr,
                        "iteration": iteration,
                        "tool_call_index": tool_calls,
                    })

                    # Verifier step (optional)
                    if self.tir_verifier_enabled:
                        ver = self._run_verifier(tool_stdout)
                        trace_entries.append({"ts": time.time(), "event": "verifier", "result": ver})
                        if not ver.get("ok", False):
                            # Ask planner (LLM) to reconsider / retry by appending a system message
                            messages.append(Message.from_role_and_content(Role.SYSTEM, f"Verifier failed: {ver.get('reason', '')}. Please retry or explain."))
                            # allow planner to produce a corrected action; do not immediately give up
                            continue

                    # If OK, append the tool response messages into the conversation and continue planning
                    messages.extend(response_msgs)

            if final_answer_found:
                return final_answer_found

            # Fallback: return rendered conversation for training as before
            return encoding.decode_utf8(
                encoding.render_conversation_for_training(
                    Conversation.from_messages(messages),
                    RenderConversationConfig(auto_drop_analysis=False),
                )
            )

        except KeyboardInterrupt:
            raise
        except Exception as e:
            return ""
        finally:
            # persist trace entries
            try:
                self._write_trace(trace_entries, trace_id)
            except Exception:
                pass

            # Return tool to pool instead of closing it
            if python_tool is not None:
                try:
                    if python_tool._jupyter_session is not None:
                        try:
                            test_output = python_tool._jupyter_session.execute("1+1", timeout=1.0)
                            if "[ERROR]" not in test_output and "Traceback" not in test_output:
                                try:
                                    python_pool.put(python_tool, block=False)
                                except queue.Full:
                                    python_tool.close()
                            else:
                                python_tool.close()
                        except Exception:
                            try:
                                python_tool.close()
                            except Exception:
                                pass
                    else:
                        try:
                            python_pool.put(python_tool, block=False)
                        except queue.Full:
                            pass
                except Exception:
                    try:
                        python_tool.close()
                    except Exception:
                        pass

    # --- parallel inference and parsing responses - reuse prior implementations ---
    def _inference_parallel(self, prompts: list) -> list:
        stop_event = threading.Event()
        answers_collected: List[int] = []
        raw_responses = [""] * len(prompts)
        majority_threshold = len(prompts) / 2

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from collections import Counter

        executor = ThreadPoolExecutor(max_workers=self.k)
        futures = []
        future_to_idx = {}
        try:
            for i, p in enumerate(prompts):
                fut = executor.submit(self.single_generate_tir, p, stop_event, i)
                futures.append(fut)
                future_to_idx[fut] = i

            completed_count = 0
            for fut in as_completed(futures):
                idx = future_to_idx.get(fut, -1)
                if idx < 0:
                    continue
                try:
                    result_text = fut.result(timeout=1.0)
                except Exception:
                    result_text = ""
                raw_responses[idx] = result_text
                completed_count += 1
                ans = self.extract_boxed_text(result_text)
                if ans is not None:
                    answers_collected.append(ans)
                    counts = Counter(answers_collected)
                    if counts:
                        most_common_ans, count = counts.most_common(1)[0]
                        if count > majority_threshold:
                            stop_event.set()
                            for f in futures:
                                if f is not fut and not f.done():
                                    try:
                                        f.cancel()
                                    except Exception:
                                        pass
                            break
        finally:
            stop_event.set()
            for fut in futures:
                if not fut.done():
                    try:
                        fut.cancel()
                    except Exception:
                        pass
            try:
                import sys
                if sys.version_info >= (3, 9):
                    executor.shutdown(wait=True, timeout=60.0, cancel_futures=True)
                else:
                    executor.shutdown(wait=True)
            except TypeError:
                try:
                    executor.shutdown(wait=True)
                except Exception:
                    executor.shutdown(wait=False)
            except Exception:
                try:
                    executor.shutdown(wait=False)
                except Exception:
                    pass

        return raw_responses

    def extract_boxed_text(self, text: str) -> int | None:
        import re
        pattern = r'oxed{(.*?)}'
        matches = re.findall(pattern, str(text))
        if matches:
            for match in reversed(matches):
                if match:
                    try:
                        clean_match = match.strip().replace(',', '').replace(' ', '')
                        val = int(float(clean_match[:20]))
                        if 0 <= val <= 99999:
                            return val
                    except Exception:
                        pass
        pattern = r'(?i)final\s+answer\s*(?:is|:)?\s*(\d+)'
        matches = re.findall(pattern, text)
        if matches:
            for match in reversed(matches):
                if match:
                    try:
                        val = int(match)
                        if 0 <= val <= 99999:
                            return val
                    except Exception:
                        pass
        return None

    def parse_responses(self, responses: list) -> int:
        from collections import Counter
        answers = [self.extract_boxed_text(r) for r in responses]
        valid_answers = [a for a in answers if a is not None]
        if not valid_answers:
            return 8687
        counter = Counter(valid_answers)
        most_common_list = counter.most_common(2)
        if len(most_common_list) > 1 and most_common_list[0][1] == most_common_list[1][1]:
            tied_answers = [ans for ans, cnt in counter.items() if cnt == most_common_list[0][1]]
            answer = max(tied_answers)
        else:
            answer = most_common_list[0][0]
        return answer
