"""Run a Hugging Face causal language model on PsychEthicsBench."""

from __future__ import annotations

import argparse
import json
import re
from string import Formatter
from pathlib import Path

from .validate_predictions import load_jsonl


def _template_fields(template: str) -> set[str]:
    return {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name is not None
    }


def _format_options(options: list[str]) -> str:
    rendered = []
    for index, text in enumerate(options):
        expected_letter = chr(ord("A") + index)
        match = re.match(r"^\s*([A-Z])[.)]\s+", text)
        if match and match.group(1) == expected_letter:
            rendered.append(text)
        else:
            option_text = text[match.end() :] if match else text
            rendered.append(f"{expected_letter}. {option_text}")
    return "\n".join(rendered)


def render_prompt(task: str, template: str, record: dict) -> str:
    fields = _template_fields(template)
    if task == "mcq":
        if "question" not in record or "options" not in record:
            raise ValueError("MCQ records must contain question and options")
        missing = {"question", "options"} - fields
        unsupported = fields - {"question", "options"}
        if missing or unsupported:
            raise ValueError(
                f"MCQ prompt must contain question and options placeholders; "
                f"missing={sorted(missing)}, unsupported={sorted(unsupported)}"
            )
        options = _format_options(record["options"])
        return template.format(question=record["question"], options=options)
    if task == "oeq":
        if "question" not in record:
            raise ValueError("OEQ records must contain question")
        if fields != {"question"}:
            raise ValueError(
                "OEQ prompt must contain only the question placeholder; "
                f"found={sorted(fields)} (options are MCQ-only)"
            )
        return template.format(question=record["question"])
    raise ValueError(f"unsupported task: {task}")


def parse_mcq_answer(text: str) -> list[str]:
    try:
        answer = json.loads(text.strip())
    except (json.JSONDecodeError, AttributeError) as exc:
        raise ValueError("MCQ output must be a JSON list of uppercase letters") from exc
    if not isinstance(answer, list) or any(
        not isinstance(letter, str)
        or len(letter) != 1
        or not letter.isascii()
        or not letter.isupper()
        for letter in answer
    ):
        raise ValueError("MCQ output must be a JSON list of uppercase letters")
    return answer


def parse_mcq_output(text: str, output_format: str) -> list[str]:
    if output_format == "list":
        return parse_mcq_answer(text)
    if output_format == "reasoning":
        try:
            structured = json.loads(text.strip())
            answer = structured["answer"]
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(
                "reasoning MCQ output must be a JSON object with an answer field"
            ) from exc
        return parse_mcq_answer(json.dumps(answer))
    raise ValueError(f"unsupported MCQ output format: {output_format}")


def _prepare_model_input(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def run(args: argparse.Namespace) -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "run_hf requires the optional packages listed in requirements.txt"
        ) from exc

    records = load_jsonl(args.data)
    template = args.prompt.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            prompts = [
                _prepare_model_input(tokenizer, render_prompt(args.task, template, record))
                for record in batch
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            if hasattr(inputs, "to"):
                inputs = inputs.to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_length = inputs["input_ids"].shape[1]
            texts = tokenizer.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True
            )
            for record, text in zip(batch, texts):
                text = text.strip()
                if args.task == "mcq":
                    try:
                        answer = parse_mcq_output(text, args.mcq_output_format)
                    except ValueError as exc:
                        raise ValueError(
                            f"invalid model output for {record['id']}: {exc}"
                        ) from exc
                    prediction = {"id": record["id"], "answer": answer}
                else:
                    prediction = {"id": record["id"], "response": text}
                output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                output.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("mcq", "oeq"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--mcq-output-format",
        choices=("list", "reasoning"),
        default="list",
        help="Expected MCQ output shape; use reasoning with mcq_reasoning prompts.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
