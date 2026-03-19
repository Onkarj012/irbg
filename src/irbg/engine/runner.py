from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from irbg.config import get_model_config
from irbg.db.operations import (
    DbConfig,
    connect,
    create_benchmark_run,
    insert_response,
    mark_benchmark_run_completed,
    mark_benchmark_run_failed,
    upsert_model,
    upsert_scenario,
    upsert_scenario_record,
)
from irbg.engine.provider import OpenRouterClient
from irbg.engine.types import ChatMessage
from irbg.engine.variant_generator import (
    generate_prompts_for_template,
    generate_single_prompt_for_variant,
)
from irbg.scenarios.discovery import load_template_files
from irbg.scenarios.loader import load_scenario
from irbg.scenarios.template_loader import load_scenario_template
from irbg.scenarios.template_models import RenderedPrompt


@dataclass(frozen=True)
class RunOnceResult:
    run_id: str
    response_id: str
    model_alias: str
    scenario_id: str
    success: bool
    error: str | None = None


@dataclass(frozen=True)
class RunBatchResult:
    run_id: str
    model_alias: str
    scenario_id: str
    mode: str
    total_count: int
    success_count: int
    failure_count: int


@dataclass(frozen=True)
class RunFolderResult:
    run_id: str
    model_alias: str
    folder_path: str
    mode: str
    scenario_count: int
    total_prompt_count: int
    success_count: int
    failure_count: int


def run_single_scenario(
    *,
    model_alias: str,
    scenario_file: Path,
    db_path: Path,
    mode: str = "baseline",
) -> RunOnceResult:
    model = get_model_config(model_alias)
    scenario = load_scenario(scenario_file)

    client = _build_client_from_env()
    conn = connect(DbConfig(path=db_path))

    try:
        upsert_model(
            conn,
            id=model.alias,
            name=model.name,
            provider=model.provider,
            model_id=model.model_id,
        )

        upsert_scenario(conn, scenario=scenario)

        config_snapshot = json.dumps(
            {
                "model_alias": model.alias,
                "provider_model_id": model.model_id,
                "scenario_file": str(scenario_file),
                "mode": mode,
            }
        )

        run_id = create_benchmark_run(
            conn,
            model_id=model.alias,
            mode=mode,
            status="running",
            config_snapshot=config_snapshot,
        )

        provider_response = client.chat(
            model_id=model.model_id,
            system_prompt=scenario.system_prompt,
            user_prompt=scenario.user_prompt,
            temperature=model.temperature,
            max_tokens=model.max_tokens,
        )

        response_id = insert_response(
            conn,
            run_id=run_id,
            scenario_id=scenario.id,
            variant_id=None,
            mode=mode,
            turn_number=1,
            system_prompt_sent=scenario.system_prompt,
            user_prompt_sent=scenario.user_prompt,
            raw_response=provider_response.text
            if provider_response.success
            else None,
            response_tokens=provider_response.total_tokens,
            latency_ms=provider_response.latency_ms,
        )

        if provider_response.success:
            mark_benchmark_run_completed(conn, run_id=run_id)
        else:
            mark_benchmark_run_failed(conn, run_id=run_id)

        return RunOnceResult(
            run_id=run_id,
            response_id=response_id,
            model_alias=model.alias,
            scenario_id=scenario.id,
            success=provider_response.success,
            error=provider_response.error,
        )
    finally:
        client.close()
        conn.close()


def run_single_template_variant(
    *,
    model_alias: str,
    scenario_file: Path,
    variant_id: str,
    db_path: Path,
    mode: str = "baseline",
) -> RunOnceResult:
    model = get_model_config(model_alias)
    template = load_scenario_template(scenario_file)
    rendered = generate_single_prompt_for_variant(
        template,
        variant_id=variant_id,
        mode=mode,
    )

    client = _build_client_from_env()
    conn = connect(DbConfig(path=db_path))

    try:
        upsert_model(
            conn,
            id=model.alias,
            name=model.name,
            provider=model.provider,
            model_id=model.model_id,
        )

        upsert_scenario_record(
            conn,
            id=template.id,
            pillar=template.pillar,
            category=template.category,
            jurisdiction=template.jurisdiction,
            difficulty=template.difficulty,
        )

        config_snapshot = json.dumps(
            {
                "model_alias": model.alias,
                "provider_model_id": model.model_id,
                "scenario_file": str(scenario_file),
                "mode": mode,
                "variant_id": variant_id,
            }
        )

        run_id = create_benchmark_run(
            conn,
            model_id=model.alias,
            mode=mode,
            status="running",
            config_snapshot=config_snapshot,
        )

        if mode == "adversarial" and template.adversarial_turns:
            response_ids, success, error = _execute_adversarial_sequence(
                client=client,
                conn=conn,
                run_id=run_id,
                model_id=model.model_id,
                rendered=rendered,
                adversarial_turns=template.adversarial_turns,
                temperature=model.temperature,
                max_tokens=model.max_tokens,
            )
            response_id = response_ids[-1]
        else:
            provider_response = client.chat(
                model_id=model.model_id,
                system_prompt=rendered.system_prompt,
                user_prompt=rendered.user_prompt,
                temperature=model.temperature,
                max_tokens=model.max_tokens,
            )

            response_id = insert_response(
                conn,
                run_id=run_id,
                scenario_id=template.id,
                variant_id=rendered.variant_id,
                mode=mode,
                turn_number=1,
                system_prompt_sent=rendered.system_prompt,
                user_prompt_sent=rendered.user_prompt,
                raw_response=provider_response.text
                if provider_response.success
                else None,
                response_tokens=provider_response.total_tokens,
                latency_ms=provider_response.latency_ms,
            )
            success = provider_response.success
            error = provider_response.error

        if success:
            mark_benchmark_run_completed(conn, run_id=run_id)
        else:
            mark_benchmark_run_failed(conn, run_id=run_id)

        return RunOnceResult(
            run_id=run_id,
            response_id=response_id,
            model_alias=model.alias,
            scenario_id=template.id,
            success=success,
            error=error,
        )
    finally:
        client.close()
        conn.close()


def run_all_template_variants(
    *,
    model_alias: str,
    scenario_file: Path,
    db_path: Path,
    mode: str = "baseline",
) -> RunBatchResult:
    model = get_model_config(model_alias)
    template = load_scenario_template(scenario_file)
    rendered_prompts = generate_prompts_for_template(
        template,
        mode=mode if mode != "adversarial" else "baseline",
    )

    client = _build_client_from_env()
    conn = connect(DbConfig(path=db_path))

    success_count = 0
    failure_count = 0

    try:
        upsert_model(
            conn,
            id=model.alias,
            name=model.name,
            provider=model.provider,
            model_id=model.model_id,
        )

        upsert_scenario_record(
            conn,
            id=template.id,
            pillar=template.pillar,
            category=template.category,
            jurisdiction=template.jurisdiction,
            difficulty=template.difficulty,
        )

        config_snapshot = json.dumps(
            {
                "model_alias": model.alias,
                "provider_model_id": model.model_id,
                "scenario_file": str(scenario_file),
                "mode": mode,
                "variant_group": template.variant_group,
                "variant_count": len(rendered_prompts),
            }
        )

        run_id = create_benchmark_run(
            conn,
            model_id=model.alias,
            mode=mode,
            status="running",
            config_snapshot=config_snapshot,
        )

        for rendered in rendered_prompts:
            if mode == "adversarial" and template.adversarial_turns:
                _, success, _ = _execute_adversarial_sequence(
                    client=client,
                    conn=conn,
                    run_id=run_id,
                    model_id=model.model_id,
                    rendered=rendered,
                    adversarial_turns=template.adversarial_turns,
                    temperature=model.temperature,
                    max_tokens=model.max_tokens,
                )
            else:
                provider_response = _execute_rendered_prompt(
                    client=client,
                    conn=conn,
                    run_id=run_id,
                    model_id=model.model_id,
                    rendered=rendered,
                    temperature=model.temperature,
                    max_tokens=model.max_tokens,
                )
                success = provider_response.success

            if success:
                success_count += 1
            else:
                failure_count += 1

        if failure_count == 0:
            mark_benchmark_run_completed(conn, run_id=run_id)
        else:
            mark_benchmark_run_failed(conn, run_id=run_id)

        return RunBatchResult(
            run_id=run_id,
            model_alias=model.alias,
            scenario_id=template.id,
            mode=mode,
            total_count=len(rendered_prompts),
            success_count=success_count,
            failure_count=failure_count,
        )
    finally:
        client.close()
        conn.close()


def run_template_folder(
    *,
    model_alias: str,
    folder_path: Path,
    db_path: Path,
    mode: str = "baseline",
) -> RunFolderResult:
    model = get_model_config(model_alias)
    scenario_files = load_template_files(folder_path)

    client = _build_client_from_env()
    conn = connect(DbConfig(path=db_path))

    success_count = 0
    failure_count = 0
    total_prompt_count = 0

    try:
        upsert_model(
            conn,
            id=model.alias,
            name=model.name,
            provider=model.provider,
            model_id=model.model_id,
        )

        config_snapshot = json.dumps(
            {
                "model_alias": model.alias,
                "provider_model_id": model.model_id,
                "folder_path": str(folder_path),
                "mode": mode,
                "scenario_file_count": len(scenario_files),
            }
        )

        run_id = create_benchmark_run(
            conn,
            model_id=model.alias,
            mode=mode,
            status="running",
            config_snapshot=config_snapshot,
        )

        for scenario_file in scenario_files:
            template = load_scenario_template(scenario_file)

            upsert_scenario_record(
                conn,
                id=template.id,
                pillar=template.pillar,
                category=template.category,
                jurisdiction=template.jurisdiction,
                difficulty=template.difficulty,
            )

            rendered_prompts = generate_prompts_for_template(
                template,
                mode=mode if mode != "adversarial" else "baseline",
            )

            total_prompt_count += len(rendered_prompts)

            for rendered in rendered_prompts:
                if mode == "adversarial" and template.adversarial_turns:
                    _, success, _ = _execute_adversarial_sequence(
                        client=client,
                        conn=conn,
                        run_id=run_id,
                        model_id=model.model_id,
                        rendered=rendered,
                        adversarial_turns=template.adversarial_turns,
                        temperature=model.temperature,
                        max_tokens=model.max_tokens,
                    )
                else:
                    provider_response = _execute_rendered_prompt(
                        client=client,
                        conn=conn,
                        run_id=run_id,
                        model_id=model.model_id,
                        rendered=rendered,
                        temperature=model.temperature,
                        max_tokens=model.max_tokens,
                    )
                    success = provider_response.success

                if success:
                    success_count += 1
                else:
                    failure_count += 1

        if failure_count == 0:
            mark_benchmark_run_completed(conn, run_id=run_id)
        else:
            mark_benchmark_run_failed(conn, run_id=run_id)

        return RunFolderResult(
            run_id=run_id,
            model_alias=model.alias,
            folder_path=str(folder_path),
            mode=mode,
            scenario_count=len(scenario_files),
            total_prompt_count=total_prompt_count,
            success_count=success_count,
            failure_count=failure_count,
        )
    finally:
        client.close()
        conn.close()


def _execute_rendered_prompt(
    *,
    client: OpenRouterClient,
    conn,
    run_id: str,
    model_id: str,
    rendered: RenderedPrompt,
    temperature: float,
    max_tokens: int,
):
    provider_response = client.chat(
        model_id=model_id,
        system_prompt=rendered.system_prompt,
        user_prompt=rendered.user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    insert_response(
        conn,
        run_id=run_id,
        scenario_id=rendered.scenario_id,
        variant_id=rendered.variant_id,
        mode=rendered.mode,
        turn_number=1,
        system_prompt_sent=rendered.system_prompt,
        user_prompt_sent=rendered.user_prompt,
        raw_response=provider_response.text
        if provider_response.success
        else None,
        response_tokens=provider_response.total_tokens,
        latency_ms=provider_response.latency_ms,
    )

    return provider_response


def _execute_adversarial_sequence(
    *,
    client: OpenRouterClient,
    conn,
    run_id: str,
    model_id: str,
    rendered: RenderedPrompt,
    adversarial_turns,
    temperature: float,
    max_tokens: int,
) -> tuple[list[str], bool, str | None]:
    messages = [
        ChatMessage(role="system", content=rendered.system_prompt),
        ChatMessage(role="user", content=rendered.user_prompt),
    ]

    response_ids: list[str] = []

    first_response = client.chat_messages(
        model_id=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    response_ids.append(
        insert_response(
            conn,
            run_id=run_id,
            scenario_id=rendered.scenario_id,
            variant_id=rendered.variant_id,
            mode="adversarial",
            turn_number=1,
            system_prompt_sent=rendered.system_prompt,
            user_prompt_sent=rendered.user_prompt,
            raw_response=first_response.text
            if first_response.success
            else None,
            response_tokens=first_response.total_tokens,
            latency_ms=first_response.latency_ms,
        )
    )

    if not first_response.success:
        return response_ids, False, first_response.error

    messages.append(ChatMessage(role="assistant", content=first_response.text))

    turn_number = 2
    for turn in adversarial_turns:
        messages.append(ChatMessage(role="user", content=turn.user_prompt))

        response = client.chat_messages(
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        response_ids.append(
            insert_response(
                conn,
                run_id=run_id,
                scenario_id=rendered.scenario_id,
                variant_id=rendered.variant_id,
                mode="adversarial",
                turn_number=turn_number,
                system_prompt_sent=rendered.system_prompt,
                user_prompt_sent=turn.user_prompt,
                raw_response=response.text if response.success else None,
                response_tokens=response.total_tokens,
                latency_ms=response.latency_ms,
            )
        )

        if not response.success:
            return response_ids, False, response.error

        messages.append(ChatMessage(role="assistant", content=response.text))
        turn_number += 1

    return response_ids, True, None


def _build_client_from_env() -> OpenRouterClient:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file."
        )

    base_url = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )
    app_name = os.getenv("OPENROUTER_APP_NAME", "IRBG")
    site_url = os.getenv("OPENROUTER_SITE_URL")

    return OpenRouterClient(
        api_key=api_key,
        base_url=base_url,
        app_name=app_name,
        site_url=site_url,
    )
