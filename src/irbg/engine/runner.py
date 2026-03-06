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
)
from irbg.engine.provider import OpenRouterClient
from irbg.scenarios.loader import load_scenario


@dataclass(frozen=True)
class RunOnceResult:
    run_id: str
    response_id: str
    model_alias: str
    scenario_id: str
    success: bool
    error: str | None = None


def run_single_scenario(
    *,
    model_alias: str,
    scenario_file: Path,
    db_path: Path,
    mode: str = "baseline",
) -> RunOnceResult:
    model = get_model_config(model_alias)
    scenario = load_scenario(scenario_file)

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

    conn = connect(DbConfig(path=db_path))
    client = OpenRouterClient(
        api_key=api_key,
        base_url=base_url,
        app_name=app_name,
        site_url=site_url,
    )

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
