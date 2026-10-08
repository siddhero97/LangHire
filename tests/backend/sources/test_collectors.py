import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from backend.sources.collectors import (
    collect_with_apify,
    collect_with_firecrawl,
    collect_with_greenhouse,
)


@pytest.mark.asyncio
async def test_collect_with_apify_success(monkeypatch):
    api_key = "test_apify_key"
    title = "Software Engineer"
    location = "San Francisco"

    # Mock responses for run creation, status polling, and dataset items
    mock_run_resp = MagicMock()
    mock_run_resp.raise_for_status = MagicMock()
    mock_run_resp.json.return_value = {
        "data": {"id": "run_123", "defaultDatasetId": "dataset_123"}
    }

    mock_status_resp = MagicMock()
    mock_status_resp.raise_for_status = MagicMock()
    mock_status_resp.json.return_value = {"data": {"status": "SUCCEEDED"}}

    mock_items_resp = MagicMock()
    mock_items_resp.raise_for_status = MagicMock()
    mock_items_resp.json.return_value = [
        {
            "title": "Backend Engineer",
            "companyName": "Acme Corp",
            "location": "San Francisco, CA",
            "url": "https://www.linkedin.com/jobs/view/123456789/",
            "easyApply": True,
            "description": "Great python role",
        }
    ]

    async def mock_post(url, **kwargs):
        if "runs" in url:
            return mock_run_resp
        return MagicMock()

    async def mock_get(url, **kwargs):
        if "actor-runs" in url:
            return mock_status_resp
        if "datasets" in url:
            return mock_items_resp
        return MagicMock()

    with patch("httpx.AsyncClient.post", side_effect=mock_post), \
         patch("httpx.AsyncClient.get", side_effect=mock_get), \
         patch("backend.sources.collectors.asyncio.sleep", new_callable=AsyncMock):
        jobs = await collect_with_apify(api_key, title, location, max_jobs=5)

    assert len(jobs) == 1
    assert jobs[0]["title"] == "Backend Engineer"
    assert jobs[0]["company"] == "Acme Corp"
    assert jobs[0]["easy_apply"] is True


@pytest.mark.asyncio
async def test_collect_with_firecrawl_success(monkeypatch):
    api_key = "test_fc_key"
    board_url = "https://careers.example.com"

    mock_res = MagicMock()
    mock_res.raise_for_status = MagicMock()
    mock_res.json.return_value = {
        "success": True,
        "data": {
            "jobs": [
                {
                    "title": "Frontend Developer",
                    "location": "Remote",
                    "url": "https://careers.example.com/jobs/999",
                }
            ]
        },
    }

    async def mock_post(url, **kwargs):
        return mock_res

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        jobs = await collect_with_firecrawl(api_key, board_url)

    assert len(jobs) == 1
    assert jobs[0]["title"] == "Frontend Developer"
    assert jobs[0]["url"] == "https://careers.example.com/jobs/999"


@pytest.mark.asyncio
async def test_collect_with_greenhouse_success():
    board_token = "stripe"

    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {
        "jobs": [
            {
                "title": "Staff Engineer",
                "location": {"name": "Seattle, WA"},
                "absolute_url": "https://boards.greenhouse.io/stripe/jobs/112233",
            }
        ]
    }

    async def mock_get(url, **kwargs):
        return mock_res

    with patch("httpx.AsyncClient.get", side_effect=mock_get):
        jobs = await collect_with_greenhouse(board_token)

    assert len(jobs) == 1
    assert jobs[0]["title"] == "Staff Engineer"
    assert jobs[0]["company"] == "Stripe"
    assert jobs[0]["url"] == "https://boards.greenhouse.io/stripe/jobs/112233"
