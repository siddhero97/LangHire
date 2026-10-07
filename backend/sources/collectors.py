import httpx
import json
import urllib.parse
from datetime import datetime, timezone
import asyncio

async def collect_with_apify(api_key: str, title: str, location: str, max_jobs: int = 10) -> list[dict]:
    """Use Apify's LinkedIn Jobs Scraper actor."""
    if not api_key:
        raise ValueError("Apify API key is required")
        
    actor_id = "curious_coder/linkedin-jobs-scraper"
    run_url = f"https://api.apify.com/v2/acts/{actor_id}/runs?token={api_key}"
    
    input_data = {
        "queries": [f"{title} {location}".strip()],
        "maxItems": max_jobs if max_jobs > 0 else 50,
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Start the run
        res = await client.post(run_url, json=input_data)
        res.raise_for_status()
        run_data = res.json()["data"]
        run_id = run_data["id"]
        dataset_id = run_data["defaultDatasetId"]
        
        # Poll for completion
        while True:
            await asyncio.sleep(5)
            status_res = await client.get(f"https://api.apify.com/v2/actor-runs/{run_id}?token={api_key}")
            status = status_res.json()["data"]["status"]
            if status in ["SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"]:
                break
                
        # Fetch results
        items_res = await client.get(f"https://api.apify.com/v2/datasets/{dataset_id}/items?token={api_key}")
        items_res.raise_for_status()
        items = items_res.json()
        
        jobs = []
        for item in items:
            url = item.get("url", "")
            if url:
                jobs.append({
                    "title": item.get("title", ""),
                    "company": item.get("companyName", ""),
                    "location": item.get("location", ""),
                    "url": url,
                    "easy_apply": item.get("easyApply", False),
                    "description": item.get("description", ""),
                })
        return jobs


async def collect_with_firecrawl(api_key: str, board_url: str) -> list[dict]:
    """Use Firecrawl to extract jobs from a company's career page."""
    if not api_key:
        raise ValueError("Firecrawl API key is required")
        
    extract_url = "https://api.firecrawl.dev/v1/extract"
    prompt = "Extract a list of all open job positions. Include title, location, and the direct link to the application."
    schema = {
        "type": "object",
        "properties": {
            "jobs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "location": {"type": "string"},
                        "url": {"type": "string"}
                    },
                    "required": ["title", "url"]
                }
            }
        }
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "urls": [board_url],
        "prompt": prompt,
        "schema": schema,
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        res = await client.post(extract_url, headers=headers, json=payload)
        res.raise_for_status()
        data = res.json()
        
        jobs = []
        if data.get("success") and data.get("data"):
            extracted = data["data"].get("jobs", []) if isinstance(data["data"], dict) else data["data"][0].get("jobs", [])
            for item in extracted:
                if item.get("url"):
                    # Basic mapping
                    jobs.append({
                        "title": item.get("title", ""),
                        "company": urllib.parse.urlparse(board_url).netloc, # generic fallback
                        "location": item.get("location", ""),
                        "url": item.get("url", ""),
                        "easy_apply": False,
                    })
        return jobs


async def collect_with_greenhouse(board_token: str) -> list[dict]:
    """Collect directly from a Greenhouse public board API."""
    url = f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs"
    async with httpx.AsyncClient(timeout=15.0) as client:
        res = await client.get(url)
        if res.status_code != 200:
            return []
            
        data = res.json()
        jobs = []
        for job in data.get("jobs", []):
            jobs.append({
                "title": job.get("title", ""),
                "company": board_token.title(),
                "location": job.get("location", {}).get("name", ""),
                "url": job.get("absolute_url", ""),
                "easy_apply": False,
            })
        return jobs
