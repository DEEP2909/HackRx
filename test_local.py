import asyncio
import httpx
import json

async def test_api():
    """Test the API locally"""
    
    # API endpoint
    url = "http://localhost:8000/api/v1/hackrx/run"
    
    # Headers
    headers = {
        "Authorization": "Bearer 283be22af070a69d1a1f2913035eeb991e034a5aecc249ccae684d3f5cd3cc59",
        "Content-Type": "application/json"
    }
    
    # Request body
    data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "Does this policy cover maternity expenses, and what are the conditions?"
        ]
    }
    
    print("Testing LLM Query Retrieval System...")
    print(f"URL: {url}")
    print(f"Request: {json.dumps(data, indent=2)}")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, headers=headers, json=data)
            
            print(f"\nStatus Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(test_api())
