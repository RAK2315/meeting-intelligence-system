import requests, os

url = f"https://api.notion.com/v1/databases/{os.getenv('NOTION_DATABASE_ID')}"
headers = {
    "Authorization": f"Bearer {os.getenv('NOTION_TOKEN')}",
    "Notion-Version": "2022-06-28"
}

res = requests.get(url, headers=headers)
print(res.json())