import requests

base_url = "http://localhost:8000"

submit_path = "/submit"
submit_params = {
    "code": "abc",
    "level": 0,
    "task": 1
}

res = requests.get(f"{base_url}{submit_path}", params=submit_params)

eval_id = res.text

poll_path = "/poll"
poll_params = {"id": eval_id}

res = requests.get(f"{base_url}{poll_path}", params=poll_params)

print(res.json())
