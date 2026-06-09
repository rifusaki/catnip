import requests
import json

LS_URL   = "https://label.rifusaki.com" 
API_KEY  = "..."  # your API key here
PROJECT  = 2
OLD      = "kabru"
NEW      = "other_face"

headers = {}
if API_KEY.count('.') == 2:
    # Label Studio >= 1.22.0 uses JWT refresh tokens for PAT
    r_token = requests.post(f"{LS_URL}/api/token/refresh/", json={"refresh": API_KEY})
    r_token.raise_for_status()
    access_token = r_token.json()["access"]
    headers["Authorization"] = f"Bearer {access_token}"
else:
    # just in case
    headers["Authorization"] = f"Token {API_KEY}"

session = requests.Session()
session.headers.update(headers)


def rename_in_result(result):
    """Recursively replace old label with new one inside an annotation result."""
    changed = False
    for item in result:
        val = item.get("value", {})
        for key in ("labels", "choices", "taxonomy", "polygonlabels",
                    "rectanglelabels", "ellipselabels", "keypointlabels",
                    "brushlabels", "timeserieslabels", "videorectangle"):
            if key in val and OLD in val[key]:
                val[key] = [NEW if l == OLD else l for l in val[key]]
                changed = True
    return changed


print("Fetching tasks")

page, tasks = 1, []
while True:
    r = session.get(f"{LS_URL}/api/tasks",
                    params={"project": PROJECT, "page": page, "page_size": 200})
    r.raise_for_status()
    data = r.json()
    batch = data.get("tasks", data) if isinstance(data, dict) else data
    if not batch:
        break
    tasks.extend(batch)
    if len(batch) < 200:
        break
    page += 1

print(f"Found {len(tasks)} tasks")

updated_ann = updated_draft = skipped = 0

for t in tasks:
    tid = t["id"]

    # annotations
    if t.get("is_labeled"):
        ann_r = session.get(f"{LS_URL}/api/tasks/{tid}/annotations/")
        if ann_r.ok:
            for ann in ann_r.json():
                if rename_in_result(ann.get("result", [])):
                    r = session.patch(
                        f"{LS_URL}/api/annotations/{ann['id']}/",
                        json={"result": ann["result"]}
                    )
                    if r.ok:
                        updated_ann += 1
                    else:
                        print(f"WARN annotation {ann['id']}: {r.status_code} {r.text[:120]}")

    # drafts
    if t.get("drafts"):
        dr = session.get(f"{LS_URL}/api/tasks/{tid}/drafts/")
        if dr.ok:
            for draft in dr.json():
                if rename_in_result(draft.get("result", [])):
                    r = session.patch(
                        f"{LS_URL}/api/drafts/{draft['id']}/",
                        json={"result": draft["result"]}
                    )
                    if r.ok:
                        updated_draft += 1
                    else:
                        print(f"WARN draft {draft['id']}: {r.status_code} {r.text[:120]}")

print(f"\nUpdated {updated_ann} annotations, {updated_draft} drafts.")
