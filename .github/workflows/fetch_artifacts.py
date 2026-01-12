#  This code is a part of the HEASARC-tutorials project, created by David J Turner (djturner@umbc.edu)
#  Generative AI tools were used in the creation of this file.

import os
import argparse
import pathlib
import requests
from requests.exceptions import HTTPError, RequestException


def get_circleci_artifacts(project_slug, commit_sha, token):
    url = f"https://circleci.com/api/v2/project/{project_slug}/pipeline?revision={commit_sha}"
    headers = {"Circle-Token": token}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        pipelines = response.json().get("items", [])

        if not pipelines:
            print(f"No CircleCI pipeline found for commit {commit_sha}")
            return []

        # Get latest pipeline
        pipeline_id = pipelines[0]["id"]

        # Get workflows
        wf_url = f"https://circleci.com/api/v2/pipeline/{pipeline_id}/workflow"
        wf_response = requests.get(wf_url, headers=headers)
        wf_response.raise_for_status()
        workflows = wf_response.json().get("items", [])

        for wf in workflows:
            # Get successful jobs
            jobs_url = f"https://circleci.com/api/v2/workflow/{wf['id']}/job"
            jobs_response = requests.get(jobs_url, headers=headers)
            jobs_response.raise_for_status()
            jobs = jobs_response.json().get("items", [])

            for job in jobs:
                if job["status"] == "success":
                    job_num = job["job_number"]
                    # Get artifacts using v2 API
                    art_url = f"https://circleci.com/api/v2/project/{project_slug}/{job_num}/artifacts"
                    art_response = requests.get(art_url, headers=headers)
                    if art_response.status_code == 200:
                        return art_response.json().get("items", [])

    except (HTTPError, RequestException, KeyError, IndexError) as e:
        print(f"Error communicating with CircleCI: {e}")

    return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--files", required=True)
    args = parser.parse_args()

    token = os.getenv("CIRCLE_TOKEN")

    # project_slug = f"gh/{args.repo}"
    # project_slug = 'circleci/LdzbTUR6aexSM6vJCVrZk5/GRoaAeYuJNgCdifZviG612'
    project_slug = os.getenv("PROJECT_SLUG")

    artifacts = get_circleci_artifacts(project_slug, args.commit, token)

    if not artifacts:
        print("STATUS: NO_ARTIFACTS")
        return

    output_dir = pathlib.Path("downloaded_artifacts")
    output_dir.mkdir(exist_ok=True)

    for md_path in args.files.split():
        if not md_path.startswith("tutorials/"): continue

        rel_path = md_path.replace("tutorials/", "")
        artifact_suffix = f"executed_notebooks/{rel_path.replace('.md', '.ipynb')}"

        for art in artifacts:
            if art["path"].endswith(artifact_suffix):
                print(f"Downloading artifact for {md_path}")
                resp = requests.get(art["url"], headers={"Circle-Token": token})
                local_path = output_dir / rel_path.replace(".md", ".ipynb")
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(resp.content)
                break

if __name__ == "__main__":
    main()
