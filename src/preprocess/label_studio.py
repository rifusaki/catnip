"""
Label Studio label-rename utilities.

Provides:

* :func:`rename_in_result` — pure helper that rewrites a Label Studio
  annotation ``result`` list in-place, replacing every occurrence of
  ``OLD`` (default ``"kabru"``) with ``NEW`` (default ``"other_face"``).
* :class:`LabelStudioClient` — thin wrapper around the ``requests``
  session that handles JWT refresh-token auth, paginates through tasks,
  and patches annotations / drafts whose ``result`` was changed.

Environment variables (read at client-construction time):

* ``LS_URL``     — base URL (default: ``https://label.rifusaki.com``).
* ``LS_API_KEY`` — required, the personal access token.
* ``LS_PROJECT`` — project ID (default: ``2``).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Iterable

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (read once on import; overridable per-client too)
# ---------------------------------------------------------------------------

LS_URL: str = os.environ.get("LS_URL", "https://label.rifusaki.com")
API_KEY: str = os.environ.get("LS_API_KEY", "")
PROJECT: int = int(os.environ.get("LS_PROJECT", "2"))
OLD: str = "kabru"
NEW: str = "other_face"

_PAGE_SIZE: int = 200

# Every Label Studio control field that may contain the renamed label.
_LABEL_FIELDS: tuple[str, ...] = (
    "labels",
    "choices",
    "taxonomy",
    "polygonlabels",
    "rectanglelabels",
    "ellipselabels",
    "keypointlabels",
    "brushlabels",
    "timeserieslabels",
    "videorectangle",
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def rename_in_result(
    result: list[dict[str, Any]],
    old: str | None = None,
    new: str | None = None,
) -> bool:
    """Recursively replace ``old`` with ``new`` inside an annotation result list.

    Defaults to the module-level ``OLD`` and ``NEW`` constants (so the
    zero-argument call preserves the original behaviour of the old
    ``utils/change_label.py`` script).

    Mutates ``result`` in place (the ``value`` dicts are sub-dicts of
    ``result`` items).  Returns ``True`` if any replacement was made.
    """
    old_label = OLD if old is None else old
    new_label = NEW if new is None else new
    changed = False
    for item in result:
        val = item.get("value", {})
        for key in _LABEL_FIELDS:
            if key in val and old_label in val[key]:
                val[key] = [new_label if lbl == old_label else lbl for lbl in val[key]]
                changed = True
    return changed


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LabelStudioClient:
    """Small wrapper around the Label Studio HTTP API.

    Example::

        client = LabelStudioClient()  # reads LS_URL / LS_API_KEY / LS_PROJECT
        n = client.rename_all(
            new_label="other_face", old_label="kabru", include_drafts=True
        )
        print(f"Updated {n} records.")
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        project: int | None = None,
    ) -> None:
        self.url: str = url if url is not None else LS_URL
        self.api_key: str = api_key if api_key is not None else API_KEY
        self.project: int = project if project is not None else PROJECT
        self.session: requests.Session = self._build_session()

    # ----- Auth -----

    def _build_session(self) -> requests.Session:
        """Authenticate and return a configured ``requests.Session``.

        Raises:
            RuntimeError: if no API key is configured.
            requests.HTTPError: if the token refresh endpoint fails.
        """
        if not self.api_key:
            raise RuntimeError(
                "LS_API_KEY environment variable must be set. "
                "Please set LS_API_KEY to your Label Studio API key and try again."
            )

        headers: dict[str, str] = {}
        if self.api_key.count(".") == 2:
            # Label Studio >= 1.22.0 uses JWT refresh tokens for PAT
            r_token = requests.post(
                f"{self.url}/api/token/refresh/", json={"refresh": self.api_key}
            )
            r_token.raise_for_status()
            access_token = r_token.json()["access"]
            headers["Authorization"] = f"Bearer {access_token}"
        else:
            # legacy plain token
            headers["Authorization"] = f"Token {self.api_key}"

        session = requests.Session()
        session.headers.update(headers)
        return session

    # ----- Task listing -----

    def list_annotations(self) -> list[dict[str, Any]]:
        """Fetch every task in the configured project (across pages).

        Returns the raw task dicts, with ``id``, ``is_labeled``, ``drafts``,
        and other Label Studio fields populated.
        """
        logger.info("Fetching tasks from project %s", self.project)

        tasks: list[dict[str, Any]] = []
        page = 1
        while True:
            r = self.session.get(
                f"{self.url}/api/tasks",
                params={"project": self.project, "page": page, "page_size": _PAGE_SIZE},
            )
            r.raise_for_status()
            data = r.json()
            batch = data.get("tasks", data) if isinstance(data, dict) else data
            if not batch:
                break
            tasks.extend(batch)
            if len(batch) < _PAGE_SIZE:
                break
            page += 1

        logger.info("Found %d tasks", len(tasks))
        return tasks

    # ----- Rename -----

    def rename_label(
        self,
        annotation_id: int,
        new_label: str | None = None,
        old_label: str | None = None,
    ) -> bool:
        """Rename ``old_label`` (default module ``OLD``) in one annotation.

        Returns ``True`` if the annotation was patched.
        """
        ann_r = self.session.get(f"{self.url}/api/annotations/{annotation_id}/")
        if not ann_r.ok:
            logger.warning(
                "WARN annotation %s: %s %s",
                annotation_id, ann_r.status_code, ann_r.text[:120],
            )
            return False

        ann = ann_r.json()
        return self._patch_record(
            record_id=ann["id"],
            endpoint="annotations",
            result=ann.get("result", []),
            new_label=new_label if new_label is not None else NEW,
            old_label=old_label if old_label is not None else OLD,
        )

    def rename_all(
        self,
        task_ids: Iterable[int] | None = None,
        new_label: str | None = None,
        old_label: str | None = None,
        include_drafts: bool = True,
    ) -> dict[str, int]:
        """Rename labels across every task in the project (or a subset).

        If *task_ids* is ``None`` (the default) every task is processed; the
        iteration uses :meth:`list_annotations` for the task list and
        paginates through each task's annotations/drafts.

        Returns counts: ``{"annotations": int, "drafts": int}``.
        """
        old_label = old_label if old_label is not None else OLD
        new_label = new_label if new_label is not None else NEW

        if task_ids is None:
            tasks = self.list_annotations()
        else:
            tasks = [{"id": tid, "is_labeled": True} for tid in task_ids]

        updated_ann = 0
        updated_draft = 0

        for task in tasks:
            tid = task["id"]

            # --- annotations ---
            if task.get("is_labeled", True):
                ann_r = self.session.get(f"{self.url}/api/tasks/{tid}/annotations/")
                if ann_r.ok:
                    for ann in ann_r.json():
                        if self._patch_record(
                            record_id=ann["id"],
                            endpoint="annotations",
                            result=ann.get("result", []),
                            new_label=new_label,
                            old_label=old_label,
                        ):
                            updated_ann += 1

            # --- drafts ---
            if include_drafts and task.get("drafts"):
                dr = self.session.get(f"{self.url}/api/tasks/{tid}/drafts/")
                if dr.ok:
                    for draft in dr.json():
                        if self._patch_record(
                            record_id=draft["id"],
                            endpoint="drafts",
                            result=draft.get("result", []),
                            new_label=new_label,
                            old_label=old_label,
                        ):
                            updated_draft += 1

        logger.info("Updated %d annotations, %d drafts.", updated_ann, updated_draft)
        return {"annotations": updated_ann, "drafts": updated_draft}

    # ----- Internal -----

    def _patch_record(
        self,
        record_id: int,
        endpoint: str,
        result: list[dict[str, Any]],
        new_label: str,
        old_label: str,
    ) -> bool:
        """Patch a single annotation/draft if its result contains *old_label*.

        Returns ``True`` if the record was updated.
        """
        if not rename_in_result(result, old=old_label, new=new_label):
            return False

        r = self.session.patch(
            f"{self.url}/api/{endpoint}/{record_id}/",
            json={"result": result},
        )
        if r.ok:
            return True
        logger.warning(
            "WARN %s %s: %s %s",
            endpoint.rstrip("s"), record_id, r.status_code, r.text[:120],
        )
        return False


# ---------------------------------------------------------------------------
# CLI parity — preserved main() from the original utils/change_label.py
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the label renaming process against the configured Label Studio project."""
    try:
        client = LabelStudioClient()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    counts = client.rename_all()
    # Print a single-line summary on top of the logger output.
    print(json.dumps({"updated": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
