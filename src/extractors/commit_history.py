"""
Commit History Extractor — extracts compliance-relevant commits from the
PyTorch git history.

Compliance relevance:
    - Security commits map to EU AI Act Articles 15 and 17.
    - Determinism commits map to Article 15.
    - Deprecation commits map to Article 13.
    - Breaking-change commits map to Article 13.
    - Data-handling commits map to Article 10.

Uses ``git log --grep`` to filter commits matching compliance keywords,
capped at 5000 commits to prevent runaway extraction.
"""

from __future__ import annotations

import subprocess
import logging
from pathlib import Path
from typing import Any

from .base import EntityRecord, BaseExtractor, compute_stable_id

logger = logging.getLogger("pct.extractors.commit_history")

# ---------------------------------------------------------------------------
# Compliance keywords to grep for in commit messages
# ---------------------------------------------------------------------------

COMPLIANCE_KEYWORDS: list[str] = [
    "security", "deterministic", "deprecat", "breaking",
    "backward compat", "audit", "compliance", "privacy",
    "fairness", "bias", "reproducib", "hook", "dispatch",
    "numerical stability", "overflow", "underflow",
    "nan", "inf", "CVE", "vulnerability", "SECURITY",
    "data_handling", "data handling", "data governance",
]

# ---------------------------------------------------------------------------
# Commit-type classification → compliance tags
# ---------------------------------------------------------------------------

_COMMIT_TYPE_TAGS: dict[str, list[str]] = {
    "security":       ["eu_ai_act_art_15", "eu_ai_act_art_17"],
    "determinism":    ["eu_ai_act_art_15"],
    "deprecation":    ["eu_ai_act_art_13"],
    "breaking_change": ["eu_ai_act_art_13"],
    "data_handling":  ["eu_ai_act_art_10"],
    "extensibility":  [],      # informational
    "general":        [],      # no specific article
}

# Maximum number of commits to extract (prevents runaway on large repos)
_MAX_COMMITS = 5000


class CommitHistoryExtractor(BaseExtractor):
    """
    Extract compliance-relevant commits from the PyTorch git history.

    Runs ``git log --grep`` with a combined regex of all compliance keywords
    and classifies each matching commit into a commit type, then tags it
    with the appropriate EU AI Act articles.
    """

    def __init__(self, repo_path: Path, output_path: Path) -> None:
        """
        Initialise the commit history extractor.

        Parameters
        ----------
        repo_path : Path
            Root of the PyTorch repository checkout.
        output_path : Path
            Directory where output JSONL files are written.
        """
        super().__init__(name="commit_history", repo_path=repo_path, output_path=output_path)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def extract(self) -> int:
        """
        Run the commit history extraction pass.

        Returns
        -------
        int
            Total number of records produced.
        """
        self.logger.info("Starting commit history extraction")
        output_file = str(self.output_path / "commit_history.jsonl")

        # Build a combined grep pattern from all compliance keywords
        grep_pattern = "|".join(COMPLIANCE_KEYWORDS)

        # Execute git log with grep filtering
        try:
            result = subprocess.run(
                [
                    "git", "-C", str(self.repo_path), "log",
                    "--all",
                    f"--grep={grep_pattern}",
                    "--extended-regexp",    # required for | to work as alternation
                    "--regexp-ignore-case",
                    # Delimiter-separated format for reliable parsing
                    "--format=%H|||%an|||%ae|||%aI|||%s|||%b",
                    "-n", str(_MAX_COMMITS),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            self.logger.error("Git log command timed out after 120 seconds")
            self._errors += 1
            return 0
        except FileNotFoundError:
            self.logger.error("Git executable not found — is git installed?")
            self._errors += 1
            return 0

        if result.returncode != 0:
            self.logger.error("Git log failed (rc=%d): %s", result.returncode, result.stderr[:500])
            self._errors += 1
            return 0

        # Mark the git history as one "file" processed
        self._files_processed += 1

        # Parse each commit line
        for line in result.stdout.strip().splitlines():
            parts = line.split("|||")
            if len(parts) < 5:
                # Malformed line — skip silently
                continue

            commit_hash = parts[0]
            author = parts[1]
            email = parts[2]
            date = parts[3]
            subject = parts[4]
            body = parts[5] if len(parts) > 5 else ""

            full_message = f"{subject}\n{body}".lower()

            # Find which compliance keywords matched
            matched_keywords = [
                kw for kw in COMPLIANCE_KEYWORDS
                if kw.lower() in full_message
            ]

            # Classify the commit type
            commit_type = self._classify_commit(matched_keywords)

            # Look up compliance tags for this commit type
            compliance_tags = _COMMIT_TYPE_TAGS.get(commit_type, [])

            # Truncated hash for display and ID purposes
            short_hash = commit_hash[:12]

            record = self.make_record(
                source_file=f"git:{short_hash}",
                language="config",
                entity_name=short_hash,
                entity_type="commit",
                subcategory=commit_type,
                module_path="git.history",
                qualified_name=f"git.{short_hash}",
                start_line=0,
                end_line=0,
                raw_text=f"{subject}\n{body}"[:2000],
                compliance_tags=list(compliance_tags),
                extraction_confidence=0.9,
                metadata={
                    "hash": commit_hash,
                    "author": author,
                    "email": email,
                    "date": date,
                    "subject": subject,
                    "matched_keywords": matched_keywords,
                    "commit_type": commit_type,
                },
            )
            self.write_record(record, output_file)

        # Flush remaining buffered records
        self.flush(output_file)

        self.logger.info(
            "Commit history extraction complete — %d records produced",
            self._records_produced,
        )
        self.report_stats()
        return self._records_produced

    # ------------------------------------------------------------------ #
    # Commit classification helper
    # ------------------------------------------------------------------ #

    @staticmethod
    def _classify_commit(matched_keywords: list[str]) -> str:
        """
        Classify a commit into a type based on matched compliance keywords.

        The classification follows a priority order: security > determinism >
        deprecation > breaking_change > data_handling > extensibility > general.

        Parameters
        ----------
        matched_keywords : list[str]
            Keywords that were found in the commit message.

        Returns
        -------
        str
            Commit type string (e.g. ``"security"``, ``"determinism"``).
        """
        kw_set = {kw.lower() for kw in matched_keywords}

        # Security takes highest priority
        if kw_set & {"security", "cve", "vulnerability"}:
            return "security"

        # Determinism / reproducibility
        if kw_set & {"deterministic", "reproducib"}:
            return "determinism"

        # Deprecation
        if any("deprecat" in kw for kw in kw_set):
            return "deprecation"

        # Breaking changes
        if kw_set & {"breaking", "backward compat"}:
            return "breaking_change"

        # Data handling
        if kw_set & {"data_handling", "data handling", "data governance"}:
            return "data_handling"

        # Extensibility (hooks, dispatch)
        if kw_set & {"hook", "dispatch"}:
            return "extensibility"

        return "general"
