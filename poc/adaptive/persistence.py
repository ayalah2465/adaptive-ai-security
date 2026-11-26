import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from .feedback_loop import DefenseCandidate


@dataclass(frozen=True)
class ConfigurationVersion:
    version: int
    candidate: DefenseCandidate
    candidate_id: str | None = None
    restored_from: int | None = None


class ConfigurationPersistence:
    """Store checkpoints and approved versions in a local SQLite database."""

    def __init__(self, path: Path, baseline: DefenseCandidate) -> None:
        self._path = path
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS configuration_versions (
                    version INTEGER PRIMARY KEY,
                    candidate TEXT NOT NULL,
                    candidate_id TEXT,
                    restored_from INTEGER,
                    approval_thread TEXT UNIQUE
                )
            """)
            connection.execute(
                "INSERT OR IGNORE INTO configuration_versions (version, candidate) VALUES (0, ?)",
                (baseline.model_dump_json(),)
            )
            connection.execute("""
                CREATE TABLE IF NOT EXISTS adaptive_experiments (
                    experiment_id TEXT PRIMARY KEY,
                    approval_thread TEXT UNIQUE NOT NULL,
                    report_path TEXT NOT NULL,
                    report TEXT NOT NULL
                )
            """)
        self._checkpoint_connection = sqlite3.connect(path, check_same_thread=False)
        self.checkpointer = SqliteSaver(self._checkpoint_connection)

    def close(self) -> None:
        self._checkpoint_connection.close()

    def save_experiment(self, thread_id: str, report_path: Path, report: dict) -> None:
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute("""
                INSERT INTO adaptive_experiments
                    (experiment_id, approval_thread, report_path, report)
                VALUES (?, ?, ?, ?)
            """, (report["experiment_id"], thread_id, str(report_path.resolve()),
                  json.dumps(report, ensure_ascii=False, allow_nan=False)))

    def load_experiment(self, experiment_id: str) -> tuple[str, Path, dict]:
        with closing(sqlite3.connect(self._path)) as connection:
            row = connection.execute("""
                SELECT approval_thread, report_path, report
                FROM adaptive_experiments WHERE experiment_id = ?
            """, (experiment_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown adaptive experiment")
        thread_id, report_path, report = row
        return thread_id, Path(report_path), json.loads(report)

    def finalize_experiment(self, report: dict) -> dict:
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT report FROM adaptive_experiments WHERE experiment_id = ?",
                (report["experiment_id"],)
            ).fetchone()
            if row is None:
                raise ValueError("Unknown adaptive experiment")
            existing = json.loads(row[0])
            if "status" in existing:
                return existing
            connection.execute(
                "UPDATE adaptive_experiments SET report = ? WHERE experiment_id = ?",
                (json.dumps(report, ensure_ascii=False, allow_nan=False), report["experiment_id"])
            )
        return report

    def activated_version(self, thread_id: str) -> int | None:
        with closing(sqlite3.connect(self._path)) as connection:
            row = connection.execute(
                "SELECT version FROM configuration_versions WHERE approval_thread = ?", (thread_id,)
            ).fetchone()
        return row[0] if row is not None else None

    def history(self) -> tuple[ConfigurationVersion, ...]:
        with closing(sqlite3.connect(self._path)) as connection:
            rows = connection.execute("""
                SELECT version, candidate, candidate_id, restored_from
                FROM configuration_versions ORDER BY version
            """).fetchall()
        return tuple(
            ConfigurationVersion(version, DefenseCandidate.model_validate_json(candidate),
                                 candidate_id, restored_from)
            for version, candidate, candidate_id, restored_from in rows
        )

    def append(
        self,
        version: ConfigurationVersion,
        *,
        expected_version: int,
        approval_thread: str | None = None
    ) -> None:
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            if approval_thread is not None and connection.execute(
                "SELECT 1 FROM configuration_versions WHERE approval_thread = ?",
                (approval_thread,)
            ).fetchone():
                raise ValueError("Approval has already been used")

            active, = connection.execute("SELECT MAX(version) FROM configuration_versions").fetchone()
            if active != expected_version:
                raise ValueError("Approval is stale for the active version")
            connection.execute("""
                INSERT INTO configuration_versions
                    (version, candidate, candidate_id, restored_from, approval_thread)
                VALUES (?, ?, ?, ?, ?)
            """, (
                version.version, version.candidate.model_dump_json(), version.candidate_id,
                version.restored_from, approval_thread
            ))
