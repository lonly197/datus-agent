# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Update prompt templates script.

This script copies prompt templates from the project directory to {agent.home}/template,
allowing users to customize templates while preserving the ability to update from the project.

Usage:
    # Dry run - show what would be copied/updated
    python -m scripts.update_prompt_templates --config <config_path> --dry-run

    # Copy missing templates (safe mode - won't overwrite existing)
    python -m scripts.update_prompt_templates --config <config_path>

    # Sync all templates (force overwrite existing)
    python -m scripts.update_prompt_templates --config <config_path> --force

    # List available templates
    python -m scripts.update_prompt_templates --config <config_path> --list

    # Sync specific template
    python -m scripts.update_prompt_templates --config <config_path> --template gen_sql_user
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.utils.loggings import configure_logging, get_logger
from datus.utils.path_manager import get_path_manager

logger = get_logger(__name__)


def get_project_templates_dir() -> Path:
    """Get the project templates directory."""
    return PROJECT_ROOT / "datus" / "prompts" / "prompt_templates"


def parse_template_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse template filename to extract name and version.

    Args:
        filename: Template filename like "gen_sql_user_1.1.j2"

    Returns:
        Tuple of (template_name, version) or None if invalid
    """
    # Remove .j2 extension
    if not filename.endswith(".j2"):
        return None

    name = filename[:-3]

    # Match pattern: {name}_{version}
    match = re.match(r"^(.+)_(\d+\.\d+)$", name)
    if match:
        return match.group(1), match.group(2)

    return None


def get_local_templates(templates_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Get all templates from a directory, grouped by name.

    Args:
        templates_dir: Directory containing template files

    Returns:
        Dict mapping template_name to {version: Path}
    """
    templates = {}

    if not templates_dir.exists():
        return templates

    for file_path in templates_dir.glob("*.j2"):
        parsed = parse_template_filename(file_path.name)
        if parsed:
            name, version = parsed
            if name not in templates:
                templates[name] = {}
            templates[name][version] = file_path

    return templates


def get_latest_template_version(versions: Dict[str, Path]) -> str:
    """Get the latest version from a dict of versions."""
    def version_key(v):
        try:
            return tuple(map(int, v.split(".")))
        except (ValueError, AttributeError):
            return (0, 0)

    sorted_versions = sorted(versions.keys(), key=version_key)
    return sorted_versions[-1] if sorted_versions else ""


def compare_versions(v1: str, v2: str) -> int:
    """
    Compare two version strings.

    Returns:
        1 if v1 > v2, -1 if v1 < v2, 0 if equal
    """
    def version_key(v):
        try:
            return tuple(map(int, v.split(".")))
        except (ValueError, AttributeError):
            return (0, 0)

    v1_key = version_key(v1)
    v2_key = version_key(v2)

    if v1_key > v2_key:
        return 1
    elif v1_key < v2_key:
        return -1
    else:
        return 0


def analyze_templates(
    project_templates: Dict[str, Dict[str, Path]],
    user_templates: Dict[str, Dict[str, Path]],
) -> Dict[str, dict]:
    """
    Analyze template differences between project and user directories.

    Returns:
        Dict with template analysis results
    """
    analysis = {}

    # Get all template names
    all_names = set(project_templates.keys()) | set(user_templates.keys())

    for name in sorted(all_names):
        project_versions = project_templates.get(name, {})
        user_versions = user_templates.get(name, {})

        project_latest = get_latest_template_version(project_versions)
        user_latest = get_latest_template_version(user_versions)

        # Determine status
        if not project_versions:
            status = "only_in_user"
        elif not user_versions:
            status = "only_in_project"
        else:
            cmp = compare_versions(project_latest, user_latest)
            if cmp > 0:
                status = "update_available"
            elif cmp < 0:
                status = "user_ahead"
            else:
                status = "up_to_date"

        analysis[name] = {
            "status": status,
            "project_latest": project_latest,
            "user_latest": user_latest,
            "project_versions": list(project_versions.keys()),
            "user_versions": list(user_versions.keys()),
            "project_path": project_versions.get(project_latest),
            "user_path": user_versions.get(user_latest),
        }

    return analysis


def copy_template(
    src_path: Path,
    dst_path: Path,
    force: bool = False,
) -> Tuple[bool, str]:
    """
    Copy a single template file.

    Args:
        src_path: Source template path
        dst_path: Destination template path
        force: Whether to overwrite existing file

    Returns:
        Tuple of (success: bool, message: str)
    """
    if not src_path.exists():
        return False, f"Source file does not exist: {src_path}"

    # Ensure destination directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists() and not force:
        return False, "File already exists (use --force to overwrite)"

    try:
        shutil.copy2(src_path, dst_path)
        return True, f"Copied: {src_path.name}"
    except Exception as e:
        return False, f"Failed to copy: {e}"


def update_templates(
    agent_config: AgentConfig,
    force: bool = False,
    dry_run: bool = False,
    specific_templates: Optional[List[str]] = None,
) -> Dict:
    """
    Update templates from project to user directory.

    Args:
        agent_config: Agent configuration
        force: Whether to overwrite existing files
        dry_run: If True, only show what would be done
        specific_templates: If provided, only update these templates

    Returns:
        Dict with update results
    """
    results = {
        "copied": [],
        "skipped": [],
        "failed": [],
        "already_up_to_date": [],
    }

    # Get directories
    project_templates_dir = get_project_templates_dir()
    path_manager = get_path_manager()
    user_templates_dir = path_manager.template_dir

    logger.info(f"Project templates: {project_templates_dir}")
    logger.info(f"User templates: {user_templates_dir}")
    logger.info(f"Datus home: {path_manager.datus_home}")
    logger.info("")

    # Load all templates
    project_templates = get_local_templates(project_templates_dir)
    user_templates = get_local_templates(user_templates_dir)

    logger.info(f"Found {len(project_templates)} templates in project")
    logger.info(f"Found {len(user_templates)} templates in user directory")
    logger.info("")

    # Analyze templates
    analysis = analyze_templates(project_templates, user_templates)

    # Process templates
    for name, info in analysis.items():
        # Filter for specific templates if provided
        if specific_templates and name not in specific_templates:
            continue

        status = info["status"]
        project_path = info["project_path"]
        user_path = info["user_path"]

        if status in ["only_in_project", "update_available"]:
            if dry_run:
                action = "Would update" if status == "update_available" else "Would copy"
                logger.info(f"[DRY RUN] {action}: {name} ({info['project_latest']})")
                if status == "update_available":
                    logger.info(f"  Current: {info['user_latest']} -> Latest: {info['project_latest']}")
                results["copied"].append(name)
            else:
                success, msg = copy_template(
                    project_path,
                    user_path,
                    force=force,
                )
                if success:
                    logger.info(f"[OK] {msg}: {name}")
                    results["copied"].append(name)
                else:
                    logger.warning(f"[SKIP] {msg}: {name}")
                    results["skipped"].append(name)

        elif status == "only_in_user":
            logger.info(f"[SKIP] Template only in user directory: {name} ({info['user_latest']})")
            results["already_up_to_date"].append(name)

        elif status == "up_to_date":
            logger.info(f"[OK] Already up to date: {name} ({info['project_latest']})")
            results["already_up_to_date"].append(name)

        elif status == "user_ahead":
            logger.warning(
                f"[WARN] User has newer version: {name} "
                f"(user: {info['user_latest']}, project: {info['project_latest']})"
            )
            if force:
                if dry_run:
                    logger.info(f"[DRY RUN] Would downgrade: {name}")
                    results["copied"].append(name)
                else:
                    success, msg = copy_template(
                        project_path,
                        user_path,
                        force=True,
                    )
                    if success:
                        logger.info(f"[OK] Downgraded: {name} to {info['project_latest']}")
                        results["copied"].append(name)
                    else:
                        logger.error(f"[FAIL] {msg}")
                        results["failed"].append(name)
            else:
                logger.info(f"  Use --force to overwrite with project version")
                results["skipped"].append(name)

    return results


def list_templates(agent_config: AgentConfig) -> None:
    """List all available templates."""
    project_templates_dir = get_project_templates_dir()
    path_manager = get_path_manager()
    user_templates_dir = path_manager.template_dir

    project_templates = get_local_templates(project_templates_dir)
    user_templates = get_local_templates(user_templates_dir)

    analysis = analyze_templates(project_templates, user_templates)

    print("\n" + "=" * 80)
    print("AVAILABLE PROMPT TEMPLATES")
    print("=" * 80)
    print(f"\nProject templates: {project_templates_dir}")
    print(f"User templates: {user_templates_dir}")
    print(f"\nTotal: {len(analysis)} templates\n")

    # Group by status
    by_status = {
        "up_to_date": [],
        "update_available": [],
        "only_in_project": [],
        "only_in_user": [],
        "user_ahead": [],
    }

    for name, info in analysis.items():
        by_status[info["status"]].append(name)

    status_labels = {
        "up_to_date": "Up to Date",
        "update_available": "Update Available",
        "only_in_project": "Only in Project",
        "only_in_user": "Only in User",
        "user_ahead": "User Has Newer",
    }

    for status, label in status_labels.items():
        templates = by_status.get(status, [])
        if templates:
            print(f"\n{label} ({len(templates)}):")
            for name in sorted(templates):
                info = analysis[name]
                versions_str = ""
                if status == "up_to_date":
                    versions_str = f"[{info['project_latest']}]"
                elif status == "update_available":
                    versions_str = f"[{info['user_latest']} -> {info['project_latest']}]"
                elif status == "only_in_project":
                    versions_str = f"[{info['project_latest']}]"
                elif status == "only_in_user":
                    versions_str = f"[{info['user_latest']}]"
                elif status == "user_ahead":
                    versions_str = f"[project: {info['project_latest']}, user: {info['user_latest']}]"

                print(f"  - {name} {versions_str}")

    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update prompt templates from project to user directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run - show what would be updated
  python -m scripts.update_prompt_templates --config ~/.datus/conf/agent.yml --dry-run

  # Copy missing templates (won't overwrite existing)
  python -m scripts.update_prompt_templates --config ~/.datus/conf/agent.yml

  # Force overwrite all templates
  python -m scripts.update_prompt_templates --config ~/.datus/conf/agent.yml --force

  # Sync specific template
  python -m scripts.update_prompt_templates --config ~/.datus/conf/agent.yml --template gen_sql_user

  # List all templates with status
  python -m scripts.update_prompt_templates --config ~/.datus/conf/agent.yml --list
        """,
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to agent configuration file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing templates (including downgrades)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available templates with status",
    )
    parser.add_argument(
        "--template",
        action="append",
        dest="templates",
        default=[],
        help="Specific template(s) to update (can be used multiple times)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging(debug=args.verbose)

    # Load agent configuration
    try:
        agent_config = load_agent_config(config=args.config)
    except Exception as e:
        logger.error(f"Failed to load agent configuration: {e}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("Datus Prompt Template Update Script")
    logger.info("=" * 80)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Datus home: {get_path_manager().datus_home}")
    logger.info("")

    if args.list:
        list_templates(agent_config)
        sys.exit(0)

    if args.dry_run:
        logger.info("[DRY RUN MODE - No changes will be made]")
        logger.info("")

    # Run update
    results = update_templates(
        agent_config=agent_config,
        force=args.force,
        dry_run=args.dry_run,
        specific_templates=args.templates or None,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = len(results["copied"]) + len(results["skipped"]) + len(results["failed"])
    if total > 0:
        print(f"\nCopied: {len(results['copied'])}")
        print(f"Skipped: {len(results['skipped'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"Up to date: {len(results['already_up_to_date'])}")
        print("")

        if results["copied"]:
            print("Copied templates:")
            for name in sorted(results["copied"]):
                print(f"  + {name}")

        if results["skipped"]:
            print("Skipped templates:")
            for name in sorted(results["skipped"]):
                print(f"  - {name}")

        if results["failed"]:
            print("Failed templates:")
            for name in sorted(results["failed"]):
                print(f"  ! {name}")

    print("\n" + "=" * 80)

    # Exit with error code if any failures
    if results["failed"]:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
