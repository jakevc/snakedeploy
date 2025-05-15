from collections import namedtuple
import copy
import json
import os
from pathlib import Path
import subprocess as sp
import tempfile
import re
from glob import glob
from itertools import chain
import github
from urllib3.util.retry import Retry
import random
import shutil

from packaging import version as packaging_version
import yaml
from github import Github, GithubException
from reretry import retry

import rattler
from rattler.platform import Platform
from rattler.match_spec import MatchSpec
from rattler.shell import Shell
from rattler.repo_data import RepoDataRecord

from snakedeploy.exceptions import UserError
from snakedeploy.logger import logger
from snakedeploy.utils import YamlDumper
from snakedeploy.conda_version import VersionOrder


class RattlerResult:
    def __init__(self, success=True, stdout="", stderr=""):
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = 0 if success else 1


def pin_conda_envs(
    conda_env_paths: list,
    conda_frontend=None,  # Kept for backward compatibility but ignored
    create_prs=False,
    pr_add_label=False,
    entity_regex=None,
    warn_on_error=False,
):
    """Pin given conda envs by creating <conda-env>.<platform>.pin.txt
    files with explicit URLs for all packages in each env."""
    return CondaEnvProcessor(conda_frontend=conda_frontend).process(
        conda_env_paths,
        update_envs=False,
        pin_envs=True,
        create_prs=create_prs,
        pr_add_label=pr_add_label,
        entity_regex=entity_regex,
        warn_on_error=warn_on_error,
    )


def update_conda_envs(
    conda_env_paths: list,
    conda_frontend=None,  # Kept for backward compatibility but ignored
    create_prs=False,
    pin_envs=False,
    pr_add_label=False,
    entity_regex=None,
    warn_on_error=False,
):
    """Update the given conda env definitions such that all dependencies
    in them are set to the latest feasible versions."""
    return CondaEnvProcessor().process(
        conda_env_paths,
        create_prs=create_prs,
        update_envs=True,
        pin_envs=pin_envs,
        pr_add_label=pr_add_label,
        entity_regex=entity_regex,
        warn_on_error=warn_on_error,
    )


File = namedtuple("File", "path, content, is_updated, msg")


class CondaEnvProcessor:
    def __init__(self, conda_frontend=None):
        # conda_frontend parameter is kept for backward compatibility but ignored
        self.conda_frontend = "rattler"  # Always use rattler
        self.use_rattler = True

        self.platform = Platform.current()
        self.info = {"platform": str(self.platform).split("-")[0]}

    def process(
        self,
        conda_env_paths,
        create_prs=False,
        update_envs=True,
        pin_envs=True,
        pr_add_label=False,
        entity_regex=None,
        warn_on_error=False,
    ):
        repo = None
        if create_prs:
            g = Github(
                os.environ["GITHUB_TOKEN"],
                retry=Retry(
                    total=10, status_forcelist=(500, 502, 504), backoff_factor=0.3
                ),
            )
            repo = g.get_repo(os.environ["GITHUB_REPOSITORY"]) if create_prs else None
        conda_envs = list(chain.from_iterable(map(glob, conda_env_paths)))
        random.shuffle(conda_envs)

        if not conda_envs:
            logger.info(
                f"No conda envs found at given paths: {', '.join(conda_env_paths)}"
            )
        for conda_env_path in conda_envs:
            if create_prs:
                entity = conda_env_path
                if entity_regex is not None:
                    m = re.match(entity_regex, conda_env_path)
                    if m is None:
                        raise UserError(
                            f"Given --entity-regex did not match any {conda_env_path}."
                        )
                    try:
                        entity = m.group("entity")
                    except IndexError:
                        raise UserError(
                            "No group 'entity' found in given --entity-regex."
                        )
                if pr_add_label and not entity_regex:
                    raise UserError(
                        "Cannot add label to PR without --entity-regex specified."
                    )

                assert (
                    pin_envs or update_envs
                ), "bug: either pin_envs or update_envs must be True"
                mode = "bump" if update_envs else "pin"
                pr = PR(
                    f"perf: auto{mode} {entity}",
                    f"Automatic {mode} of {entity}.",
                    f"auto{mode}/{entity.replace('/', '-')}",
                    repo,
                    label=entity if pr_add_label else None,
                )
            else:
                pr = None
            try:
                updated = False
                if update_envs:
                    logger.info(f"Updating {conda_env_path}...")
                    updated = self.update_env(
                        conda_env_path, pr=pr, warn_on_error=warn_on_error
                    )
                if pin_envs and (
                    not update_envs
                    or updated
                    or not self.get_pin_file_path(conda_env_path).exists()
                ):
                    logger.info(f"Pinning {conda_env_path}...")
                    self.update_pinning(conda_env_path, pr)
            except sp.CalledProcessError as e:
                msg = f"Failed for conda env {conda_env_path}:\n{e.stderr}\n{e.stdout}"
                if warn_on_error:
                    logger.warning(msg)
                else:
                    raise UserError(msg)
            if create_prs:
                pr.create()

    def update_env(
        self,
        conda_env_path,
        pr=None,
        warn_on_error=False,
    ):
        spec_re = re.compile("(?P<name>[^=>< ]+)[ =><]+")
        with open(conda_env_path, "r") as infile:
            conda_env = yaml.load(infile, Loader=yaml.SafeLoader)

        def process_dependencies(func):
            def process_dependency(dep):
                if isinstance(dep, dict):
                    # leave e.g. pip subdicts unchanged
                    return dep
                m = spec_re.match(dep)
                if m is None:
                    # cannot parse the spec, leave unchanged
                    return dep
                return func(m.group("name"))

            return [process_dependency(dep) for dep in conda_env["dependencies"]]

        def get_pkg_versions(conda_env_path):
            with tempfile.TemporaryDirectory(dir=".", prefix=".") as tmpdir:
                self.exec_conda(f"env create --file {conda_env_path} --prefix {tmpdir}")
                result = self.exec_conda(f"list --json --prefix {tmpdir}")
                results = json.loads(result.stdout)
                pkg_versions = {pkg["name"]: pkg["version"] for pkg in results}
                self.exec_conda(f"env remove --prefix {tmpdir} -y")
            return pkg_versions, results

        logger.info("Resolving prior versions...")
        prior_pkg_versions, _ = get_pkg_versions(conda_env_path)

        unconstrained_deps = process_dependencies(lambda name: name)
        unconstrained_env = dict(conda_env)
        unconstrained_env["dependencies"] = unconstrained_deps

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", dir=".", prefix="."
        ) as tmpenv:
            yaml.dump(unconstrained_env, tmpenv, Dumper=YamlDumper)
            logger.info("Resolving posterior versions...")
            posterior_pkg_versions, posterior_pkg_json = get_pkg_versions(tmpenv.name)

        def downgraded():
            downgraded_pkgs = []
            for pkg_name, version in posterior_pkg_versions.items():
                try:
                    version = VersionOrder(version)
                except packaging_version.InvalidVersion as e:
                    logger.debug(json.dumps(posterior_pkg_json, indent=2))
                    raise UserError(
                        f"Cannot parse version {version} of package {pkg_name}: {e}"
                    )
                prior_version = prior_pkg_versions.get(pkg_name)
                if prior_version is not None and version < VersionOrder(prior_version):
                    downgraded_pkgs.append(pkg_name)
            return downgraded_pkgs

        downgraded_pkgs = set(unconstrained_deps) & set(downgraded())
        if downgraded_pkgs:
            msg = (
                f"Env {conda_env_path} could not be updated because the following packages "
                f"would be downgraded: {', '.join(downgraded_pkgs)}. Please consider a manual update "
                "of the environment."
            )
            if warn_on_error:
                logger.warning(msg)
            else:
                raise UserError(msg)

        orig_env = copy.deepcopy(conda_env)

        conda_env["dependencies"] = process_dependencies(
            lambda name: f"{name} ={posterior_pkg_versions[name]}"
        )
        if orig_env != conda_env:
            with open(conda_env_path, "w") as outfile:
                yaml.dump(conda_env, outfile, Dumper=YamlDumper)
            if pr:
                with open(conda_env_path, "r") as infile:
                    content = infile.read()

                pr.add_file(
                    conda_env_path,
                    content,
                    is_updated=True,
                    msg=f"perf: update {conda_env_path}.",
                )
            return True
        else:
            logger.info("No updates in env.")
            return False

    def get_pin_file_path(self, conda_env_path):
        return Path(conda_env_path).with_suffix(f".{self.info['platform']}.pin.txt")

    def update_pinning(self, conda_env_path, pr=None):
        pin_file = self.get_pin_file_path(conda_env_path)
        old_content = None
        updated = False
        if pin_file.exists():
            with open(pin_file, "r") as infile:
                old_content = infile.read()

        with tempfile.TemporaryDirectory(dir=".", prefix=".") as tmpdir:
            self.exec_conda(f"env create --prefix {tmpdir} --file {conda_env_path}")
            self.exec_conda(
                f"list --explicit --md5 --prefix {tmpdir} > {tmpdir}/pin.txt"
            )
            with open(f"{tmpdir}/pin.txt", "r") as infile:
                new_content = infile.read()
            updated = old_content != new_content
            if updated:
                with open(pin_file, "w") as outfile:
                    outfile.write(new_content)
                if pr:
                    msg = (
                        "perf: update env pinning."
                        if old_content is not None
                        else f"feat: add pinning for {conda_env_path}."
                    )
                    pr.add_file(
                        pin_file,
                        new_content,
                        is_updated=old_content is not None,
                        msg=msg,
                    )
            self.exec_conda(f"env remove --prefix {tmpdir} -y")

    def exec_conda(self, subcmd):
        """Execute conda commands through py-rattler API"""
        result = self._exec_rattler(subcmd)
        return result

    def _exec_rattler(self, subcmd):
        """Execute py-rattler API commands based on the subcommand"""
        parts = subcmd.strip().split()
        cmd = parts[0]
        args = parts[1:]

        if cmd == "env" and len(args) >= 1:
            subcmd = args[0]
            if subcmd == "create":
                return self._rattler_create_env(args[1:])
            elif subcmd == "remove":
                return self._rattler_remove_env(args[1:])
        elif cmd == "list":
            return self._rattler_list_packages(args)
        elif cmd == "info":
            return self._rattler_info(args)

        raise UserError(f"Unsupported rattler command: {subcmd}")

    def _rattler_info(self, args):
        """Get conda info using py-rattler"""
        if "--json" in args:
            info = {
                "platform": str(self.platform).split("-")[0],
                "channels": ["conda-forge"],
            }

            return RattlerResult(success=True, stdout=json.dumps(info))

        raise UserError("Only --json format is supported for rattler info")

    def _rattler_create_env(self, args):
        """Create a conda environment using py-rattler"""
        prefix = None
        env_file = None

        i = 0
        while i < len(args):
            if args[i] == "--prefix" and i + 1 < len(args):
                prefix = args[i + 1]
                i += 2
            elif args[i] == "--file" and i + 1 < len(args):
                env_file = args[i + 1]
                i += 2
            else:
                i += 1

        if not prefix or not env_file:
            raise UserError("Missing required arguments for environment creation")

        with open(env_file, "r") as f:
            env_config = yaml.safe_load(f)

        # Extract dependencies
        dependencies = env_config.get("dependencies", [])
        specs = []

        for dep in dependencies:
            if isinstance(dep, dict):
                continue
            specs.append(MatchSpec(dep))

        channels = env_config.get("channels", ["conda-forge"])

        os.makedirs(prefix, exist_ok=True)

        try:
            result = rattler.solve(
                specs=specs, platforms=[self.platform], channels=channels
            )

            if not result.success:
                raise UserError(f"Could not solve environment: {result.error}")

            rattler.install(
                records=result.records, prefix=prefix, platforms=[self.platform]
            )

            return RattlerResult(success=True)
        except Exception as e:
            return RattlerResult(success=False, stderr=str(e))

    def _rattler_list_packages(self, args):
        """List packages in a conda environment using py-rattler"""
        prefix = None
        output_json = False
        explicit = False
        md5 = False

        i = 0
        while i < len(args):
            if args[i] == "--prefix" and i + 1 < len(args):
                prefix = args[i + 1]
                i += 2
            elif args[i] == "--json":
                output_json = True
                i += 1
            elif args[i] == "--explicit":
                explicit = True
                i += 1
            elif args[i] == "--md5":
                md5 = True
                i += 1
            else:
                i += 1

        if not prefix:
            raise UserError("Missing prefix for listing packages")

        try:
            shell = Shell(prefix=prefix)

            packages = shell.installed_packages()

            if explicit and md5:
                result = ["# This file may be used to create an environment using:"]
                result.append("# $ conda create --name <env> --file <this file>")
                result.append("@EXPLICIT")

                for pkg in packages:
                    record = RepoDataRecord.from_package(pkg)
                    result.append(f"{record.url}#{record.md5}")

                redirect_idx = -1
                for i, arg in enumerate(args):
                    if arg == ">":
                        redirect_idx = i
                        break

                if redirect_idx >= 0 and redirect_idx + 1 < len(args):
                    output_file = args[redirect_idx + 1]
                    with open(output_file, "w") as f:
                        f.write("\n".join(result))

                    return RattlerResult(success=True)

                return RattlerResult(success=True, stdout="\n".join(result))

            if output_json:
                result = []
                for pkg in packages:
                    record = RepoDataRecord.from_package(pkg)
                    result.append(
                        {
                            "name": record.name,
                            "version": record.version,
                            "build": record.build_string,
                            "channel": record.channel,
                        }
                    )

                return RattlerResult(success=True, stdout=json.dumps(result))

            result = []
            for pkg in packages:
                record = RepoDataRecord.from_package(pkg)
                result.append(f"{record.name} {record.version} {record.build_string}")

            return RattlerResult(success=True, stdout="\n".join(result))
        except Exception as e:
            return RattlerResult(success=False, stderr=str(e))

    def _rattler_remove_env(self, args):
        """Remove a conda environment using py-rattler"""
        prefix = None

        i = 0
        while i < len(args):
            if args[i] == "--prefix" and i + 1 < len(args):
                prefix = args[i + 1]
                i += 2
            elif args[i] == "-y":
                i += 1
            else:
                i += 1

        if not prefix:
            raise UserError("Missing prefix for environment removal")

        try:
            if os.path.exists(prefix):
                shutil.rmtree(prefix)

            return RattlerResult(success=True)
        except Exception as e:
            return RattlerResult(success=False, stderr=str(e))


class PR:
    def __init__(self, title, body, branch, repo, label=None):
        self.title = title
        self.body = body
        self.files = []
        self.branch = branch
        self.repo = repo
        self.base_ref = (
            os.environ.get("GITHUB_BASE_REF") or os.environ["GITHUB_REF_NAME"]
        )
        self.label = label

    def add_file(self, filepath, content, is_updated, msg):
        self.files.append(File(str(filepath), content, is_updated, msg))

    @retry(tries=2, delay=60)
    def create(self):
        if not self.files:
            logger.info("No files to commit.")
            return

        branch_exists = False
        try:
            b = self.repo.get_branch(self.branch)
            logger.info(f"Branch {b} already exists.")
            branch_exists = True
        except GithubException as e:
            if e.status != 404:
                raise e
            logger.info(f"Creating branch {self.branch}...")
            self.repo.create_git_ref(
                ref=f"refs/heads/{self.branch}",
                sha=self.repo.get_branch(self.base_ref).commit.sha,
            )
        for file in self.files:
            sha = None
            if branch_exists:
                logger.info(f"Obtaining sha of {file.path} on branch {self.branch}...")
                try:
                    # try to get sha if file exists
                    sha = self.repo.get_contents(file.path, self.branch).sha
                except github.GithubException.UnknownObjectException as e:
                    if e.status != 404:
                        raise e
            elif file.is_updated:
                logger.info(
                    f"Obtaining sha of {file.path} on branch {self.base_ref}..."
                )
                sha = self.repo.get_contents(file.path, self.base_ref).sha

            if sha is not None:
                self.repo.update_file(
                    file.path,
                    file.msg,
                    file.content,
                    sha,
                    branch=self.branch,
                )
            else:
                self.repo.create_file(
                    file.path, file.msg, file.content, branch=self.branch
                )

        pr_exists = any(
            pr.head.label.split(":", 1)[1] == self.branch
            for pr in self.repo.get_pulls(state="open", base=self.base_ref)
        )
        if pr_exists:
            logger.info("PR already exists.")
        else:
            pr = self.repo.create_pull(
                title=self.title,
                body=self.body,
                head=self.branch,
                base=self.base_ref,
            )
            pr.add_to_labels(self.label)
            logger.info(f"Created PR: {pr.html_url}")
